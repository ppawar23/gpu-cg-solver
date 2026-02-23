/**
 * =============================================================================
 * spmv_kernels.cuh — Custom CSR SpMV CUDA Kernels
 * =============================================================================
 *
 * Owner: Payal (code owner + verification + reporting)
 * Tuning/Profiling: Avah (RTX 3080 + MX250)
 *
 * Two SpMV strategies for CSR format, as specified in the proposal:
 *
 *   Kernel A: Row-per-thread — one CUDA thread processes one entire row.
 *   Kernel B: Warp-per-row   — 32 threads cooperatively process one row.
 *
 * Both compute  y = A * x  where A is in CSR format.
 *
 * Integration rule (from proposal Section 5): only SpMV changes between
 * variants; dot/axpy remain constant (cuBLAS). This ensures clean
 * attribution of performance differences to the SpMV kernel alone.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Kernel A: ROW-PER-THREAD
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Strategy: Thread i processes all nonzeros in row i sequentially.
 *
 *   Pros:
 *     - Simple, low overhead
 *     - Good for very short rows (few nnz per row)
 *     - Memory access to values[] and col_idx[] is coalesced across
 *       adjacent threads IF rows have similar length
 *
 *   Cons:
 *     - No intra-row parallelism (wasted SIMD lanes for long rows)
 *     - Load imbalance if row lengths vary widely
 *     - Random access to x[] via col_idx[] causes cache pressure
 *
 *   For Poisson: rows have 3-5 entries (2D) or 4-7 entries (3D),
 *   so this kernel should work reasonably well since rows are short
 *   and uniform in length.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Kernel B: WARP-PER-ROW
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Strategy: 32 threads in a warp cooperatively process one row.
 *   Thread lane t handles entries at positions row_start + t, row_start + 32 + t, ...
 *   Then a warp-level reduction (__shfl_down_sync) sums the partial products.
 *
 *   Pros:
 *     - Better memory coalescing for x[] access (32 threads read 32 x-values)
 *     - Intra-row parallelism for longer rows
 *     - Warp-level reduction is fast (no shared memory needed)
 *
 *   Cons:
 *     - Overhead for very short rows (most lanes idle for 5-element rows)
 *     - More complex launch configuration (grid = nrows, not nrows/blockDim)
 *
 *   For Poisson: rows have only 5-7 entries, so most warp lanes will be idle.
 *   We expect this kernel to be SLOWER than row-per-thread for Poisson stencils.
 *   That's a valid and interesting result — it shows kernel choice depends on
 *   the sparsity pattern, not just the hardware.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * References:
 *   - Bell & Garland, "Implementing Sparse Matrix-Vector Multiplication
 *     on Throughput-Oriented Processors" (2009) — canonical CSR SpMV strategies.
 *   - NVIDIA cuSPARSE documentation — baseline we compare against.
 * =============================================================================
 */

#ifndef SPMV_KERNELS_CUH
#define SPMV_KERNELS_CUH

#include <cuda_runtime.h>
#include <cstdint>

/* ========================================================================= */
/*  Kernel A: ROW-PER-THREAD                                                  */
/* ========================================================================= */

/**
 * spmv_csr_row_per_thread — One thread per row.
 *
 * @param nrows     Number of rows in the matrix.
 * @param row_ptr   CSR row pointer array (device memory, size nrows+1).
 * @param col_idx   CSR column index array (device memory, size nnz).
 * @param values    CSR values array (device memory, size nnz).
 * @param x         Input vector (device memory, size ncols).
 * @param y         Output vector (device memory, size nrows). Overwritten.
 *
 * Launch config:
 *   Grid:  (nrows + blockDim - 1) / blockDim
 *   Block: blockDim (tunable: 64, 128, 256, 512)
 *
 * Memory access pattern:
 *   - row_ptr[i] and row_ptr[i+1]: coalesced (adjacent threads read adjacent ints)
 *   - values[j], col_idx[j]: approximately coalesced for uniform row lengths
 *   - x[col_idx[j]]: RANDOM ACCESS — this is the bandwidth bottleneck
 *   - y[i]: coalesced write
 */
__global__ void spmv_csr_row_per_thread(
    int32_t        nrows,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const double*  __restrict__ values,
    const double*  __restrict__ x,
    double*        __restrict__ y)
{
    /*
     * Global thread ID = row index.
     * Each thread handles exactly one row of the matrix.
     */
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nrows) {
        /* Get the range of nonzeros for this row */
        int row_start = row_ptr[row];
        int row_end   = row_ptr[row + 1];

        /*
         * Accumulate the dot product:  y[row] = Σ values[j] * x[col_idx[j]]
         *
         * This loop runs 3-7 times for Poisson stencils (very short).
         * The compiler should unroll this efficiently.
         */
        double sum = 0.0;
        for (int j = row_start; j < row_end; ++j) {
            sum += values[j] * x[col_idx[j]];
        }

        y[row] = sum;
    }
}


/* ========================================================================= */
/*  Kernel B: WARP-PER-ROW                                                    */
/* ========================================================================= */

/**
 * spmv_csr_warp_per_row — 32 threads (one warp) per row.
 *
 * @param nrows     Number of rows.
 * @param row_ptr   CSR row pointer (device, size nrows+1).
 * @param col_idx   CSR column indices (device, size nnz).
 * @param values    CSR values (device, size nnz).
 * @param x         Input vector (device, size ncols).
 * @param y         Output vector (device, size nrows). Overwritten.
 *
 * Launch config:
 *   Each warp (32 threads) handles one row.
 *   Warps per block = blockDim / 32.
 *   Grid = (nrows + warps_per_block - 1) / warps_per_block
 *
 * Example: blockDim=256 → 8 warps/block → grid handles 8 rows/block.
 *
 * Warp-level reduction:
 *   After each thread accumulates its partial sum, we use __shfl_down_sync
 *   to reduce across the warp WITHOUT shared memory. This is faster than
 *   shared-memory reduction for small reductions (log2(32) = 5 steps).
 */
__global__ void spmv_csr_warp_per_row(
    int32_t        nrows,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const double*  __restrict__ values,
    const double*  __restrict__ x,
    double*        __restrict__ y)
{
    /*
     * Determine which row this warp is responsible for.
     *
     * Global thread ID:          tid = blockIdx.x * blockDim.x + threadIdx.x
     * Warp ID within the grid:   warp_id = tid / 32
     * Lane within the warp:      lane = tid % 32    (0..31)
     *
     * Row assignment:  row = warp_id
     */
    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;    /* which warp (== which row) */
    int lane    = tid % 32;    /* position within the warp  */

    if (warp_id < nrows) {
        int row_start = row_ptr[warp_id];
        int row_end   = row_ptr[warp_id + 1];

        /*
         * Each lane processes a strided subset of the row's nonzeros:
         *   lane 0: positions row_start+0, row_start+32, row_start+64, ...
         *   lane 1: positions row_start+1, row_start+33, row_start+65, ...
         *   ...
         *   lane 31: positions row_start+31, row_start+63, ...
         *
         * For Poisson rows with 5-7 entries, only lanes 0-6 will do work.
         * Lanes 7-31 will have sum = 0. This is the overhead of warp-per-row
         * for short rows — most lanes are idle.
         */
        double sum = 0.0;
        for (int j = row_start + lane; j < row_end; j += 32) {
            sum += values[j] * x[col_idx[j]];
        }

        /*
         * Warp-level reduction using shuffle instructions.
         *
         * __shfl_down_sync(mask, val, offset):
         *   Each thread receives the value from the thread `offset` lanes ahead.
         *   In 5 steps (offset = 16, 8, 4, 2, 1), lane 0 accumulates the
         *   total sum for the entire row.
         *
         * 0xFFFFFFFF = all 32 lanes participate in the shuffle.
         *
         * This replaces shared memory reduction — fewer instructions, no
         * bank conflicts, no __syncthreads() needed.
         */
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

        /* Only lane 0 writes the final result for this row */
        if (lane == 0) {
            y[warp_id] = sum;
        }
    }
}


/* ========================================================================= */
/*  Launcher helpers (called from cg_solver.cu and spmv_bench.cu)            */
/* ========================================================================= */

/**
 * launch_spmv_row_per_thread — Configure and launch Kernel A.
 *
 * @param nrows      Number of rows.
 * @param d_row_ptr  Device pointer to row_ptr.
 * @param d_col_idx  Device pointer to col_idx.
 * @param d_values   Device pointer to values.
 * @param d_x        Device pointer to input vector.
 * @param d_y        Device pointer to output vector.
 * @param block_size Threads per block (tunable: 64, 128, 256, 512).
 * @param stream     CUDA stream (default 0).
 */
inline void launch_spmv_row_per_thread(
    int32_t nrows,
    const int32_t* d_row_ptr, const int32_t* d_col_idx,
    const double* d_values, const double* d_x, double* d_y,
    int block_size = 256, cudaStream_t stream = 0)
{
    int grid_size = (nrows + block_size - 1) / block_size;
    spmv_csr_row_per_thread<<<grid_size, block_size, 0, stream>>>(
        nrows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
}

/**
 * launch_spmv_warp_per_row — Configure and launch Kernel B.
 *
 * Note the grid calculation: we need nrows *warps*, not nrows *threads*.
 * Each warp is 32 threads, so:
 *   total_threads_needed = nrows * 32
 *   grid = (total_threads_needed + block_size - 1) / block_size
 */
inline void launch_spmv_warp_per_row(
    int32_t nrows,
    const int32_t* d_row_ptr, const int32_t* d_col_idx,
    const double* d_values, const double* d_x, double* d_y,
    int block_size = 256, cudaStream_t stream = 0)
{
    int total_threads = nrows * 32;
    int grid_size = (total_threads + block_size - 1) / block_size;
    spmv_csr_warp_per_row<<<grid_size, block_size, 0, stream>>>(
        nrows, d_row_ptr, d_col_idx, d_values, d_x, d_y);
}

#endif /* SPMV_KERNELS_CUH */
