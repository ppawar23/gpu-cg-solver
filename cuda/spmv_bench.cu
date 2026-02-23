/**
 * =============================================================================
 * spmv_bench.cu — SpMV-Only Microbenchmark
 * =============================================================================
 *
 * Owner: Payal (code owner + verification + reporting)
 * Tuning/Profiling: Avah (RTX 3080 + MX250)
 *
 * Purpose: Time SpMV in isolation (no CG loop) so we can measure pure
 * kernel throughput without solver overhead. This lets us:
 *
 *   1. Compare row-per-thread vs warp-per-row vs cuSPARSE fairly.
 *   2. Sweep block sizes (64, 128, 256, 512) for each kernel.
 *   3. Report achieved GB/s and bandwidth efficiency.
 *   4. Feed into Nsight Compute profiling sessions.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -arch=sm_86 -o spmv_bench cuda/spmv_bench.cu \
 *        -lcusparse -I common/
 *   (Use -arch=sm_89 for RTX 4090, sm_86 for RTX 3080, sm_61 for MX250)
 *
 * Usage:
 *   ./spmv_bench --dim 2 --N 1024                    # default sweep
 *   ./spmv_bench --dim 3 --N 128 --kernel warp       # specific kernel
 *   ./spmv_bench --dim 2 --N 512 --block 128         # specific block size
 *   ./spmv_bench --sweep                             # full sweep (all combos)
 *
 * Output: CSV to stdout. Redirect:
 *   ./spmv_bench --sweep > results/rtx3080/spmv_bench.csv
 *
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>

/* Project headers (shared, no CUDA dependency) */
#include "../common/csr.h"
#include "../common/poisson.h"

/* Our custom SpMV kernels */
#include "spmv_kernels.cuh"

/* ──────────────────────────────────────────────────────────────────────── */
/*  Error checking macro — wraps every CUDA call                           */
/* ──────────────────────────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " — " << cudaGetErrorString(err) << "\n";         \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUSPARSE_CHECK(call)                                                \
    do {                                                                    \
        cusparseStatus_t err = (call);                                      \
        if (err != CUSPARSE_STATUS_SUCCESS) {                               \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__\
                      << " — code " << err << "\n";                        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* ──────────────────────────────────────────────────────────────────────── */
/*  GPU info helper — prints device name and peak bandwidth                */
/* ──────────────────────────────────────────────────────────────────────── */
struct GPUInfo {
    std::string name;
    double peak_bw_gbps;    /* theoretical peak memory bandwidth in GB/s */
};

GPUInfo get_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    GPUInfo info;
    info.name = prop.name;

    /*
     * Peak memory bandwidth = memoryClockRate * memoryBusWidth * 2 / 8
     *   memoryClockRate is in kHz
     *   memoryBusWidth is in bits
     *   *2 for DDR (double data rate)
     *   /8 to convert bits to bytes
     *   /1e6 to get GB/s
     */
    info.peak_bw_gbps = (double)prop.memoryClockRate * 1e3
                       * (double)prop.memoryBusWidth / 8.0
                       * 2.0 / 1e9;

    std::cerr << "GPU: " << info.name
              << " | Peak BW: " << std::fixed << std::setprecision(1)
              << info.peak_bw_gbps << " GB/s\n";

    return info;
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Upload CSR matrix and vectors to GPU                                   */
/* ──────────────────────────────────────────────────────────────────────── */
struct DeviceCSR {
    int32_t* d_row_ptr;
    int32_t* d_col_idx;
    double*  d_values;
    double*  d_x;       /* input vector  */
    double*  d_y;       /* output vector */
    int32_t  nrows;
    int64_t  nnz;
};

DeviceCSR upload_matrix(const CSRMatrix& A) {
    DeviceCSR d;
    d.nrows = A.nrows;
    d.nnz   = A.nnz;

    /* Allocate device memory */
    CUDA_CHECK(cudaMalloc(&d.d_row_ptr, (A.nrows + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.d_col_idx, A.nnz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d.d_values,  A.nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.d_x,       A.nrows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.d_y,       A.nrows * sizeof(double)));

    /* Copy matrix data host → device */
    CUDA_CHECK(cudaMemcpy(d.d_row_ptr, A.row_ptr.data(),
                          (A.nrows + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_col_idx, A.col_idx.data(),
                          A.nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_values,  A.values.data(),
                          A.nnz * sizeof(double), cudaMemcpyHostToDevice));

    /* Initialize x = 1.0 (arbitrary, we only care about timing, not correctness here) */
    std::vector<double> ones(A.nrows, 1.0);
    CUDA_CHECK(cudaMemcpy(d.d_x, ones.data(),
                          A.nrows * sizeof(double), cudaMemcpyHostToDevice));

    /* Zero out y */
    CUDA_CHECK(cudaMemset(d.d_y, 0, A.nrows * sizeof(double)));

    return d;
}

void free_device_csr(DeviceCSR& d) {
    cudaFree(d.d_row_ptr);
    cudaFree(d.d_col_idx);
    cudaFree(d.d_values);
    cudaFree(d.d_x);
    cudaFree(d.d_y);
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Benchmark one kernel configuration                                     */
/* ──────────────────────────────────────────────────────────────────────── */

/**
 * bench_kernel — Time a single SpMV kernel configuration.
 *
 * Runs the kernel `warmup` times (not timed) to fill caches and stabilize
 * GPU clocks, then `repeats` times with CUDA event timing.
 *
 * @param kernel_name  "row_per_thread", "warp_per_row", or "cusparse"
 * @param d            Device CSR data.
 * @param block_size   Threads per block (ignored for cuSPARSE).
 * @param A            Host matrix (for byte count calculation).
 * @param gpu_info     GPU info (for bandwidth efficiency).
 * @param dim_label    "2D" or "3D"
 * @param grid_n       Grid N used to generate the matrix.
 * @param warmup       Warmup iterations (default 10).
 * @param repeats      Timed iterations (default 100).
 */
void bench_kernel(const std::string& kernel_name,
                  DeviceCSR& d, int block_size,
                  const CSRMatrix& A, const GPUInfo& gpu_info,
                  const std::string& dim_label, int grid_n,
                  int warmup = 10, int repeats = 100)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* ── cuSPARSE setup (only if needed) ── */
    cusparseHandle_t     cusparse_handle = nullptr;
    cusparseSpMatDescr_t mat_descr       = nullptr;
    cusparseDnVecDescr_t vec_x_descr     = nullptr;
    cusparseDnVecDescr_t vec_y_descr     = nullptr;
    void*                cusparse_buf    = nullptr;
    size_t               buf_size        = 0;
    double               alpha_one       = 1.0;
    double               beta_zero       = 0.0;

    if (kernel_name == "cusparse") {
        CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));

        CUSPARSE_CHECK(cusparseCreateCsr(
            &mat_descr, d.nrows, d.nrows, (int64_t)d.nnz,
            d.d_row_ptr, d.d_col_idx, d.d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x_descr, d.nrows, d.d_x, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y_descr, d.nrows, d.d_y, CUDA_R_64F));

        /* Query buffer size */
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_one, mat_descr, vec_x_descr, &beta_zero, vec_y_descr,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size));

        if (buf_size > 0) {
            CUDA_CHECK(cudaMalloc(&cusparse_buf, buf_size));
        }
    }

    /* ── Warmup runs (not timed) ── */
    for (int i = 0; i < warmup; ++i) {
        if (kernel_name == "row_per_thread") {
            launch_spmv_row_per_thread(d.nrows, d.d_row_ptr, d.d_col_idx,
                                       d.d_values, d.d_x, d.d_y, block_size);
        } else if (kernel_name == "warp_per_row") {
            launch_spmv_warp_per_row(d.nrows, d.d_row_ptr, d.d_col_idx,
                                     d.d_values, d.d_x, d.d_y, block_size);
        } else {
            CUSPARSE_CHECK(cusparseSpMV(
                cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha_one, mat_descr, vec_x_descr, &beta_zero, vec_y_descr,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cusparse_buf));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── Timed runs ── */
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeats; ++i) {
        if (kernel_name == "row_per_thread") {
            launch_spmv_row_per_thread(d.nrows, d.d_row_ptr, d.d_col_idx,
                                       d.d_values, d.d_x, d.d_y, block_size);
        } else if (kernel_name == "warp_per_row") {
            launch_spmv_warp_per_row(d.nrows, d.d_row_ptr, d.d_col_idx,
                                     d.d_values, d.d_x, d.d_y, block_size);
        } else {
            CUSPARSE_CHECK(cusparseSpMV(
                cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha_one, mat_descr, vec_x_descr, &beta_zero, vec_y_descr,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cusparse_buf));
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    double avg_ms = (double)total_ms / repeats;

    /* ── Compute bandwidth metrics ── */
    double bytes_per_spmv = A.spmv_bytes();
    double gbps = (bytes_per_spmv / avg_ms) * 1e-6;   /* ms→s and B→GB cancel */
    double efficiency = (gbps / gpu_info.peak_bw_gbps) * 100.0;

    /* ── Output CSV row ── */
    std::cout << gpu_info.name << ","
              << dim_label << "," << grid_n << ","
              << A.nrows << "," << A.nnz << ","
              << kernel_name << "," << block_size << ","
              << std::fixed << std::setprecision(4) << avg_ms << ","
              << std::fixed << std::setprecision(2) << gbps << ","
              << std::fixed << std::setprecision(2) << efficiency << "\n";

    /* ── Cleanup ── */
    if (kernel_name == "cusparse") {
        if (cusparse_buf) cudaFree(cusparse_buf);
        cusparseDestroyDnVec(vec_y_descr);
        cusparseDestroyDnVec(vec_x_descr);
        cusparseDestroySpMat(mat_descr);
        cusparseDestroy(cusparse_handle);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  CSV header for microbenchmark output                                   */
/* ──────────────────────────────────────────────────────────────────────── */
void print_bench_csv_header() {
    std::cout << "gpu,problem_dim,grid_n,rows,nnz,"
              << "kernel_variant,block_size,"
              << "avg_spmv_ms,spmv_gbps,bw_efficiency_pct\n";
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Main: parse args and run benchmarks                                    */
/* ──────────────────────────────────────────────────────────────────────── */
int main(int argc, char* argv[]) {
    /* Default parameters */
    int    dim        = 2;       /* 2D or 3D */
    int    N          = 256;     /* grid size */
    std::string kernel = "all";  /* "row_per_thread", "warp_per_row", "cusparse", "all" */
    int    block      = 0;       /* 0 = sweep all block sizes */
    bool   sweep_all  = false;   /* full sweep mode */

    /* Parse command line */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc)
            dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc)
            kernel = argv[++i];
        else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc)
            block = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sweep") == 0)
            sweep_all = true;
        else if (strcmp(argv[i], "--help") == 0) {
            std::cerr << "Usage: spmv_bench [--dim 2|3] [--N size] "
                      << "[--kernel row_per_thread|warp_per_row|cusparse|all] "
                      << "[--block size] [--sweep]\n";
            return 0;
        }
    }

    GPUInfo gpu_info = get_gpu_info();
    print_bench_csv_header();

    /* Block sizes to sweep (from proposal: {64, 128, 256, 512}) */
    std::vector<int> block_sizes = {64, 128, 256, 512};
    if (block > 0) block_sizes = {block};

    /* Kernels to test */
    std::vector<std::string> kernels;
    if (kernel == "all" || kernel == "row_per_thread") kernels.push_back("row_per_thread");
    if (kernel == "all" || kernel == "warp_per_row")   kernels.push_back("warp_per_row");
    if (kernel == "all" || kernel == "cusparse")       kernels.push_back("cusparse");

    if (sweep_all) {
        /*
         * Full sweep: all problem sizes × all kernels × all block sizes.
         * This is what Avah runs to produce the complete benchmark dataset.
         */
        std::vector<int> sizes_2d = {64, 128, 256, 512, 1024, 2048};
        std::vector<int> sizes_3d = {32, 48, 64, 96, 128};

        /* 2D sweep */
        for (int n : sizes_2d) {
            CSRMatrix A;
            generate_poisson_2d(n, A);
            A.print_info("2D N=" + std::to_string(n));
            DeviceCSR d = upload_matrix(A);

            for (const auto& k : kernels) {
                if (k == "cusparse") {
                    bench_kernel(k, d, 0, A, gpu_info, "2D", n);
                } else {
                    for (int bs : block_sizes) {
                        bench_kernel(k, d, bs, A, gpu_info, "2D", n);
                    }
                }
            }
            free_device_csr(d);
        }

        /* 3D sweep */
        for (int n : sizes_3d) {
            CSRMatrix A;
            generate_poisson_3d(n, A);

            /* Check if matrix fits in GPU memory (rough estimate) */
            double mem_mb = A.spmv_bytes() / (1024.0 * 1024.0);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            double vram_mb = prop.totalGlobalMem / (1024.0 * 1024.0);
            if (mem_mb * 3 > vram_mb) {
                /* Need ~3x matrix size for A + x + y + workspace */
                std::cerr << "SKIP 3D N=" << n
                          << " (needs ~" << (int)(mem_mb * 3)
                          << " MB, have " << (int)vram_mb << " MB)\n";
                continue;
            }

            A.print_info("3D N=" + std::to_string(n));
            DeviceCSR d = upload_matrix(A);

            for (const auto& k : kernels) {
                if (k == "cusparse") {
                    bench_kernel(k, d, 0, A, gpu_info, "3D", n);
                } else {
                    for (int bs : block_sizes) {
                        bench_kernel(k, d, bs, A, gpu_info, "3D", n);
                    }
                }
            }
            free_device_csr(d);
        }

    } else {
        /* Single problem size */
        CSRMatrix A;
        std::string dim_label;
        if (dim == 2) {
            generate_poisson_2d(N, A);
            dim_label = "2D";
        } else {
            generate_poisson_3d(N, A);
            dim_label = "3D";
        }
        A.print_info(dim_label + " N=" + std::to_string(N));
        DeviceCSR d = upload_matrix(A);

        for (const auto& k : kernels) {
            if (k == "cusparse") {
                bench_kernel(k, d, 0, A, gpu_info, dim_label, N);
            } else {
                for (int bs : block_sizes) {
                    bench_kernel(k, d, bs, A, gpu_info, dim_label, N);
                }
            }
        }
        free_device_csr(d);
    }

    std::cerr << "SpMV benchmark complete.\n";
    return 0;
}
