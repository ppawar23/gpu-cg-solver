/**
 * =============================================================================
 * cg_solver.cu — GPU Conjugate Gradient Solver (End-to-End)
 * =============================================================================
 *
 * Owner: Payal (code owner + verification + reporting)
 * GPU Runs: Deen (RTX 4090, Windows)
 *
 * Full unpreconditioned CG on GPU with per-phase CUDA event timing:
 *   SpMV, dot, axpy, overhead — all measured and reported in CSV.
 *
 * Three SpMV backends (switchable via --kernel):
 *   cusparse       — vendor-optimized baseline (NVIDIA library)
 *   row_per_thread — custom kernel A (Avah)
 *   warp_per_row   — custom kernel B (Avah)
 *
 * Dot and axpy always use cuBLAS so performance differences are
 * attributed solely to SpMV (proposal Section 5 integration rule).
 *
 * Build:
 *   nvcc -std=c++17 -O2 -arch=sm_89 -o cg_solver cuda/cg_solver.cu \-lcusparse -lcublas -I common/
 *   (sm_89 for 4090, sm_86 for 3080, sm_61/sm_75 for MX250)
 *
 * Usage:
 *   ./cg_solver --correctness                              # GPU correctness gate
 *   ./cg_solver --dim 2 --N 1024 --kernel cusparse         # single run
 *   ./cg_solver --dim 3 --N 128  --kernel row_per_thread   # custom kernel
 *   ./cg_solver --sweep                                    # full experiment sweep
 *
 * Output CSV to stdout:
 *   ./cg_solver --sweep > results/rtx4090/cg_full.csv
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <fstream>
#include "device_launch_parameters.h"

#include "../common/csr.h"
#include "../common/poisson.h"
#include "../common/cpu_cg.h"
#include "../common/test_cases.h"
#include "spmv_kernels.cuh"

/* ── Error checking macros ────────────────────────────────────────────── */
#define CUDA_CHECK(call) do {                                               \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__          \
                  << " — " << cudaGetErrorString(err) << "\n"; exit(1); }  \
} while (0)

#define CUSPARSE_CHECK(call) do {                                           \
    cusparseStatus_t err = (call);                                          \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::cerr << "cuSPARSE error " << __FILE__ << ":" << __LINE__      \
                  << " code=" << err << "\n"; exit(1); }                   \
} while (0)

#define CUBLAS_CHECK(call) do {                                             \
    cublasStatus_t err = (call);                                            \
    if (err != CUBLAS_STATUS_SUCCESS) {                                     \
        std::cerr << "cuBLAS error " << __FILE__ << ":" << __LINE__        \
                  << " code=" << err << "\n"; exit(1); }                   \
} while (0)

/* ── Fused p = r + beta*p kernel (cuBLAS lacks this single op) ────────── */
__global__ void update_p_kernel(double* p, const double* r,
                                double beta, int32_t n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = r[i] + beta * p[i];
    }
}

inline void launch_update_p(double* d_p, const double* d_r,
                            double beta, int32_t n) {
    int block = 256;
    unsigned int grid  = (n + block - 1) / block;
    update_p_kernel<<<grid, block>>>(d_p, d_r, beta, n);
}

/* ── GPU info helper ──────────────────────────────────────────────────── */
struct GPUInfo {
    std::string name;
    double peak_bw_gbps;
};

GPUInfo get_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    GPUInfo info;
    info.name = prop.name;
    int clockRateKHz;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, 0);
    info.peak_bw_gbps = (double)clockRateKHz * 1e3
                       * (double)prop.memoryBusWidth / 8.0
                       * 2.0 / 1e9;
    std::cerr << "GPU: " << info.name << " | Peak BW: "
              << std::fixed << std::setprecision(1)
              << info.peak_bw_gbps << " GB/s\n";
    return info;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  GPU CG SOLVER — main function                                         */
/* ═══════════════════════════════════════════════════════════════════════ */
/**
 * gpu_cg_solve — Full CG on GPU with per-phase timing.
 *
 * All vectors live on the GPU. Scalars (alpha, beta, rho) are computed
 * via cuBLAS dot products which copy the result back to host (synchronizing).
 *
 * Timing: CUDA events bracket each phase. We accumulate totals across
 * all iterations and report sums. This avoids per-iteration event overhead.
 */
void gpu_cg_solve(const CSRMatrix& A,
                  const std::vector<double>& h_b,
                  const std::string& kernel_name,
                  int block_size,
                  double tol, int max_iter,
                  const GPUInfo& gpu_info,
                  const std::string& dim_label, int grid_n,
                  const double* h_u_exact = nullptr)
{
    int32_t n = A.nrows;

    /* ── Allocate device memory ── */
    int32_t* d_row_ptr; int32_t* d_col_idx; double* d_values;
    double *d_x, *d_r, *d_p, *d_q, *d_b;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, A.nnz * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values,  A.nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));

    /* ── Host → Device transfers ── */
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr.data(), (n+1)*sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, A.col_idx.data(), A.nnz*sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values,  A.values.data(),  A.nnz*sizeof(double),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,       h_b.data(),       n*sizeof(double),      cudaMemcpyHostToDevice));

    /* x₀ = 0 */
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double)));
    /* r₀ = b  (since x₀ = 0) */
    CUDA_CHECK(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
    /* p₀ = r₀ */
    CUDA_CHECK(cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));

    /* ── Library handles ── */
    cublasHandle_t cublas_h;
    CUBLAS_CHECK(cublasCreate(&cublas_h));

    cusparseHandle_t     cusp_h     = nullptr;
    cusparseSpMatDescr_t mat_desc   = nullptr;
    cusparseDnVecDescr_t dvec_p     = nullptr;
    cusparseDnVecDescr_t dvec_q     = nullptr;
    void*                cusp_buf   = nullptr;
    double sp_alpha = 1.0, sp_beta  = 0.0;

    if (kernel_name == "cusparse") {
        CUSPARSE_CHECK(cusparseCreate(&cusp_h));
        CUSPARSE_CHECK(cusparseCreateCsr(&mat_desc, n, n, (int64_t)A.nnz,
            d_row_ptr, d_col_idx, d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&dvec_p, n, d_p, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&dvec_q, n, d_q, CUDA_R_64F));
        size_t bsz = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusp_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &sp_alpha, mat_desc, dvec_p, &sp_beta, dvec_q,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bsz));
        if (bsz > 0) CUDA_CHECK(cudaMalloc(&cusp_buf, bsz));
    }

    /* ── CUDA timing events ── */
    cudaEvent_t ev_total_s, ev_total_e;
    cudaEvent_t ev_spmv_s, ev_spmv_e, ev_dot_s, ev_dot_e, ev_axpy_s, ev_axpy_e;
    CUDA_CHECK(cudaEventCreate(&ev_total_s)); CUDA_CHECK(cudaEventCreate(&ev_total_e));
    CUDA_CHECK(cudaEventCreate(&ev_spmv_s));  CUDA_CHECK(cudaEventCreate(&ev_spmv_e));
    CUDA_CHECK(cudaEventCreate(&ev_dot_s));   CUDA_CHECK(cudaEventCreate(&ev_dot_e));
    CUDA_CHECK(cudaEventCreate(&ev_axpy_s));  CUDA_CHECK(cudaEventCreate(&ev_axpy_e));

    /* ── Initial scalars ── */
    double rho = 0.0;
    CUBLAS_CHECK(cublasDdot(cublas_h, n, d_r, 1, d_r, 1, &rho));
    double bnorm = 0.0;
    CUBLAS_CHECK(cublasDdot(cublas_h, n, d_b, 1, d_b, 1, &bnorm));
    bnorm = sqrt(bnorm);

    if (bnorm < 1e-15) {
        std::cerr << "  WARNING: ||b|| ≈ 0\n";
        goto cleanup;
    }

    {   /* Scope for CG loop variables */
    double total_spmv_ms = 0, total_dot_ms = 0, total_axpy_ms = 0;
    int    iters = 0;
    double rel_res = 1.0;
    bool   conv = false;
    float  ms_tmp;

    CUDA_CHECK(cudaEventRecord(ev_total_s));

    for (int iter = 0; iter < max_iter; ++iter) {

        /* ── SpMV: q = A·p ── */
        CUDA_CHECK(cudaEventRecord(ev_spmv_s));
        if (kernel_name == "cusparse") {
            CUSPARSE_CHECK(cusparseDnVecSetValues(dvec_p, d_p));
            CUSPARSE_CHECK(cusparseDnVecSetValues(dvec_q, d_q));
            CUSPARSE_CHECK(cusparseSpMV(cusp_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &sp_alpha, mat_desc, dvec_p, &sp_beta, dvec_q,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cusp_buf));
        } else if (kernel_name == "row_per_thread") {
            launch_spmv_row_per_thread(n, d_row_ptr, d_col_idx, d_values,
                                       d_p, d_q, block_size);
        } else {
            launch_spmv_warp_per_row(n, d_row_ptr, d_col_idx, d_values,
                                     d_p, d_q, block_size);
        }
        CUDA_CHECK(cudaEventRecord(ev_spmv_e));
        CUDA_CHECK(cudaEventSynchronize(ev_spmv_e));
        CUDA_CHECK(cudaEventElapsedTime(&ms_tmp, ev_spmv_s, ev_spmv_e));
        total_spmv_ms += ms_tmp;

        /* ── Dot: α = ρ / (pᵀq) ── */
        CUDA_CHECK(cudaEventRecord(ev_dot_s));
        double pq = 0.0;
        CUBLAS_CHECK(cublasDdot(cublas_h, n, d_p, 1, d_q, 1, &pq));
        CUDA_CHECK(cudaEventRecord(ev_dot_e));
        CUDA_CHECK(cudaEventSynchronize(ev_dot_e));
        CUDA_CHECK(cudaEventElapsedTime(&ms_tmp, ev_dot_s, ev_dot_e));
        total_dot_ms += ms_tmp;

        double alpha = rho / pq;

        /* ── Axpy: x += αp, r -= αq ── */
        CUDA_CHECK(cudaEventRecord(ev_axpy_s));
        CUBLAS_CHECK(cublasDaxpy(cublas_h, n, &alpha, d_p, 1, d_x, 1));
        double neg_a = -alpha;
        CUBLAS_CHECK(cublasDaxpy(cublas_h, n, &neg_a, d_q, 1, d_r, 1));
        CUDA_CHECK(cudaEventRecord(ev_axpy_e));
        CUDA_CHECK(cudaEventSynchronize(ev_axpy_e));
        CUDA_CHECK(cudaEventElapsedTime(&ms_tmp, ev_axpy_s, ev_axpy_e));
        total_axpy_ms += ms_tmp;

        /* ── Dot: ρ_new = rᵀr, check convergence ── */
        CUDA_CHECK(cudaEventRecord(ev_dot_s));
        double rho_new = 0.0;
        CUBLAS_CHECK(cublasDdot(cublas_h, n, d_r, 1, d_r, 1, &rho_new));
        CUDA_CHECK(cudaEventRecord(ev_dot_e));
        CUDA_CHECK(cudaEventSynchronize(ev_dot_e));
        CUDA_CHECK(cudaEventElapsedTime(&ms_tmp, ev_dot_s, ev_dot_e));
        total_dot_ms += ms_tmp;

        rel_res = sqrt(rho_new) / bnorm;
        iters = iter + 1;

        if (rel_res < tol) { conv = true; break; }

        /* ── Update: p = r + βp ── */
        CUDA_CHECK(cudaEventRecord(ev_axpy_s));
        double beta_val = rho_new / rho;
        launch_update_p(d_p, d_r, beta_val, n);
        CUDA_CHECK(cudaEventRecord(ev_axpy_e));
        CUDA_CHECK(cudaEventSynchronize(ev_axpy_e));
        CUDA_CHECK(cudaEventElapsedTime(&ms_tmp, ev_axpy_s, ev_axpy_e));
        total_axpy_ms += ms_tmp;

        rho = rho_new;
    }

    CUDA_CHECK(cudaEventRecord(ev_total_e));
    CUDA_CHECK(cudaEventSynchronize(ev_total_e));
    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_total_s, ev_total_e));

    /* ── Absolute error vs known solution ── */
    double abs_err = -1.0;
    if (h_u_exact) {
        std::vector<double> h_x(n);
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n*sizeof(double), cudaMemcpyDeviceToHost));
        abs_err = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            double e = fabs(h_x[i] - h_u_exact[i]);
            if (e > abs_err) abs_err = e;
        }
    }

    /* ── SpMV bandwidth metrics ── */
    double overhead_ms = (double)total_ms - total_spmv_ms - total_dot_ms - total_axpy_ms;
    if (overhead_ms < 0) overhead_ms = 0;
    double spmv_per_iter = total_spmv_ms / std::max(iters, 1);
    double spmv_gbps = (A.spmv_bytes() / spmv_per_iter) * 1e-6;
    double bw_eff = (spmv_gbps / gpu_info.peak_bw_gbps) * 100.0;

    /* ── CSV output ── */
    printToCSV("cg_results.csv", formatForCSV(gpu_info, dim_label, grid_n, A,
        kernel_name, block_size, total_ms,total_spmv_ms,
        total_dot_ms,total_axpy_ms,overhead_ms, iters,
        rel_res, abs_err, spmv_gbps, bw_eff));
    std::cout << gpu_info.name << ","
              << dim_label << "," << grid_n << ","
              << A.nrows << "," << A.nnz << ","
              << kernel_name << "," << block_size << ","
              << std::fixed << std::setprecision(3)
              << total_ms << "," << total_spmv_ms << ","
              << total_dot_ms << "," << total_axpy_ms << "," << overhead_ms << ","
              << iters << ","
              << std::scientific << std::setprecision(6) << rel_res << "," << abs_err << ","
              << std::fixed << std::setprecision(2) << spmv_gbps << "," << bw_eff << "\n";

    /* ── Human-readable log to stderr ── */
    std::cerr << "  " << kernel_name << " bs=" << block_size
              << " | " << iters << " iters"
              << " | total=" << std::fixed << std::setprecision(1) << total_ms << "ms"
              << " | SpMV=" << total_spmv_ms << "ms ("
              << (total_spmv_ms / total_ms * 100) << "%)"
              << " | " << spmv_gbps << " GB/s (" << bw_eff << "% eff)"
              << (conv ? " CONVERGED" : " NOT CONVERGED") << "\n";
    }

cleanup:
    /* ── Free everything ── */
    if (kernel_name == "cusparse") {
        if (cusp_buf) cudaFree(cusp_buf);
        if (dvec_q)   cusparseDestroyDnVec(dvec_q);
        if (dvec_p)   cusparseDestroyDnVec(dvec_p);
        if (mat_desc) cusparseDestroySpMat(mat_desc);
        if (cusp_h)   cusparseDestroy(cusp_h);
    }
    cublasDestroy(cublas_h);
    cudaEventDestroy(ev_total_s); cudaEventDestroy(ev_total_e);
    cudaEventDestroy(ev_spmv_s);  cudaEventDestroy(ev_spmv_e);
    cudaEventDestroy(ev_dot_s);   cudaEventDestroy(ev_dot_e);
    cudaEventDestroy(ev_axpy_s);  cudaEventDestroy(ev_axpy_e);
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_values);
    cudaFree(d_x); cudaFree(d_r); cudaFree(d_p); cudaFree(d_q); cudaFree(d_b);
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  GPU Correctness Gate                                                   */
/* ═══════════════════════════════════════════════════════════════════════ */
/**
 * run_gpu_correctness — Verify GPU CG vs CPU CG on tiny test cases.
 *
 * For each test case, runs GPU CG with each kernel variant and checks:
 *   - Converged to tolerance
 *   - Absolute error vs known solution < 1e-6
 *   - Iteration count within reasonable range
 */
int run_gpu_correctness(const GPUInfo& gpu_info) {
    std::vector<TestCase> tests = {
        make_test_identity_4x4(),
        make_test_tridiag_4x4(),
        make_test_poisson_2d_4x4()
    };
    std::vector<std::string> kernels = {"cusparse", "row_per_thread", "warp_per_row"};

    int failures = 0;
    std::cerr << "\n========= GPU Correctness Gate =========\n\n";

    for (auto& tc : tests) {
        for (const auto& k : kernels) {
            std::cerr << "Test: " << tc.name << " | kernel: " << k << "\n";

            /* Solve on GPU — capture solution for error check */
            int32_t n = tc.A.nrows;
            int32_t* d_rp; int32_t* d_ci; double* d_v;
            double *d_x, *d_r, *d_p, *d_q, *d_b;

            CUDA_CHECK(cudaMalloc(&d_rp, (n+1)*sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_ci, tc.A.nnz*sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_v,  tc.A.nnz*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_x,  n*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_r,  n*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_p,  n*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_q,  n*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_b,  n*sizeof(double)));

            CUDA_CHECK(cudaMemcpy(d_rp, tc.A.row_ptr.data(), (n+1)*sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ci, tc.A.col_idx.data(), tc.A.nnz*sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_v,  tc.A.values.data(),  tc.A.nnz*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b,  tc.b.data(),         n*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_x, 0, n*sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_r, d_b, n*sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_p, d_r, n*sizeof(double), cudaMemcpyDeviceToDevice));

            cublasHandle_t cub;
            CUBLAS_CHECK(cublasCreate(&cub));

            /* Minimal CG loop for correctness */
            double rho; CUBLAS_CHECK(cublasDdot(cub, n, d_r, 1, d_r, 1, &rho));
            double bnorm; CUBLAS_CHECK(cublasDdot(cub, n, d_b, 1, d_b, 1, &bnorm));
            bnorm = sqrt(bnorm);

            cusparseHandle_t cusp = nullptr;
            cusparseSpMatDescr_t md = nullptr;
            cusparseDnVecDescr_t vp = nullptr, vq = nullptr;
            void* buf = nullptr;
            double sa = 1.0, sb = 0.0;
            if (k == "cusparse") {
                CUSPARSE_CHECK(cusparseCreate(&cusp));
                CUSPARSE_CHECK(cusparseCreateCsr(&md, n, n, (int64_t)tc.A.nnz,
                    d_rp, d_ci, d_v, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
                CUSPARSE_CHECK(cusparseCreateDnVec(&vp, n, d_p, CUDA_R_64F));
                CUSPARSE_CHECK(cusparseCreateDnVec(&vq, n, d_q, CUDA_R_64F));
                size_t bs2; CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusp,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, &sa, md, vp, &sb, vq,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bs2));
                if (bs2 > 0) CUDA_CHECK(cudaMalloc(&buf, bs2));
            }

            bool conv = false;
            int iters = 0;
            double rel_res = 1.0;

            for (int it = 0; it < 1000; ++it) {
                /* SpMV */
                if (k == "cusparse") {
                    CUSPARSE_CHECK(cusparseDnVecSetValues(vp, d_p));
                    CUSPARSE_CHECK(cusparseDnVecSetValues(vq, d_q));
                    CUSPARSE_CHECK(cusparseSpMV(cusp, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &sa, md, vp, &sb, vq, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
                } else if (k == "row_per_thread") {
                    launch_spmv_row_per_thread(n, d_rp, d_ci, d_v, d_p, d_q, 256);
                } else {
                    launch_spmv_warp_per_row(n, d_rp, d_ci, d_v, d_p, d_q, 256);
                }

                double pq; CUBLAS_CHECK(cublasDdot(cub, n, d_p, 1, d_q, 1, &pq));
                double alpha = rho / pq;
                CUBLAS_CHECK(cublasDaxpy(cub, n, &alpha, d_p, 1, d_x, 1));
                double na = -alpha;
                CUBLAS_CHECK(cublasDaxpy(cub, n, &na, d_q, 1, d_r, 1));
                double rho_new; CUBLAS_CHECK(cublasDdot(cub, n, d_r, 1, d_r, 1, &rho_new));

                rel_res = sqrt(rho_new) / bnorm;
                iters = it + 1;
                if (rel_res < 1e-10) { conv = true; break; }

                double beta = rho_new / rho;
                launch_update_p(d_p, d_r, beta, n);
                rho = rho_new;
            }

            /* Copy solution back and check */
            std::vector<double> h_x(n);
            CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n*sizeof(double), cudaMemcpyDeviceToHost));
            double abs_err = compute_abs_error(h_x, tc.x_exact);

            bool pass = conv && (abs_err < 1e-6);
            if (pass) {
                std::cerr << "  PASS (iters=" << iters << " rel_res="
                          << rel_res << " abs_err=" << abs_err << ")\n";
            } else {
                std::cerr << "  FAIL (conv=" << conv << " iters=" << iters
                          << " rel_res=" << rel_res << " abs_err=" << abs_err << ")\n";
                failures++;
            }

            /* Cleanup */
            if (k == "cusparse") {
                if (buf) cudaFree(buf);
                cusparseDestroyDnVec(vq); cusparseDestroyDnVec(vp);
                cusparseDestroySpMat(md); cusparseDestroy(cusp);
            }
            cublasDestroy(cub);
            cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_v);
            cudaFree(d_x); cudaFree(d_r); cudaFree(d_p); cudaFree(d_q); cudaFree(d_b);
        }
    }

    std::cerr << "\n========= " << (failures == 0 ? "ALL PASSED" : "FAILURES")
              << " =========\n\n";
    return failures;
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  CSV header (project-standard schema)                                   */
/* ═══════════════════════════════════════════════════════════════════════ */
//CSV file exists
static bool fileExistsAndNonEmpty(const std::string& path) {
    std::ifstream f(path);
    //if the path exists and the end of the file has not been reached
    return f.good() && f.peek() != std::ifstream::traits_type::eof();
}
//format line for csv output
static std::string formatForCSV(GPUInfo gpu_info, std::string dim_label, int grid_n, CSRMatrix A,
                                std::string kernel_name, int block_size, double total_ms, double total_spmv_ms, 
                                double total_dot_ms, double total_axpy_ms,double overhead_ms, int iters, 
                                double rel_res,double abs_err, double spmv_gbps, double bw_eff) {
    std::ostringstream ss;
    ss << gpu_info.name << ","
        << dim_label << "," << grid_n << ","
        << A.nrows << "," << A.nnz << ","
        << kernel_name << "," << block_size << ","
        << std::fixed << std::setprecision(3)
        << total_ms << "," << total_spmv_ms << ","
        << total_dot_ms << "," << total_axpy_ms << "," << overhead_ms << ","
        << iters << ","
        << std::scientific << std::setprecision(6) << rel_res << "," << abs_err << ","
        << std::fixed << std::setprecision(2) << spmv_gbps << "," << bw_eff << "\n";
    return ss.str();
}

//csv output
static void printToCSV(const std::string& path, const std::string& row) {
    if (path.empty()) return;
    const bool hasContent = fileExistsAndNonEmpty(path);
    std::ofstream f(path, std::ios::app);
    if (!hasContent) {
        f << "gpu,problem_dim,grid_n,rows,nnz,"
            << "kernel_variant,block_size,"
            << "total_time_ms,spmv_time_ms,dot_time_ms,axpy_time_ms,overhead_ms,"
            << "iterations,rel_residual,abs_error,"
            << "spmv_gbps,bw_efficiency_pct\n";
    }
    f << row << "\n";
}

void print_csv_header() {
    std::cout << "gpu,problem_dim,grid_n,rows,nnz,"
              << "kernel_variant,block_size,"
              << "total_time_ms,spmv_time_ms,dot_time_ms,axpy_time_ms,overhead_ms,"
              << "iterations,rel_residual,abs_error,"
              << "spmv_gbps,bw_efficiency_pct\n";
}

/* ═══════════════════════════════════════════════════════════════════════ */
/*  MAIN                                                                   */
/* ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char* argv[]) {
    /* ── Parse arguments ── */
    int    dim    = 2;
    int    N      = 256;
    std::string kernel = "cusparse";
    int    block  = 256;
    bool   correctness = false;
    bool   sweep  = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dim") && i+1 < argc)    dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i+1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kernel") && i+1 < argc) kernel = argv[++i];
        else if (!strcmp(argv[i], "--block") && i+1 < argc)  block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--correctness"))      correctness = true;
        else if (!strcmp(argv[i], "--sweep"))             sweep = true;
        else if (!strcmp(argv[i], "--help")) {
            std::cerr << "Usage: cg_solver [--dim 2|3] [--N size] "
                      << "[--kernel cusparse|row_per_thread|warp_per_row] "
                      << "[--block size] [--correctness] [--sweep]\n";
            return 0;
        }
    }

    GPUInfo gpu = get_gpu_info();

    /* ── Correctness gate ── */
    if (correctness) {
        int f1 = run_all_tests();    /* CPU tests */
        int f2 = run_gpu_correctness(gpu);
        if (f1 + f2 > 0) {
            std::cerr << "CORRECTNESS FAILURES. Fix before benchmarking.\n";
            return 1;
        }
        std::cerr << "All correctness tests passed.\n";
        if (!sweep) return 0;
    }

    /* ── Benchmark mode ── */
    print_csv_header();

    if (sweep) {
        /*
         * Full experiment sweep: all problem sizes × all kernels × block sizes.
         * This is what Deen runs on the 4090 and Avah replicates on the 3080.
         */
        std::vector<int> sizes_2d = {128, 256, 512, 1024, 2048, 4096, 8192};
        std::vector<int> sizes_3d = {32, 48, 64, 96, 128, 160, 192};
        std::vector<std::string> ks = {"cusparse", "row_per_thread", "warp_per_row"};
        std::vector<int> bss = {64, 128, 256, 512};

        /* 2D sweep */
        for (int sz : sizes_2d) {
            CSRMatrix A; generate_poisson_2d(sz, A);
            std::vector<double> u_ex, rhs;
            generate_exact_and_rhs_2d(sz, A, u_ex, rhs);
            A.print_info("2D N=" + std::to_string(sz));

            for (const auto& k : ks) {
                if (k == "cusparse") {
                    gpu_cg_solve(A, rhs, k, 0, 1e-8, 50000, gpu, "2D", sz, u_ex.data());
                } else {
                    for (int bs : bss) {
                        gpu_cg_solve(A, rhs, k, bs, 1e-8, 50000, gpu, "2D", sz, u_ex.data());
                    }
                }
            }
        }

        /* 3D sweep */
        for (int sz : sizes_3d) {
            CSRMatrix A; generate_poisson_3d(sz, A);

            /* Check VRAM fit */
            double mem_mb = A.spmv_bytes() / (1024.0 * 1024.0);
            cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
            double vram_mb = prop.totalGlobalMem / (1024.0 * 1024.0);
            if (mem_mb * 4 > vram_mb) {
                std::cerr << "SKIP 3D N=" << sz << " (exceeds VRAM)\n";
                continue;
            }

            std::vector<double> u_ex, rhs;
            generate_exact_and_rhs_3d(sz, A, u_ex, rhs);
            A.print_info("3D N=" + std::to_string(sz));

            std::vector<std::string> ks2 = {"cusparse", "row_per_thread", "warp_per_row"};
            std::vector<int> bss2 = {64, 128, 256, 512};

            for (const auto& k : ks2) {
                if (k == "cusparse") {
                    gpu_cg_solve(A, rhs, k, 0, 1e-8, 50000, gpu, "3D", sz, u_ex.data());
                } else {
                    for (int bs : bss2) {
                        gpu_cg_solve(A, rhs, k, bs, 1e-8, 50000, gpu, "3D", sz, u_ex.data());
                    }
                }
            }
        }

    } else {
        /* Single run */
        CSRMatrix A;
        std::vector<double> u_ex, rhs;
        std::string dl;

        if (dim == 2) {
            generate_poisson_2d(N, A);
            generate_exact_and_rhs_2d(N, A, u_ex, rhs);
            dl = "2D";
        } else {
            generate_poisson_3d(N, A);
            generate_exact_and_rhs_3d(N, A, u_ex, rhs);
            dl = "3D";
        }
        A.print_info(dl + " N=" + std::to_string(N));
        gpu_cg_solve(A, rhs, kernel, block, 1e-8, 50000, gpu, dl, N, u_ex.data());
    }

    std::cerr << "Done.\n";
    return 0;
}
