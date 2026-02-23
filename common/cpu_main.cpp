/**
 * =============================================================================
 * cpu_main.cpp — CPU-Only Driver (No CUDA Required)
 * =============================================================================
 *
 * Owner: Payal (code owner + verification + reporting)
 *
 * This program does three things:
 *   1. Run the correctness gate (tiny known-answer tests).
 *   2. Run CPU CG on 2D/3D Poisson problems and output CSV timing data.
 *   3. Validate that the Poisson generator + MMS produces correct answers.
 *
 * Build (Mac/Linux/Windows, no CUDA):
 *   g++ -std=c++17 -O2 -o cpu_main common/cpu_main.cpp -lm
 *
 * Usage:
 *   ./cpu_main                    # run correctness tests only
 *   ./cpu_main --bench            # run correctness + CPU benchmarks
 *   ./cpu_main --bench --verbose  # with per-iteration logging
 *
 * Output CSV goes to stdout (redirect to file):
 *   ./cpu_main --bench > results/cpu_baseline.csv
 *
 * =============================================================================
 */

#include "csr.h"
#include "poisson.h"
#include "cpu_cg.h"
#include "test_cases.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

/**
 * print_csv_header — Standard CSV schema matching the project spec.
 *
 * All team members write this same format so Payal's plotting scripts
 * work without manual cleanup.
 */
void print_csv_header() {
    std::cout << "gpu,problem_dim,grid_n,rows,nnz,"
              << "kernel_variant,block_size,"
              << "total_time_ms,spmv_time_ms,dot_time_ms,axpy_time_ms,overhead_ms,"
              << "iterations,rel_residual,abs_error,"
              << "spmv_gbps,bw_efficiency_pct\n";
}

/**
 * run_benchmark_2d — Generate 2D Poisson, solve with CPU CG, output CSV row.
 *
 * @param N       Grid size (N×N interior points → N² unknowns).
 * @param verbose Print per-iteration log.
 */
void run_benchmark_2d(int N, bool verbose) {
    /* Generate matrix */
    CSRMatrix A;
    generate_poisson_2d(N, A);
    A.print_info("2D Poisson N=" + std::to_string(N));

    /* Generate RHS with known solution */
    std::vector<double> u_exact, rhs;
    generate_exact_and_rhs_2d(N, A, u_exact, rhs);

    /* Solve */
    CGResult res = cpu_cg_solve(A, rhs, 1e-8, 50000, verbose);

    /* Compute absolute error vs manufactured solution */
    double abs_err = compute_abs_error(res.x, u_exact);

    /* CPU doesn't have per-phase timing breakdown — report total only.
     * spmv/dot/axpy fields are -1 to indicate "not measured separately".
     * spmv_gbps and bw_efficiency are not meaningful for CPU (no peak BW).
     */
    std::cout << "CPU,2D," << N << "," << A.nrows << "," << A.nnz << ","
              << "cpu_reference,0,"
              << std::fixed << std::setprecision(3) << res.total_time_ms << ","
              << "-1,-1,-1,-1,"
              << res.iterations << ","
              << std::scientific << std::setprecision(6)
              << res.final_rel_residual << ","
              << abs_err << ","
              << "-1,-1\n";
}

/**
 * run_benchmark_3d — Same as 2D but for 3D Poisson (N×N×N → N³ unknowns).
 */
void run_benchmark_3d(int N, bool verbose) {
    CSRMatrix A;
    generate_poisson_3d(N, A);
    A.print_info("3D Poisson N=" + std::to_string(N));

    std::vector<double> u_exact, rhs;
    generate_exact_and_rhs_3d(N, A, u_exact, rhs);

    CGResult res = cpu_cg_solve(A, rhs, 1e-8, 50000, verbose);
    double abs_err = compute_abs_error(res.x, u_exact);

    std::cout << "CPU,3D," << N << "," << A.nrows << "," << A.nnz << ","
              << "cpu_reference,0,"
              << std::fixed << std::setprecision(3) << res.total_time_ms << ","
              << "-1,-1,-1,-1,"
              << res.iterations << ","
              << std::scientific << std::setprecision(6)
              << res.final_rel_residual << ","
              << abs_err << ","
              << "-1,-1\n";
}

int main(int argc, char* argv[]) {
    /* ── Parse command-line flags ── */
    bool do_bench = false;
    bool verbose  = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0)   do_bench = true;
        if (strcmp(argv[i], "--verbose") == 0) verbose  = true;
    }

    /* ── Step 1: Always run correctness gate ── */
    int failures = run_all_tests();
    if (failures > 0) {
        std::cerr << "ERROR: " << failures
                  << " test(s) failed. Fix before benchmarking.\n";
        return 1;
    }

    if (!do_bench) {
        std::cout << "Pass --bench to run CPU benchmarks.\n";
        return 0;
    }

    /* ── Step 2: CPU benchmarks ── */
    std::cerr << "Running CPU benchmarks (output is CSV to stdout)...\n";
    print_csv_header();

    /* 2D Poisson — proposal sizes (skip huge ones on slow machines) */
    std::vector<int> sizes_2d = {32, 64, 128, 256, 512};
    /* Full proposal: {512, 1024, 2048, 4096, 8192}
     * We start small for CPU (it's slow). Add larger sizes if time permits.
     * GPU will handle the full range.
     */

    for (int N : sizes_2d) {
        run_benchmark_2d(N, verbose);
    }

    /* 3D Poisson — smaller sizes for CPU (N³ grows fast) */
    std::vector<int> sizes_3d = {16, 32, 48, 64};
    /* Full proposal: {64, 96, 128, 160, 192, 256}
     * CPU can realistically handle up to ~64 in 3D. GPU handles the rest.
     */

    for (int N : sizes_3d) {
        run_benchmark_3d(N, verbose);
    }

    std::cerr << "CPU benchmarks complete.\n";
    return 0;
}
