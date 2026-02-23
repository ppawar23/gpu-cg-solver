/**
 * =============================================================================
 * cpu_cg.h — CPU Reference Conjugate Gradient Solver
 * =============================================================================
 *
 * Owner: Payal (code owner + verification + reporting)
 *
 * This is the "ground truth" CG implementation. It runs on any platform
 * (Mac, Linux, Windows) with no CUDA dependency. Its purpose:
 *
 *   1. Verify that the GPU solver produces correct results.
 *   2. Provide reference iteration counts and residual histories.
 *   3. Serve as a CPU timing baseline for speedup calculations.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * CG Algorithm (unpreconditioned, standard formulation)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Given: SPD matrix A, right-hand side b, initial guess x₀ = 0
 * Goal:  Find x such that Ax = b
 *
 *   r₀ = b - A·x₀        (residual)
 *   p₀ = r₀               (search direction)
 *   ρ₀ = rᵀ·r             (residual norm squared)
 *
 *   for k = 0, 1, 2, ... until convergence:
 *       q  = A · p         ← SpMV  (dominates runtime)
 *       α  = ρ / (pᵀ·q)   ← dot product
 *       x  = x + α·p      ← axpy
 *       r  = r - α·q      ← axpy
 *       ρ_new = rᵀ·r      ← dot product
 *
 *       if √(ρ_new) / ||b|| < tol:  CONVERGED
 *
 *       β  = ρ_new / ρ
 *       p  = r + β·p      ← axpy
 *       ρ  = ρ_new
 *
 * Convergence criterion: relative residual  ||rₖ|| / ||b|| < tol
 * Default tolerance: 1e-8  (matches proposal Section 3.2)
 * Max iterations: 50000    (hard cap, should never be reached for Poisson)
 *
 * Reference: Saad, "Iterative Methods for Sparse Linear Systems", Ch. 6
 *            Barrett et al., "Templates for the Solution of Linear Systems"
 *
 * =============================================================================
 */

#ifndef CPU_CG_H
#define CPU_CG_H

#include "csr.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <string>

/**
 * CG result struct — returned by the solver with everything needed
 * for correctness checking and performance comparison.
 */
struct CGResult {
    std::vector<double> x;          /* solution vector                      */
    int    iterations;              /* iterations to convergence            */
    double final_rel_residual;      /* ||r_final|| / ||b||                  */
    double final_abs_error;         /* ||x - x_exact||_inf  (if available)  */
    double total_time_ms;           /* wall-clock time in milliseconds      */
    bool   converged;               /* true if tol was reached              */
};

/* ──────────────────────────────────────────────────────────────────────── */
/*  Helper: CPU SpMV   y = A * x                                          */
/* ──────────────────────────────────────────────────────────────────────── */
/**
 * cpu_spmv — Sparse matrix-vector multiply on the CPU.
 *
 * This is the textbook CSR SpMV: for each row i, accumulate
 * sum of A.values[j] * x[A.col_idx[j]] over j in [row_ptr[i], row_ptr[i+1]).
 *
 * Not optimized — purely for correctness reference.
 */
inline void cpu_spmv(const CSRMatrix& A,
                     const std::vector<double>& x,
                     std::vector<double>& y) {
    for (int32_t i = 0; i < A.nrows; ++i) {
        double sum = 0.0;
        for (int32_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.values[j] * x[A.col_idx[j]];
        }
        y[i] = sum;
    }
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Helper: dot product                                                    */
/* ──────────────────────────────────────────────────────────────────────── */
inline double cpu_dot(const std::vector<double>& a,
                      const std::vector<double>& b,
                      int32_t n) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Helper: axpy   y = y + alpha * x                                       */
/* ──────────────────────────────────────────────────────────────────────── */
inline void cpu_axpy(std::vector<double>& y,
                     double alpha,
                     const std::vector<double>& x,
                     int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Main solver: CPU CG                                                    */
/* ──────────────────────────────────────────────────────────────────────── */
/**
 * cpu_cg_solve — Run unpreconditioned CG on the CPU.
 *
 * @param A         SPD matrix in CSR format.
 * @param b         Right-hand side vector.
 * @param tol       Relative residual tolerance (default 1e-8).
 * @param max_iter  Maximum iterations (default 50000).
 * @param verbose   Print iteration log every 100 steps.
 * @return          CGResult with solution, timing, and convergence info.
 */
inline CGResult cpu_cg_solve(const CSRMatrix& A,
                             const std::vector<double>& b,
                             double tol = 1e-8,
                             int max_iter = 50000,
                             bool verbose = false) {
    int32_t n = A.nrows;
    CGResult result;
    result.converged = false;
    result.iterations = 0;

    /* Allocate working vectors */
    result.x.assign(n, 0.0);       /* x₀ = 0  (zero initial guess)        */
    std::vector<double> r(n);       /* residual                             */
    std::vector<double> p(n);       /* search direction                     */
    std::vector<double> q(n);       /* q = A·p  (SpMV result)              */

    /*
     * Initialize:
     *   r₀ = b - A·x₀ = b   (since x₀ = 0)
     *   p₀ = r₀
     */
    for (int32_t i = 0; i < n; ++i) {
        r[i] = b[i];
        p[i] = b[i];
    }

    double rho = cpu_dot(r, r, n);              /* ρ = rᵀr                 */
    double bnorm = sqrt(cpu_dot(b, b, n));      /* ||b|| for relative tol  */

    /* Guard against zero RHS (trivial solution x=0) */
    if (bnorm < 1e-15) {
        result.converged = true;
        result.final_rel_residual = 0.0;
        result.total_time_ms = 0.0;
        return result;
    }

    /* ── CG iteration loop ── */
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iter; ++iter) {

        /* q = A · p                          [SpMV — dominant cost]      */
        cpu_spmv(A, p, q);

        /* α = ρ / (pᵀ·q)                    [dot product]               */
        double pq = cpu_dot(p, q, n);
        double alpha = rho / pq;

        /* x = x + α·p                       [axpy]                      */
        cpu_axpy(result.x, alpha, p, n);

        /* r = r - α·q                       [axpy]                      */
        cpu_axpy(r, -alpha, q, n);

        /* ρ_new = rᵀ·r                      [dot product]               */
        double rho_new = cpu_dot(r, r, n);

        /* Check convergence: ||r|| / ||b|| < tol */
        double rel_res = sqrt(rho_new) / bnorm;
        result.iterations = iter + 1;
        result.final_rel_residual = rel_res;

        if (verbose && (iter % 100 == 0 || rel_res < tol)) {
            std::cout << "  CPU CG iter " << iter + 1
                      << ": rel_residual = " << rel_res << "\n";
        }

        if (rel_res < tol) {
            result.converged = true;
            break;
        }

        /* β = ρ_new / ρ */
        double beta = rho_new / rho;

        /* p = r + β·p                       [axpy-like update]          */
        for (int32_t i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rho = rho_new;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.total_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

/**
 * compute_abs_error — ||x_computed - x_exact||_inf
 *
 * Used after CG to see how close we got to the manufactured solution.
 * The infinity norm (max absolute difference) is the strictest check.
 */
inline double compute_abs_error(const std::vector<double>& x_computed,
                                const std::vector<double>& x_exact) {
    double max_err = 0.0;
    for (size_t i = 0; i < x_computed.size(); ++i) {
        double err = fabs(x_computed[i] - x_exact[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

#endif /* CPU_CG_H */
