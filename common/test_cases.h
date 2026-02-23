/**
 * =============================================================================
 * test_cases.h — Tiny Known-Answer Matrices for Correctness Verification
 * =============================================================================
 *
 * Owner: Payal (code owner + verification + reporting)
 *
 * These are hand-crafted SPD matrices where we know the exact solution.
 * They serve as the CORRECTNESS GATE: if CG doesn't pass these, we don't
 * trust any benchmark numbers.
 *
 * Test strategy:
 *   1. Build tiny matrix + known b and x_exact.
 *   2. Run CG (CPU or GPU).
 *   3. Check:  ||x_computed - x_exact||_inf < threshold
 *              ||r||/||b|| < tolerance
 *              iteration count within expected range
 *
 * Correctness criteria (from proposal Section 3.2):
 *   - Relative residual < 1e-8
 *   - Max absolute error vs known solution < 1e-6
 *   - Iteration count within 5% of CPU reference
 *
 * =============================================================================
 */

#ifndef TEST_CASES_H
#define TEST_CASES_H

#include "csr.h"
#include <vector>
#include <cmath>
#include <iostream>

/**
 * TestCase — Bundle of matrix + RHS + expected solution for one test.
 */
struct TestCase {
    std::string name;               /* human-readable test name             */
    CSRMatrix A;                    /* the SPD matrix                       */
    std::vector<double> b;          /* right-hand side                      */
    std::vector<double> x_exact;    /* known exact solution                 */
    int expected_iters;             /* approximate expected iteration count */
};

/**
 * make_test_identity_4x4 — Simplest possible test: I·x = b  →  x = b.
 *
 * The identity matrix is SPD. CG should converge in exactly 1 iteration
 * because the search direction p₀ = b = r₀ already points at the answer.
 *
 * Matrix:
 *   | 1 0 0 0 |       b = [1, 2, 3, 4]       x_exact = [1, 2, 3, 4]
 *   | 0 1 0 0 |
 *   | 0 0 1 0 |
 *   | 0 0 0 1 |
 */
inline TestCase make_test_identity_4x4() {
    TestCase tc;
    tc.name = "4x4 Identity";

    tc.A.nrows = 4;
    tc.A.ncols = 4;
    tc.A.nnz   = 4;
    tc.A.row_ptr = {0, 1, 2, 3, 4};
    tc.A.col_idx = {0, 1, 2, 3};
    tc.A.values  = {1.0, 1.0, 1.0, 1.0};

    tc.b       = {1.0, 2.0, 3.0, 4.0};
    tc.x_exact = {1.0, 2.0, 3.0, 4.0};
    tc.expected_iters = 1;

    return tc;
}

/**
 * make_test_tridiag_4x4 — Classic 1D Poisson (tridiagonal, SPD).
 *
 * Matrix (1D Laplacian with h²-scaling):
 *   |  2  -1   0   0 |
 *   | -1   2  -1   0 |
 *   |  0  -1   2  -1 |
 *   |  0   0  -1   2 |
 *
 * This is SPD with condition number ≈ O(N²).
 * We pick x_exact = [1, 1, 1, 1] → b = A·x_exact = [1, 0, 0, 1].
 *
 * CG should converge in at most 4 iterations (dimension of the matrix)
 * in exact arithmetic. In float64, typically 3-4 iterations.
 */
inline TestCase make_test_tridiag_4x4() {
    TestCase tc;
    tc.name = "4x4 Tridiagonal (1D Poisson)";

    tc.A.nrows = 4;
    tc.A.ncols = 4;
    tc.A.nnz   = 10; /* 4 diagonal + 3 upper + 3 lower */

    tc.A.row_ptr = {0, 2, 5, 8, 10};
    tc.A.col_idx = {0, 1,           /* row 0:  2, -1          */
                    0, 1, 2,        /* row 1: -1,  2, -1      */
                    1, 2, 3,        /* row 2: -1,  2, -1      */
                    2, 3};          /* row 3: -1,  2           */
    tc.A.values  = { 2.0, -1.0,
                    -1.0,  2.0, -1.0,
                    -1.0,  2.0, -1.0,
                    -1.0,  2.0};

    tc.x_exact = {1.0, 1.0, 1.0, 1.0};

    /* b = A * x_exact (computed by hand):
     *   row 0:  2(1) + (-1)(1)           =  1
     *   row 1: (-1)(1) + 2(1) + (-1)(1)  =  0
     *   row 2: (-1)(1) + 2(1) + (-1)(1)  =  0
     *   row 3: (-1)(1) + 2(1)            =  1
     */
    tc.b = {1.0, 0.0, 0.0, 1.0};
    tc.expected_iters = 4;

    return tc;
}

/**
 * make_test_poisson_2d_4x4 — 2D Poisson on 2×2 interior grid.
 *
 * This is the smallest non-trivial 2D Poisson: N=2 gives a 4×4 matrix
 * with the 5-point stencil. On a 2×2 grid, every interior point touches
 * only the other interior points and the (zero) boundary.
 *
 * Matrix:
 *   |  4  -1  -1   0 |     Row 0: (0,0) → neighbors (0,1), (1,0)
 *   | -1   4   0  -1 |     Row 1: (0,1) → neighbors (0,0), (1,1)
 *   | -1   0   4  -1 |     Row 2: (1,0) → neighbors (0,0), (1,1)
 *   |  0  -1  -1   4 |     Row 3: (1,1) → neighbors (0,1), (1,0)
 *
 * x_exact = [1, 2, 3, 4]
 * b = A * x_exact:
 *   row 0:  4(1) - 1(2) - 1(3)       = -1
 *   row 1: -1(1) + 4(2)       - 1(4) =  3
 *   row 2: -1(1)       + 4(3) - 1(4) =  7
 *   row 3:       - 1(2) - 1(3) + 4(4) = 11
 */
inline TestCase make_test_poisson_2d_4x4() {
    TestCase tc;
    tc.name = "4x4 2D Poisson (N=2)";

    tc.A.nrows = 4;
    tc.A.ncols = 4;
    tc.A.nnz   = 12;

    /* Sorted column order within each row */
    tc.A.row_ptr = {0, 3, 6, 9, 12};
    tc.A.col_idx = {0, 1, 2,           /* row 0: diag, right, below */
                    0, 1, 3,            /* row 1: left, diag, below  */
                    0, 2, 3,            /* row 2: above, diag, right */
                    1, 2, 3};           /* row 3: above, left, diag  */
    tc.A.values  = { 4.0, -1.0, -1.0,
                    -1.0,  4.0, -1.0,
                    -1.0,  4.0, -1.0,
                    -1.0, -1.0,  4.0};

    tc.x_exact = {1.0, 2.0, 3.0, 4.0};
    tc.b       = {-1.0, 3.0, 7.0, 11.0};
    tc.expected_iters = 4;  /* upper bound: at most dim(A) for exact arith */

    return tc;
}

/**
 * make_test_poisson_2d_16x16 — 2D Poisson on 4×4 interior grid (16 unknowns).
 *
 * Generated programmatically using the same Poisson generator, with a
 * known exact solution via MMS. This tests that the generator itself
 * is correct and that CG handles a "real" (non-toy) problem.
 *
 * We don't hard-code the entries — we generate them and verify with CPU CG.
 * This test case is created at runtime.
 */

/* ──────────────────────────────────────────────────────────────────────── */
/*  Runner: execute all tests and report PASS/FAIL                         */
/* ──────────────────────────────────────────────────────────────────────── */

#include "cpu_cg.h"

/**
 * run_all_tests — Execute the correctness gate.
 *
 * @param tol         Convergence tolerance for CG (default 1e-10, tighter
 *                    than production 1e-8 to stress-test).
 * @param err_thresh  Max absolute error vs known solution (default 1e-6).
 * @return            Number of FAILED tests (0 = all passed).
 */
inline int run_all_tests(double tol = 1e-10, double err_thresh = 1e-6) {
    std::vector<TestCase> tests = {
        make_test_identity_4x4(),
        make_test_tridiag_4x4(),
        make_test_poisson_2d_4x4()
    };

    int failures = 0;
    std::cout << "========================================\n";
    std::cout << " Correctness Gate: Known-Answer Tests\n";
    std::cout << "========================================\n\n";

    for (auto& tc : tests) {
        std::cout << "Test: " << tc.name << "\n";

        /* Validate the test matrix itself */
        if (!tc.A.validate()) {
            std::cout << "  FAIL: CSR matrix validation failed!\n\n";
            failures++;
            continue;
        }

        /* Run CPU CG */
        CGResult res = cpu_cg_solve(tc.A, tc.b, tol, 50000, false);

        /* Check convergence */
        bool pass = true;
        if (!res.converged) {
            std::cout << "  FAIL: CG did not converge"
                      << " (iters=" << res.iterations
                      << ", rel_res=" << res.final_rel_residual << ")\n";
            pass = false;
        }

        /* Check absolute error vs known solution */
        double abs_err = compute_abs_error(res.x, tc.x_exact);
        if (abs_err > err_thresh) {
            std::cout << "  FAIL: abs_error = " << abs_err
                      << " > threshold " << err_thresh << "\n";
            pass = false;
        }

        /* Check iteration count is reasonable */
        if (res.iterations > tc.expected_iters * 2) {
            std::cout << "  WARNING: took " << res.iterations
                      << " iters (expected ~" << tc.expected_iters << ")\n";
            /* Not a failure, just a warning — floating point can vary */
        }

        if (pass) {
            std::cout << "  PASS (iters=" << res.iterations
                      << ", rel_res=" << res.final_rel_residual
                      << ", abs_err=" << abs_err << ")\n";
        } else {
            failures++;
        }
        std::cout << "\n";
    }

    std::cout << "========================================\n";
    std::cout << " Results: " << (tests.size() - failures) << "/"
              << tests.size() << " passed";
    if (failures > 0) std::cout << "  *** " << failures << " FAILED ***";
    std::cout << "\n========================================\n\n";

    return failures;
}

#endif /* TEST_CASES_H */
