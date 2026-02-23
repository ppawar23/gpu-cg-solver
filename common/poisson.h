/**
 * =============================================================================
 * poisson.h — 2D and 3D Poisson CSR Matrix Generator
 * =============================================================================
 *
 * Owner: Payal (code owner + verification + reporting)
 *
 * Generates the sparse matrix from finite-difference discretization of:
 *
 *     −∇²u = f     on the unit square (2D) or unit cube (3D)
 *                   with Dirichlet (zero) boundary conditions
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 2D Case — 5-point stencil on N×N interior grid
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   For interior point (i,j), the stencil is:
 *
 *                -1
 *           -1    4   -1            ×  (1/h²),  h = 1/(N+1)
 *                -1
 *
 *   We store the h²-scaled version (diagonal = 4, off-diagonal = -1)
 *   to avoid floating-point scaling issues.
 *
 *   Matrix dimension: N² × N²
 *   NNZ: ≤ 5N²  (boundary rows have fewer neighbors)
 *   Row ordering: lexicographic — row = i*N + j   for i,j ∈ [0, N-1]
 *
 *   Target sizes from proposal:  N ∈ {512, 1024, 2048, 4096, 8192}
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 3D Case — 7-point stencil on N×N×N interior grid
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   Diagonal = 6, off-diagonals = -1 (six neighbors: ±x, ±y, ±z)
 *
 *   Matrix dimension: N³ × N³
 *   NNZ: ≤ 7N³
 *   Row ordering: lexicographic — row = i*N² + j*N + k
 *
 *   Target sizes from proposal:  N ∈ {64, 96, 128, 160, 192, 256}
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Properties (both cases):
 *   ✓ Symmetric Positive Definite (SPD) → CG guaranteed to converge
 *   ✓ Deterministic — same N always gives identical matrix
 *   ✓ Diagonally dominant → numerically stable
 *   ✓ Condition number O(N²) → iteration count grows with N
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * We also generate a right-hand-side (RHS) vector via Method of Manufactured
 * Solutions (MMS):  pick a known u_exact, compute b = A * u_exact.
 * This lets us verify the solver's answer against ground truth.
 *
 * =============================================================================
 */

#ifndef POISSON_H
#define POISSON_H

#include "csr.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * generate_poisson_2d — Build 2D Poisson matrix (5-point stencil).
 *
 * @param N      Interior grid points per dimension. Matrix is N²×N².
 * @param[out] A Filled CSR matrix.
 *
 * Algorithm:
 *   Pass 1 — count nnz per row (to size row_ptr).
 *   Pass 2 — fill col_idx and values.
 *
 * Within each row, entries are stored in sorted column order
 * (required by cuSPARSE and good for cache on GPU).
 */
inline void generate_poisson_2d(int N, CSRMatrix& A) {
    int64_t total_rows = (int64_t)N * N;
    A.nrows = (int32_t)total_rows;
    A.ncols = (int32_t)total_rows;

    /*
     * Upper bound on nnz: each of N² rows has at most 5 entries.
     * Actual nnz is slightly less due to boundary rows missing neighbors.
     * We'll count exactly, then resize.
     */

    /* ── Pass 1: count exact nnz ── */
    A.row_ptr.resize(total_rows + 1);
    int64_t nnz_count = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int entries = 1; /* diagonal (always present) */
            if (i > 0)     entries++; /* neighbor above: row (i-1)*N + j */
            if (i < N - 1) entries++; /* neighbor below: row (i+1)*N + j */
            if (j > 0)     entries++; /* neighbor left:  row i*N + (j-1) */
            if (j < N - 1) entries++; /* neighbor right: row i*N + (j+1) */

            int row = i * N + j;
            A.row_ptr[row] = (int32_t)nnz_count;
            nnz_count += entries;
        }
    }
    A.row_ptr[total_rows] = (int32_t)nnz_count;
    A.nnz = nnz_count;

    /* ── Pass 2: fill values and col_idx (sorted column order) ── */
    A.col_idx.resize(nnz_count);
    A.values.resize(nnz_count);

    int64_t idx = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int row = i * N + j;

            /*
             * Insert entries in ascending column order:
             *   (i-1,j)  →  col = (i-1)*N + j        [if i > 0]
             *   (i,j-1)  →  col = i*N + (j-1)        [if j > 0]
             *   (i,j)    →  col = i*N + j             [diagonal, always]
             *   (i,j+1)  →  col = i*N + (j+1)        [if j < N-1]
             *   (i+1,j)  →  col = (i+1)*N + j        [if i < N-1]
             */

            /* Neighbor above */
            if (i > 0) {
                A.col_idx[idx] = (i - 1) * N + j;
                A.values[idx]  = -1.0;
                idx++;
            }

            /* Neighbor left */
            if (j > 0) {
                A.col_idx[idx] = i * N + (j - 1);
                A.values[idx]  = -1.0;
                idx++;
            }

            /* Diagonal */
            A.col_idx[idx] = row;
            A.values[idx]  = 4.0;
            idx++;

            /* Neighbor right */
            if (j < N - 1) {
                A.col_idx[idx] = i * N + (j + 1);
                A.values[idx]  = -1.0;
                idx++;
            }

            /* Neighbor below */
            if (i < N - 1) {
                A.col_idx[idx] = (i + 1) * N + j;
                A.values[idx]  = -1.0;
                idx++;
            }
        }
    }
}

/**
 * generate_poisson_3d — Build 3D Poisson matrix (7-point stencil).
 *
 * @param N      Interior grid points per dimension. Matrix is N³×N³.
 * @param[out] A Filled CSR matrix.
 *
 * Same approach as 2D but with 6 neighbors (±i, ±j, ±k) and diagonal = 6.
 * Row ordering: row = i*N² + j*N + k   for i,j,k ∈ [0, N-1].
 */
inline void generate_poisson_3d(int N, CSRMatrix& A) {
    int64_t total_rows = (int64_t)N * N * N;
    A.nrows = (int32_t)total_rows;
    A.ncols = (int32_t)total_rows;

    /* ── Pass 1: count exact nnz ── */
    A.row_ptr.resize(total_rows + 1);
    int64_t nnz_count = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int entries = 1; /* diagonal */
                if (i > 0)     entries++;   /* -x neighbor */
                if (i < N - 1) entries++;   /* +x neighbor */
                if (j > 0)     entries++;   /* -y neighbor */
                if (j < N - 1) entries++;   /* +y neighbor */
                if (k > 0)     entries++;   /* -z neighbor */
                if (k < N - 1) entries++;   /* +z neighbor */

                int row = i * N * N + j * N + k;
                A.row_ptr[row] = (int32_t)nnz_count;
                nnz_count += entries;
            }
        }
    }
    A.row_ptr[total_rows] = (int32_t)nnz_count;
    A.nnz = nnz_count;

    /* ── Pass 2: fill values and col_idx (sorted column order) ── */
    A.col_idx.resize(nnz_count);
    A.values.resize(nnz_count);

    int64_t idx = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                /*
                 * 7-point stencil neighbors in sorted column order:
                 *
                 *   (i-1, j,   k  )  col = (i-1)*N² + j*N + k      [-x]
                 *   (i,   j-1, k  )  col = i*N²     + (j-1)*N + k  [-y]
                 *   (i,   j,   k-1)  col = i*N²     + j*N + (k-1)  [-z]
                 *   (i,   j,   k  )  col = i*N²     + j*N + k      [diag]
                 *   (i,   j,   k+1)  col = i*N²     + j*N + (k+1)  [+z]
                 *   (i,   j+1, k  )  col = i*N²     + (j+1)*N + k  [+y]
                 *   (i+1, j,   k  )  col = (i+1)*N² + j*N + k      [+x]
                 */

                /* -x neighbor */
                if (i > 0) {
                    A.col_idx[idx] = (i - 1) * N * N + j * N + k;
                    A.values[idx]  = -1.0;
                    idx++;
                }

                /* -y neighbor */
                if (j > 0) {
                    A.col_idx[idx] = i * N * N + (j - 1) * N + k;
                    A.values[idx]  = -1.0;
                    idx++;
                }

                /* -z neighbor */
                if (k > 0) {
                    A.col_idx[idx] = i * N * N + j * N + (k - 1);
                    A.values[idx]  = -1.0;
                    idx++;
                }

                /* Diagonal */
                A.col_idx[idx] = i * N * N + j * N + k;
                A.values[idx]  = 6.0;
                idx++;

                /* +z neighbor */
                if (k < N - 1) {
                    A.col_idx[idx] = i * N * N + j * N + (k + 1);
                    A.values[idx]  = -1.0;
                    idx++;
                }

                /* +y neighbor */
                if (j < N - 1) {
                    A.col_idx[idx] = i * N * N + (j + 1) * N + k;
                    A.values[idx]  = -1.0;
                    idx++;
                }

                /* +x neighbor */
                if (i < N - 1) {
                    A.col_idx[idx] = (i + 1) * N * N + j * N + k;
                    A.values[idx]  = -1.0;
                    idx++;
                }
            }
        }
    }
}

/**
 * =============================================================================
 * Method of Manufactured Solutions (MMS) — RHS generation
 * =============================================================================
 *
 * We pick a known exact solution u_exact and compute b = A * u_exact.
 * Then when CG solves Ax = b, we can check ||x - u_exact|| directly.
 *
 * Chosen exact solution:
 *   2D:  u(x,y)   = sin(π·x) · sin(π·y)
 *   3D:  u(x,y,z) = sin(π·x) · sin(π·y) · sin(π·z)
 *
 * where (x,y,z) are physical coordinates on the unit domain.
 * Grid point (i,j) maps to x = (i+1)·h, y = (j+1)·h, h = 1/(N+1).
 *
 * This choice is smooth, satisfies the Dirichlet BCs (sin(0)=sin(π·1)=0
 * at boundaries), and gives a well-conditioned problem.
 * =============================================================================
 */

/**
 * generate_exact_and_rhs_2d — Compute u_exact and b = A * u_exact for 2D.
 *
 * @param N           Grid dimension (same as used in generate_poisson_2d).
 * @param A           The CSR matrix (already generated).
 * @param[out] u_exact  Known solution vector (size N²).
 * @param[out] rhs      Right-hand side b = A * u_exact (size N²).
 */
inline void generate_exact_and_rhs_2d(int N, const CSRMatrix& A,
                                       std::vector<double>& u_exact,
                                       std::vector<double>& rhs) {
    int64_t n = (int64_t)N * N;
    double h = 1.0 / (N + 1);  /* grid spacing */

    u_exact.resize(n);
    rhs.resize(n, 0.0);

    /* Compute u_exact at each interior grid point */
    for (int i = 0; i < N; ++i) {
        double x = (i + 1) * h;        /* physical x-coordinate */
        for (int j = 0; j < N; ++j) {
            double y = (j + 1) * h;    /* physical y-coordinate */
            int row = i * N + j;
            u_exact[row] = sin(M_PI * x) * sin(M_PI * y);
        }
    }

    /* Compute rhs = A * u_exact  (standard CSR SpMV on CPU) */
    for (int32_t i = 0; i < A.nrows; ++i) {
        double sum = 0.0;
        for (int32_t jj = A.row_ptr[i]; jj < A.row_ptr[i + 1]; ++jj) {
            sum += A.values[jj] * u_exact[A.col_idx[jj]];
        }
        rhs[i] = sum;
    }
}

/**
 * generate_exact_and_rhs_3d — Compute u_exact and b = A * u_exact for 3D.
 *
 * Same idea as 2D but with three spatial dimensions.
 */
inline void generate_exact_and_rhs_3d(int N, const CSRMatrix& A,
                                       std::vector<double>& u_exact,
                                       std::vector<double>& rhs) {
    int64_t n = (int64_t)N * N * N;
    double h = 1.0 / (N + 1);

    u_exact.resize(n);
    rhs.resize(n, 0.0);

    for (int i = 0; i < N; ++i) {
        double x = (i + 1) * h;
        for (int j = 0; j < N; ++j) {
            double y = (j + 1) * h;
            for (int k = 0; k < N; ++k) {
                double z = (k + 1) * h;
                int row = i * N * N + j * N + k;
                u_exact[row] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }

    for (int32_t i = 0; i < A.nrows; ++i) {
        double sum = 0.0;
        for (int32_t jj = A.row_ptr[i]; jj < A.row_ptr[i + 1]; ++jj) {
            sum += A.values[jj] * u_exact[A.col_idx[jj]];
        }
        rhs[i] = sum;
    }
}

#endif /* POISSON_H */
