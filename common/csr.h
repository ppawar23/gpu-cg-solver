/**
 * =============================================================================
 * csr.h — Compressed Sparse Row (CSR) Matrix Format
 * =============================================================================
 *
 * Shared header used by ALL parts of the project: CPU solver, GPU solver,
 * Poisson generator, and test cases. This file has NO CUDA dependency so it
 * compiles on Mac, Linux, and Windows with any C++17 compiler.
 *
 * CSR Storage Layout
 * ------------------
 * For a matrix with `nrows` rows and `nnz` nonzero entries:
 *
 *   row_ptr[nrows+1]  — row_ptr[i] is the index into values/col_idx where
 *                        row i begins.  row_ptr[nrows] == nnz.
 *   col_idx[nnz]      — column index for each nonzero.
 *   values[nnz]       — the nonzero value (double precision).
 *
 * Example (3×3):
 *       | 1  0  2 |       values   = [1, 2, 3, 4, 5]
 *       | 0  3  0 |       col_idx  = [0, 2, 1, 0, 2]
 *       | 4  0  5 |       row_ptr  = [0, 2, 3, 5]
 *
 * Memory footprint per matrix:
 *   nnz*12 bytes + (nrows+1)*4 bytes   (double values + int32 indices)
 *
 * For our largest case (3D Poisson N=256):
 *   rows ≈ 16.7M,  nnz ≈ 117M  →  ≈ 1.34 GB
 *
 * =============================================================================
 */

#ifndef CSR_H
#define CSR_H

#include <vector>
#include <cstdint>
#include <iostream>
#include <string>
#include <cmath>

/**
 * CSRMatrix — Host-side CSR storage.
 *
 * int32_t indices handle up to ~2 billion rows/nnz (well beyond our problems).
 * double values give CG the precision it needs for convergence to 1e-8.
 */
struct CSRMatrix {
    int32_t nrows;      /* number of rows (== ncols for square Poisson)       */
    int32_t ncols;      /* number of columns                                  */
    int64_t nnz;        /* total nonzeros (int64 for safety on large 3D)      */

    std::vector<int32_t> row_ptr;   /* size: nrows + 1                        */
    std::vector<int32_t> col_idx;   /* size: nnz                              */
    std::vector<double>  values;    /* size: nnz                              */

    /**
     * spmv_bytes() — Total bytes moved during one SpMV  y = A*x.
     *
     * This is the "naive" byte count (standard HPC convention):
     *   Reads:  row_ptr + col_idx + values + x (assume all cache misses)
     *   Writes: y vector
     *
     * Used to compute effective bandwidth:  GB/s = spmv_bytes() / time
     */
    double spmv_bytes() const {
        double read_b =
            (double)(nrows + 1) * sizeof(int32_t)     /* row_ptr              */
          + (double)nnz         * sizeof(int32_t)      /* col_idx              */
          + (double)nnz         * sizeof(double)        /* values               */
          + (double)nrows       * sizeof(double);       /* x vector (worst case)*/
        double write_b =
            (double)nrows       * sizeof(double);       /* y vector             */
        return read_b + write_b;
    }

    /** Print a one-line summary for logging. */
    void print_info(const std::string& label = "CSRMatrix") const {
        std::cout << "[" << label << "] "
                  << nrows << " x " << ncols
                  << ", nnz = " << nnz
                  << ", avg nnz/row = " << (nrows > 0 ? (double)nnz / nrows : 0)
                  << ", memory ~ " << (spmv_bytes() / (1024.0 * 1024.0)) << " MB\n";
    }

    /** Sanity check — catches corrupted generation before GPU upload. */
    bool validate() const {
        if ((int32_t)row_ptr.size() != nrows + 1)  return false;
        if ((int64_t)col_idx.size() != nnz)         return false;
        if ((int64_t)values.size()  != nnz)          return false;
        if (row_ptr[0] != 0)                         return false;
        if (row_ptr[nrows] != (int32_t)nnz)          return false;
        for (int32_t i = 0; i < nrows; ++i)
            if (row_ptr[i] > row_ptr[i + 1]) return false;
        for (int64_t j = 0; j < nnz; ++j)
            if (col_idx[j] < 0 || col_idx[j] >= ncols) return false;
        return true;
    }
};

#endif /* CSR_H */
