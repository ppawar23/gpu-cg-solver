# GPU-Accelerated Conjugate Gradient Solver

**Team:** Deen Grey, Avah Afshari, Payal Pawar

GPU-accelerated Conjugate Gradient (CG) solver for sparse SPD systems
arising from 2D/3D Poisson equations, with custom SpMV kernel analysis
across NVIDIA architectures (RTX 4090 vs RTX 3080, MX250 stretch goal).

---

## Repository Structure

```
gpu-cg-solver/
├── common/                # Shared code — NO CUDA dependency (compiles on Mac)
│   ├── csr.h              # CSR matrix format definition (Payal)
│   ├── poisson.h          # 2D/3D Poisson matrix generator + MMS RHS (Payal)
│   ├── cpu_cg.h           # CPU reference CG solver (Payal)
│   ├── test_cases.h       # Known-answer test cases + correctness runner (Payal)
│   └── cpu_main.cpp       # CPU-only driver (Payal)
├── cuda/                  # CUDA code — requires NVIDIA GPU + CUDA Toolkit
│   ├── spmv_kernels.cuh   # Custom SpMV kernels (Payal; tuned by Avah)
│   ├── spmv_bench.cu      # SpMV-only microbenchmark (Payal; run by Avah)
│   └── cg_solver.cu       # Full GPU CG solver with per-phase timing (Payal; run by Deen)
├── scripts/
│   ├── run_all.sh         # Full experiment sweep automation (Payal)
│   ├── plot.py            # Generate all project charts (Payal)
│   └── validate.py        # CSV schema + sanity checker (Payal)
├── results/               # Experiment output (CSVs)
│   ├── rtx4090/
│   ├── rtx3080/
│   └── mx250/
├── report/
│   └── figures/           # Generated charts
├── CMakeLists.txt         # Build system
└── README.md              # This file
```

---

## Quick Start

### 1. Payal (Mac — CPU only, no CUDA needed)

```bash
# Build CPU-only driver
cd gpu-cg-solver
g++ -std=c++17 -O2 -o cpu_main common/cpu_main.cpp -lm

# Run correctness tests
./cpu_main

# Run CPU benchmarks (CSV to stdout)
./cpu_main --bench > results/cpu_baseline.csv

# Generate plots (after Deen/Avah commit their CSVs)
pip install matplotlib pandas seaborn
python scripts/plot.py --dir results/ --out report/figures/

# Validate CSVs
python scripts/validate.py results/
```

### 2. Deen (Windows, RTX 4090)

```bash
# Build with CMake
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build . --config Release
cd ..

# Or build directly with nvcc
nvcc -std=c++17 -O2 -arch=sm_89 -o cg_solver cuda/cg_solver.cu -lcusparse -lcublas -I common/

# Run correctness gate first!
./cg_solver --correctness

# Single run
./cg_solver --dim 2 --N 1024 --kernel cusparse

# Full sweep (all sizes × kernels × block sizes)
./cg_solver --sweep > results/rtx4090/cg_full.csv
```

### 3. Avah (Windows, RTX 3080)

```bash
# Build — RTX 3080
mkdir build && cd build 
mkdir results\rtx3080 
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release 
cd .. 
cmake --build . --config Release  

# Correctness check first (always do this before sweeps) 
.\Release\cg_solver.exe --correctness 

# Single kernel test 
.\Release\spmv_bench.exe --dim 2 --N 1024 --kernel row_per_thread --block 256 

# Full CG sweep 
.\Release\cg_solver.exe --sweep > results\rtx3080\cg_full.csv 

# SpMV sweep 
.\Release\spmv_bench.exe --sweep > results\rtx3080\spmv_bench.csv 
```

### For Prolfiling using RTX 3080
```bash
# For creating the exe for profiling
nvcc -ccbin [Path to Microsoft MSVC x64 cl.exe] -arch=sm_86 -o [exe name] [file name].cu -lcublas -lcusparse 

# For running the profiler
ncu -o [exe name]Report ./[exe name] 
```
---

## Command Reference

### cg_solver

| Flag | Description | Default |
|------|-------------|---------|
| `--dim 2\|3` | Problem dimension | 2 |
| `--N <int>` | Grid size per dimension | 256 |
| `--kernel <name>` | `cusparse`, `row_per_thread`, `warp_per_row` | cusparse |
| `--block <int>` | Threads per block | 256 |
| `--correctness` | Run correctness gate only | — |
| `--sweep` | Full experiment sweep | — |

### spmv_bench

| Flag | Description | Default |
|------|-------------|---------|
| `--dim 2\|3` | Problem dimension | 2 |
| `--N <int>` | Grid size | 256 |
| `--kernel <name>` | `row_per_thread`, `warp_per_row`, `cusparse`, `all` | all |
| `--block <int>` | Specific block size (0 = sweep all) | 0 |
| `--sweep` | Full sweep (all sizes × kernels × blocks) | — |

---

## CSV Output Schema

### CG Solver (cg_full.csv)
```
gpu, problem_dim, grid_n, rows, nnz,
kernel_variant, block_size,
total_time_ms, spmv_time_ms, dot_time_ms, axpy_time_ms, overhead_ms,
iterations, rel_residual, abs_error,
spmv_gbps, bw_efficiency_pct
```

### SpMV Bench (spmv_bench.csv)
```
gpu, problem_dim, grid_n, rows, nnz,
kernel_variant, block_size,
avg_spmv_ms, spmv_gbps, bw_efficiency_pct
```

---

## Correctness Criteria

From the proposal:
- Relative residual `||r|| / ||b|| < 1e-8`
- Max absolute error vs known solution `< 1e-6`
- Iteration count within 5% of CPU reference

**Always run `--correctness` before `--sweep`.**

---

## Nsight Profiling (Avah)

```bash
# Nsight Systems — timeline view
nsys profile --output spmv_timeline ./spmv_bench --dim 2 --N 1024 --kernel row_per_thread --block 256

# Nsight Compute — kernel metrics (one kernel at a time)
ncu --set full --output spmv_ncu ./spmv_bench --dim 2 --N 1024 --kernel row_per_thread --block 256

# Open in Nsight GUI and capture roofline, occupancy, throughput screenshots
```

---

## References

- Saad, *Iterative Methods for Sparse Linear Systems* — CG algorithm, convergence theory
- Barrett et al., *Templates for the Solution of Linear Systems* — CG template
- Hestenes & Stiefel (1952) — Original CG formulation
- Bell & Garland (2009), *Implementing SpMV on Throughput-Oriented Processors* — CSR kernel strategies
- NVIDIA cuSPARSE, cuBLAS, Nsight documentation
