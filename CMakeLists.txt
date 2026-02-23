# =============================================================================
# CMakeLists.txt — Build system for GPU CG Solver
# =============================================================================
#
# This builds two CUDA executables:
#   1. cg_solver  — Full CG solver (Deen's end-to-end driver)
#   2. spmv_bench — SpMV-only microbenchmark (Avah's profiling harness)
#
# And one CPU-only executable:
#   3. cpu_main   — CPU reference solver + correctness tests (Payal, no CUDA)
#
# Build instructions:
# -------------------
#   mkdir build && cd build
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"    # RTX 4090
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES="86"    # RTX 3080
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES="61"    # MX250 (Pascal)
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES="75"    # MX250 (Turing variant)
#   cmake --build . --config Release
#
# Or with nvcc directly (see individual .cu files for commands).
#
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(gpu_cg_solver LANGUAGES CXX CUDA)

# ── C++ and CUDA standards ──
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Default CUDA architecture if not specified ──
# User should override with -DCMAKE_CUDA_ARCHITECTURES=XX
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    message(WARNING "CUDA arch not set, defaulting to sm_86 (RTX 3080). "
                    "Set -DCMAKE_CUDA_ARCHITECTURES=89 for RTX 4090.")
endif()

# ── Include paths ──
include_directories(${CMAKE_SOURCE_DIR}/common)

# ── Find CUDA libraries ──
find_package(CUDAToolkit REQUIRED)

# ── Target 1: cg_solver (Deen's GPU CG driver) ──
add_executable(cg_solver cuda/cg_solver.cu)
target_link_libraries(cg_solver
    CUDA::cusparse
    CUDA::cublas
)
target_compile_options(cg_solver PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math>
)

# ── Target 2: spmv_bench (Avah's microbenchmark) ──
add_executable(spmv_bench cuda/spmv_bench.cu)
target_link_libraries(spmv_bench
    CUDA::cusparse
)
target_compile_options(spmv_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math>
)

# ── Target 3: cpu_main (Payal's CPU-only driver, no CUDA) ──
add_executable(cpu_main common/cpu_main.cpp)
# No CUDA linkage needed — pure C++
target_compile_options(cpu_main PRIVATE -O2)
