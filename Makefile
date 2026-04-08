# Directories
SRC_DIR := src
INC_DIR := include
OBJ_DIR := build/$(BUILD_TYPE)
BIN_DIR := bin/$(BUILD_TYPE)
MAT_DIR := matrix
RES_DIR := results

# Build type (default: release)
BUILD_TYPE ?= release

# Compiler + flags
NVCC := nvcc
MPICXX := mpic++

# Detect MPI availability
HAS_MPI := $(shell which mpic++ > /dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_MPI),1)
    MPI_INCLUDES := $(shell mpic++ --showme:compile 2>/dev/null || echo "-I/usr/lib/x86_64-linux-gnu/openmpi/include")
else
    MPI_INCLUDES :=
endif

ifeq ($(BUILD_TYPE),debug)
    NVCCFLAGS := -g -G -O0 -std=c++11
else
    NVCCFLAGS := -O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true -std=c++11
endif

# Base includes and libraries
INCLUDES := -I$(INC_DIR) -I$(INC_DIR)/solvers
LDFLAGS := -lcusparse -lcublas
CUDA_LDFLAGS := -L/usr/local/cuda/lib64 -lcudart

# Sources / objets
CU_SRCS := $(shell find $(SRC_DIR) -name '*.cu')
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))

# SpMV benchmark: exclude generators, CG solver, and multi-GPU sources
CU_SPMV_SRCS := $(filter-out $(SRC_DIR)/matrix/generate_matrix.cu $(SRC_DIR)/main/cg_solver.cu $(SRC_DIR)/main/cg_solver_mgpu_stencil.cu $(SRC_DIR)/main/cg_solver_mgpu_stencil_3d.cu $(SRC_DIR)/main/cg_solver_single_gpu_3d.cu $(SRC_DIR)/main/generate_matrix_3d.cu $(SRC_DIR)/main/generate_matrix_3d_27pt.cu $(SRC_DIR)/solvers/cg_solver_mgpu_partitioned.cu $(SRC_DIR)/solvers/cg_solver_mgpu_partitioned_3d.cu $(SRC_DIR)/solvers/cg_solver_mgpu_overlap.cu $(SRC_DIR)/spmv/spmv_stencil_partitioned_halo_kernel.cu $(SRC_DIR)/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu $(SRC_DIR)/spmv/benchmark_stats_mgpu_partitioned.cu, $(CU_SRCS))
CU_SPMV_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SPMV_SRCS))

# Matrix generator (2D 5-point stencil)
CU_GEN_SRCS := $(SRC_DIR)/matrix/generate_matrix.cu $(SRC_DIR)/io/io.cu
CU_GEN_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_GEN_SRCS))

# Matrix generator 3D (3D 7-point stencil)
CU_GEN3D_SRCS := $(SRC_DIR)/main/generate_matrix_3d.cu $(SRC_DIR)/io/io.cu
CU_GEN3D_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_GEN3D_SRCS))

# Matrix generator 3D 27-point stencil
CU_GEN3D_27PT_SRCS := $(SRC_DIR)/main/generate_matrix_3d_27pt.cu $(SRC_DIR)/io/io.cu
CU_GEN3D_27PT_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_GEN3D_27PT_SRCS))

# Binaries
BIN_SPMV := $(BIN_DIR)/spmv_bench
BIN_GEN  := $(BIN_DIR)/generate_matrix
BIN_GEN3D := $(BIN_DIR)/generate_matrix_3d
BIN_GEN3D_27PT := $(BIN_DIR)/generate_matrix_3d_27pt
BIN_CG   := $(BIN_DIR)/cg_solver
BIN_MGPU_STENCIL := $(BIN_DIR)/cg_solver_mgpu_stencil
BIN_MGPU_STENCIL_3D := $(BIN_DIR)/cg_solver_mgpu_stencil_3d
BIN_SINGLE_GPU_3D := $(BIN_DIR)/cg_solver_single_gpu_3d

# CG solver: exclude generator, spmv_bench, and multi-GPU sources
CU_CG_SRCS := $(filter-out $(SRC_DIR)/matrix/generate_matrix.cu $(SRC_DIR)/main/main.cu $(SRC_DIR)/main/cg_solver_mgpu_stencil.cu $(SRC_DIR)/main/cg_solver_mgpu_stencil_3d.cu $(SRC_DIR)/main/cg_solver_single_gpu_3d.cu $(SRC_DIR)/main/generate_matrix_3d.cu $(SRC_DIR)/main/generate_matrix_3d_27pt.cu $(SRC_DIR)/solvers/cg_solver_mgpu_partitioned.cu $(SRC_DIR)/solvers/cg_solver_mgpu_partitioned_3d.cu $(SRC_DIR)/solvers/cg_solver_mgpu_overlap.cu $(SRC_DIR)/spmv/spmv_stencil_partitioned_halo_kernel.cu $(SRC_DIR)/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu $(SRC_DIR)/spmv/benchmark_stats_mgpu_partitioned.cu, $(CU_SRCS))
CU_CG_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_CG_SRCS))

# Single-GPU 3D solver
CU_SINGLE_GPU_3D_SRCS := $(SRC_DIR)/main/cg_solver_single_gpu_3d.cu $(SRC_DIR)/spmv/spmv_stencil_3d_partitioned_halo_kernel.cu $(SRC_DIR)/io/io.cu $(SRC_DIR)/spmv/spmv_cusparse_csr.cu $(SRC_DIR)/solvers/cg_metrics.cu
CU_SINGLE_GPU_3D_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SINGLE_GPU_3D_SRCS))

# PHONY targets
.PHONY: all clean help check-mpi-message
.PHONY: spmv_bench generate_matrix generate_matrix_3d generate_matrix_3d_27pt cg_solver cg_solver_mgpu_stencil cg_solver_mgpu_stencil_3d cg_solver_single_gpu_3d
.PHONY: spmv gen gen3d gen3d_27pt cg cg3d_mgpu cg3d

# Main target - conditionally include MPI targets
ifeq ($(HAS_MPI),1)
    ALL_TARGETS := $(BIN_SPMV) $(BIN_GEN) $(BIN_GEN3D) $(BIN_GEN3D_27PT) $(BIN_MGPU_STENCIL) $(BIN_MGPU_STENCIL_3D) $(BIN_SINGLE_GPU_3D)
else
    ALL_TARGETS := $(BIN_SPMV) $(BIN_GEN)
endif

all: $(ALL_TARGETS) check-mpi-message

check-mpi-message:
ifeq ($(HAS_MPI),0)
	@echo ""
	@echo "NOTE: MPI not found (mpic++ not in PATH)"
	@echo "      Skipped: CG solver (requires MPI even for single-GPU)"
	@echo "      Install OpenMPI/MPICH: apt install openmpi-bin libopenmpi-dev"
endif

# Help target
help:
	@echo "Available targets:"
	@echo "  make              - Build all binaries (requires MPI for CG solver)"
	@echo ""
	@echo "Explicit targets:"
	@echo "  make spmv_bench              - SpMV benchmark (bin/spmv_bench)"
	@echo "  make generate_matrix         - 2D matrix generator (bin/generate_matrix)"
	@echo "  make generate_matrix_3d      - 3D matrix generator (bin/generate_matrix_3d)"
	@echo "  make cg_solver_mgpu_stencil  - CG solver (bin/cg_solver_mgpu_stencil, MPI)"
	@echo ""
	@echo "Short aliases:"
	@echo "  make spmv         - Alias for spmv_bench"
	@echo "  make gen          - Alias for generate_matrix"
	@echo "  make gen3d        - Alias for generate_matrix_3d"
	@echo "  make cg           - Alias for cg_solver_mgpu_stencil"
	@echo ""
	@echo "Other targets:"
	@echo "  make clean        - Remove all build artifacts"
	@echo "  make help         - Show this help message"

# SpMV benchmark binary
$(BIN_SPMV): $(CU_SPMV_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Matrix generator binary
$(BIN_GEN): $(CU_GEN_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

# 3D Matrix generator binary
$(BIN_GEN3D): $(CU_GEN3D_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

# 3D 27-point Matrix generator binary
$(BIN_GEN3D_27PT): $(CU_GEN3D_27PT_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

# CG solver binary
$(BIN_CG): $(CU_CG_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Compile CUDA sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# ============================================================================
# Multi-GPU CG Solvers with MPI
# ============================================================================

# MPI objects (shared utilities)
OBJ_MGPU_IO := $(OBJ_DIR)/mgpu/io.o
OBJ_MGPU_CSR := $(OBJ_DIR)/mgpu/spmv_csr.o
OBJ_MGPU_STENCIL_SPMV := $(OBJ_DIR)/mgpu/spmv_stencil_csr_direct.o
OBJ_MGPU_HALO_KERNEL := $(OBJ_DIR)/mgpu/spmv_stencil_partitioned_halo_kernel.o
OBJ_MGPU_BENCH_STATS_PARTITIONED := $(OBJ_DIR)/mgpu/benchmark_stats_mgpu_partitioned.o
OBJ_MGPU_CG_METRICS := $(OBJ_DIR)/mgpu/cg_metrics.o

# Stencil solver objects
OBJ_MGPU_STENCIL_MAIN := $(OBJ_DIR)/mgpu/cg_solver_mgpu_stencil.o
OBJ_MGPU_STENCIL_SOLVER := $(OBJ_DIR)/mgpu/cg_solver_mgpu_partitioned.o
OBJ_MGPU_OVERLAP_SOLVER := $(OBJ_DIR)/mgpu/cg_solver_mgpu_overlap.o

# 3D stencil solver objects
OBJ_MGPU_STENCIL_3D_MAIN := $(OBJ_DIR)/mgpu/cg_solver_mgpu_stencil_3d.o
OBJ_MGPU_3D_SOLVER := $(OBJ_DIR)/mgpu/cg_solver_mgpu_partitioned_3d.o
OBJ_MGPU_3D_HALO_KERNEL := $(OBJ_DIR)/mgpu/spmv_stencil_3d_partitioned_halo_kernel.o
OBJ_MGPU_3D_27PT_HALO_KERNEL := $(OBJ_DIR)/mgpu/spmv_stencil_3d_27pt_partitioned_halo_kernel.o

# Compile MPI sources with NVCC + MPI headers
$(OBJ_DIR)/mgpu/%.o: $(SRC_DIR)/main/%.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/%.o: $(SRC_DIR)/solvers/%.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/io.o: $(SRC_DIR)/io/io.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/spmv_csr.o: $(SRC_DIR)/spmv/spmv_cusparse_csr.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/spmv_stencil_csr_direct.o: $(SRC_DIR)/spmv/spmv_stencil_csr_direct.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/spmv_stencil_partitioned_halo_kernel.o: $(SRC_DIR)/spmv/spmv_stencil_partitioned_halo_kernel.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/benchmark_stats_mgpu_partitioned.o: $(SRC_DIR)/spmv/benchmark_stats_mgpu_partitioned.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/spmv_stencil_3d_partitioned_halo_kernel.o: $(SRC_DIR)/spmv/spmv_stencil_3d_partitioned_halo_kernel.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

$(OBJ_DIR)/mgpu/spmv_stencil_3d_27pt_partitioned_halo_kernel.o: $(SRC_DIR)/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu
	@mkdir -p $(OBJ_DIR)/mgpu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(MPI_INCLUDES) -c $< -o $@

# Link stencil solver with MPI (halo P2P approach + overlap variant)
# Note: OBJ_MGPU_3D_HALO_KERNEL needed because overlap solver contains 3D functions
$(BIN_MGPU_STENCIL): $(OBJ_MGPU_STENCIL_MAIN) $(OBJ_MGPU_STENCIL_SOLVER) $(OBJ_MGPU_OVERLAP_SOLVER) $(OBJ_MGPU_3D_HALO_KERNEL) $(OBJ_MGPU_3D_27PT_HALO_KERNEL) $(OBJ_MGPU_IO) $(OBJ_MGPU_CSR) $(OBJ_MGPU_STENCIL_SPMV) $(OBJ_MGPU_HALO_KERNEL) $(OBJ_MGPU_BENCH_STATS_PARTITIONED) $(OBJ_MGPU_CG_METRICS)
	@mkdir -p $(BIN_DIR)
	$(MPICXX) $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

# Link 3D stencil solver with MPI (synchronous + overlap, 7-point + 27-point)
$(BIN_MGPU_STENCIL_3D): $(OBJ_MGPU_STENCIL_3D_MAIN) $(OBJ_MGPU_3D_SOLVER) $(OBJ_MGPU_STENCIL_SOLVER) $(OBJ_MGPU_OVERLAP_SOLVER) $(OBJ_MGPU_3D_HALO_KERNEL) $(OBJ_MGPU_3D_27PT_HALO_KERNEL) $(OBJ_MGPU_IO) $(OBJ_MGPU_CSR) $(OBJ_MGPU_STENCIL_SPMV) $(OBJ_MGPU_HALO_KERNEL) $(OBJ_MGPU_BENCH_STATS_PARTITIONED) $(OBJ_MGPU_CG_METRICS)
	@mkdir -p $(BIN_DIR)
	$(MPICXX) $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

# Single-GPU 3D solver binary
$(BIN_SINGLE_GPU_3D): $(CU_SINGLE_GPU_3D_OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(CUDA_LDFLAGS)

# ============================================================================
# Explicit targets (match binary names)
# ============================================================================

spmv_bench: $(BIN_SPMV)
generate_matrix: $(BIN_GEN)
generate_matrix_3d: $(BIN_GEN3D)
generate_matrix_3d_27pt: $(BIN_GEN3D_27PT)
cg_solver: $(BIN_CG)
cg_solver_mgpu_stencil: $(BIN_MGPU_STENCIL)
cg_solver_mgpu_stencil_3d: $(BIN_MGPU_STENCIL_3D)
cg_solver_single_gpu_3d: $(BIN_SINGLE_GPU_3D)

# ============================================================================
# Short aliases
# ============================================================================

spmv: spmv_bench
gen: generate_matrix
gen3d: generate_matrix_3d
gen3d_27pt: generate_matrix_3d_27pt
cg: cg_solver_mgpu_stencil
cg3d_mgpu: cg_solver_mgpu_stencil_3d
cg3d: cg_solver_single_gpu_3d

# ============================================================================
# Clean
# ============================================================================

clean:
	rm -rf build bin results

