# Auto-detect AmgX installation
AMGX_SRC_DIR := $(CURDIR)/../../amgx-src
AMGX_INSTALL_DIR := $(CURDIR)/../../amgx

# Priority: amgx-src/build > amgx > /usr/local
ifeq ($(wildcard $(AMGX_SRC_DIR)/include/amgx_c.h),$(AMGX_SRC_DIR)/include/amgx_c.h)
    # AmgX source tree found
    AMGX_INCLUDE = -I$(AMGX_SRC_DIR)/include
    AMGX_LIBDIR = $(AMGX_SRC_DIR)/build
else ifeq ($(wildcard $(AMGX_INSTALL_DIR)/include/amgx_c.h),$(AMGX_INSTALL_DIR)/include/amgx_c.h)
    # AmgX install tree found
    AMGX_INCLUDE = -I$(AMGX_INSTALL_DIR)/include
    AMGX_LIBDIR = $(AMGX_INSTALL_DIR)/lib
else
    # Fallback to system install
    AMGX_INCLUDE = -I/usr/local/include
    AMGX_LIBDIR = /usr/local/lib
endif

# Without a libamgx.a in AMGX_LIBDIR the linker silently takes whatever /usr/local/lib holds, which
# may be an AmgX built without MPI: every distributed call then fails with "Incorrect parameters".
ifeq ($(wildcard $(AMGX_LIBDIR)/libamgx.a),)
    $(warning no libamgx.a in $(AMGX_LIBDIR): the linker will fall back to its default search path)
endif

CXX = nvcc
CXXFLAGS = -std=c++17 -O3 --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true $(AMGX_INCLUDE)
LDFLAGS = -L$(AMGX_LIBDIR) -lamgx -lcudart -lcusparse -lcublas -lcusolver -lcuda -lmpi

# MPI flags for multi-GPU
MPICXX = mpic++
MPI_CXXFLAGS = $(CXXFLAGS) -I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LDFLAGS = $(LDFLAGS) -lmpi -lmpi_cxx

all: amgx_spmv_bench amgx_cg_solver amgx_cg_solver_mgpu

amgx_spmv_bench: amgx_spmv_stencil.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

amgx_cg_solver: amgx_cg_solver.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# The in-memory 3D stencil generators are shared with the custom solver (src/io/io.cu)
SHOWCASE_ROOT := $(CURDIR)/../../..
amgx_cg_solver_mgpu: amgx_cg_solver_mgpu.cpp stencil_partition.h amgx_benchmark.h $(SHOWCASE_ROOT)/src/io/io.cu
	$(CXX) $(MPI_CXXFLAGS) -I$(SHOWCASE_ROOT)/include $< $(SHOWCASE_ROOT)/src/io/io.cu -o $@ $(MPI_LDFLAGS)

clean:
	rm -f amgx_spmv_bench amgx_cg_solver amgx_cg_solver_mgpu

help:
	@echo "AmgX Benchmarks Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all                  - Build all benchmarks"
	@echo "  amgx_spmv_bench      - Build SpMV benchmark"
	@echo "  amgx_cg_solver       - Build single-GPU CG solver"
	@echo "  amgx_cg_solver_mgpu  - Build multi-GPU MPI CG solver"
	@echo "  clean                - Remove all binaries"
	@echo ""
	@echo "AmgX auto-detection:"
	@echo "  Detected include: $(AMGX_INCLUDE)"
	@echo "  Detected libdir:  $(AMGX_LIBDIR)"

.PHONY: all clean help
