#!/bin/bash
# AmgX Installation Script for Linux/CUDA Environments
# Handles various platforms and permission constraints
# Author: Bouhrour Stephane
# Date: 2025-09-24

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AMGX_VERSION="v2.5.0"  # Tag v2.5.0 = commit cc1cebd, the AmgX used for the published comparison

# Determine project root and install location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_PREFIX="$PROJECT_ROOT/external/amgx"
TEMP_DIR="/tmp/amgx_build_$$"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check CUDA installation and version
check_cuda() {
    print_status "Checking CUDA installation..."
    
    if ! command -v nvcc &> /dev/null; then
        print_error "CUDA toolkit not found. nvcc is required for AmgX compilation."
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    print_success "Found CUDA version: $CUDA_VERSION"
    
    # Check for cuSOLVER library
    if [[ -f "/usr/local/cuda/lib64/libcusolver.so" ]] || [[ -f "/usr/lib/x86_64-linux-gnu/libcusolver.so" ]]; then
        print_success "cuSOLVER library found"
    else
        print_warning "cuSOLVER library not found in standard locations"
        print_status "AmgX may fail to link. Continuing anyway..."
    fi
}

# Get supported CUDA architectures based on CUDA version
get_cuda_architectures() {
    local cuda_major=$(echo "$CUDA_VERSION" | cut -d. -f1)

    case "$cuda_major" in
        11)
            # CUDA 11.x: Volta (70) through Ampere (86)
            echo "70;75;80;86"
            ;;
        12)
            # CUDA 12.x: Volta (70) through Hopper (90)
            echo "70;75;80;86;89;90"
            ;;
        13|*)
            # CUDA 13+: Ampere (80) through Blackwell (100)
            # Volta (70) and Turing (75) deprecated
            echo "80;86;89;90;100"
            ;;
    esac
}

# Check system dependencies
check_dependencies() {
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Essential build tools
    command -v cmake >/dev/null 2>&1 || missing_deps+=("cmake")
    command -v make >/dev/null 2>&1 || missing_deps+=("make")
    command -v g++ >/dev/null 2>&1 || missing_deps+=("g++")
    command -v git >/dev/null 2>&1 || missing_deps+=("git")
    command -v bc >/dev/null 2>&1 || missing_deps+=("bc")
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_warning "Missing dependencies: ${missing_deps[*]}"
        print_status "Attempting to install dependencies..."
        
        # Try different package managers (only if sudo available)
        if sudo -n true 2>/dev/null; then
            if command -v apt-get >/dev/null 2>&1; then
                sudo apt-get update && sudo apt-get install -y cmake build-essential git bc
            elif command -v yum >/dev/null 2>&1; then
                sudo yum install -y cmake gcc-c++ make git bc
            else
                print_warning "Package manager not supported, continuing with manual installation"
            fi
        else
            print_warning "No sudo access - cannot auto-install dependencies"
            print_status "Please install manually: ${missing_deps[*]}"
            print_status "Continuing anyway - some dependencies might be available..."
        fi
    else
        print_success "All dependencies found"
    fi
}

# Check available disk space
check_disk_space() {
    print_status "Checking disk space..."
    
    local available=$(df "$HOME" | awk 'NR==2 {print $4}')
    local required=2097152  # 2GB in KB
    
    if [[ $available -lt $required ]]; then
        print_warning "Low disk space: $(($available/1024))MB available, 2GB recommended"
        print_status "Continuing with installation..."
    else
        print_success "Sufficient disk space: $(($available/1024))MB available"
    fi
}

# Test sudo access (informational only, not fatal)
check_sudo() {
    if sudo -n true 2>/dev/null; then
        print_success "Sudo access available"
    else
        print_warning "No sudo access - using local installation prefix"
    fi
    # Always return success - no sudo is not fatal
    return 0
}

# Clean up previous installations
# Returns 1 if user wants to keep existing installation (skip build)
cleanup_previous() {
    print_status "Checking for previous installations..."

    # Remove temporary directory if it exists
    [[ -d "$TEMP_DIR" ]] && rm -rf "$TEMP_DIR"

    # Check for previous installation
    if [[ -d "$INSTALL_PREFIX" ]] && [[ -f "$INSTALL_PREFIX/include/amgx_c.h" ]]; then
        print_success "AmgX already installed at $INSTALL_PREFIX"
        read -p "Reinstall AmgX? [y/N]: " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Keeping existing installation"
            return 1  # Skip build
        fi
        rm -rf "$INSTALL_PREFIX"
        print_status "Previous installation removed, proceeding with fresh install"
    fi
    return 0
}

# Download and build AmgX
build_amgx() {
    print_status "Starting AmgX build process..."
    
    # Create temporary build directory
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Clone AmgX repository
    print_status "Cloning AmgX repository (version $AMGX_VERSION)..."
    git clone --depth 1 --branch "$AMGX_VERSION" https://github.com/NVIDIA/AMGX.git
    cd AMGX
    
    # Initialize submodules (required for Thrust)
    print_status "Initializing git submodules (Thrust dependency)..."
    git submodule update --init --recursive
    
    # Create build directory
    mkdir build && cd build
    
    # Configure CMake
    print_status "Configuring CMake build..."
    local cmake_args=(
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )

    # Enable MPI support if available (required for multi-GPU distributed API)
    if command -v mpicc >/dev/null 2>&1; then
        cmake_args+=(-DCMAKE_NO_MPI=0)
        print_status "MPI found - building AmgX with MPI support (required for multi-GPU)"
    else
        cmake_args+=(-DCMAKE_NO_MPI=1)
        print_warning "MPI not found - building AmgX without MPI (multi-GPU will not work)"
    fi

    # Target architectures: AMGX_CUDA_ARCHITECTURES if set (e.g. "80" for A100 only), else the
    # list for the CUDA version. AmgX's own default (90;100;120) has no code for sm_80 or sm_86/89.
    local cuda_archs="${AMGX_CUDA_ARCHITECTURES:-$(get_cuda_architectures)}"
    cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$cuda_archs")
    print_status "Using CUDA architectures: $cuda_archs (CUDA $CUDA_VERSION)"
    
    cmake "${cmake_args[@]}" ..
    
    # Build AmgX
    print_status "Building AmgX (this may take 10-15 minutes)..."
    local nproc_count=$(nproc)
    make -j"$nproc_count"
    
    # Install AmgX
    print_status "Installing AmgX to $INSTALL_PREFIX..."
    make install
    
    print_success "AmgX build completed successfully"
}

# Update AMGX benchmark Makefile and build benchmarks
build_amgx_benchmarks() {
    local amgx_benchmark_dir="$PROJECT_ROOT/external/benchmarks/amgx"
    local amgx_makefile="$amgx_benchmark_dir/Makefile"

    if [[ ! -f "$amgx_makefile" ]]; then
        print_warning "AMGX benchmark Makefile not found at $amgx_makefile"
        return
    fi

    # The benchmark Makefile finds the installation in external/amgx by itself
    print_status "Building AmgX benchmark binaries..."
    cd "$amgx_benchmark_dir"

    if make clean && make -j"$(nproc)"; then
        print_success "AmgX benchmarks built successfully"
        if [[ -f "amgx_cg_solver" ]]; then
            print_success "  - amgx_cg_solver (single-GPU)"
        fi
        if [[ -f "amgx_cg_solver_mgpu" ]]; then
            print_success "  - amgx_cg_solver_mgpu (multi-GPU)"
        fi
    else
        print_warning "AmgX benchmark build failed - you can build manually later"
        print_status "  cd external/benchmarks/amgx && make"
    fi

    cd "$PROJECT_ROOT"
}

# Verify installation
verify_installation() {
    print_status "Verifying AmgX installation..."
    
    # Check header files
    if [[ -f "$INSTALL_PREFIX/include/amgx_c.h" ]]; then
        print_success "AmgX headers found"
    else
        print_error "AmgX headers not found at $INSTALL_PREFIX/include/"
        return 1
    fi
    
    # Check library files
    if [[ -f "$INSTALL_PREFIX/lib/libamgx.a" ]] || [[ -f "$INSTALL_PREFIX/lib/libamgxsh.so" ]]; then
        print_success "AmgX libraries found"
    else
        print_error "AmgX libraries not found at $INSTALL_PREFIX/lib/"
        return 1
    fi
    
    print_success "AmgX installation verified"
}

# Cleanup temporary files
cleanup_temp() {
    print_status "Cleaning up temporary files..."
    [[ -d "$TEMP_DIR" ]] && rm -rf "$TEMP_DIR"
    print_success "Cleanup completed"
}

# Main installation process
main() {
    echo "=================================================="
    echo "  AmgX Installation Script for Linux/CUDA        "  
    echo "=================================================="
    echo
    
    # Pre-installation checks
    check_cuda
    check_dependencies
    check_disk_space
    check_sudo
    
    # Check for existing installation
    if cleanup_previous; then
        # Build AmgX library
        build_amgx

        echo
        echo "=================================================="
        print_success "AmgX library installed successfully!"
        echo "=================================================="
    else
        echo
        echo "=================================================="
        print_success "Using existing AmgX installation"
        echo "=================================================="
    fi

    # Build our AmgX benchmark binaries
    build_amgx_benchmarks

    # Verify installation works
    verify_installation

    # Final cleanup
    cleanup_temp
    echo
    echo "Installation details:"
    echo "  - AmgX version: $AMGX_VERSION"
    echo "  - Install location: $INSTALL_PREFIX"
    echo "  - Headers: $INSTALL_PREFIX/include/"
    echo "  - Libraries: $INSTALL_PREFIX/lib/"
    echo
    echo "Run the comparison (run_all.sh detects AmgX in external/amgx):"
    echo "  ./scripts/run_all.sh --quick"
    echo
}

# Error handling
trap 'print_error "Installation failed at line $LINENO. Check the output above for details."' ERR
trap cleanup_temp EXIT

# Run main installation
main "$@"