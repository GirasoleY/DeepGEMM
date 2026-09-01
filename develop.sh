# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Link CUTLASS includes
ln -sf $script_dir/third-party/cutlass/include/cutlass deep_gemm/include
ln -sf $script_dir/third-party/cutlass/include/cute deep_gemm/include

# The regular wheel/build path vendors the validated NCCL headers. Development
# imports use the source package directly, so expose the same include layout via
# a feature-gated symlink.
if [ "${DG_MEGAMOE_GIN:-0}" = "1" ]; then
    nccl_root=${DG_NCCL_ROOT:-${NCCL_ROOT:-}}
    if [ -z "$nccl_root" ]; then
        echo "DG_MEGAMOE_GIN=1 requires DG_NCCL_ROOT (or NCCL_ROOT)" >&2
        exit 1
    fi
    if [ -e deep_gemm/include/nccl ] && [ ! -L deep_gemm/include/nccl ]; then
        echo "deep_gemm/include/nccl exists and is not a generated symlink" >&2
        exit 1
    fi
    ln -sfn "$nccl_root/include" deep_gemm/include/nccl
elif [ -L deep_gemm/include/nccl ]; then
    rm -f deep_gemm/include/nccl
fi

# Remove old dist file, build files, and build
rm -rf build dist
rm -rf *.egg-info
"${PYTHON:-python3}" setup.py build

# Find the .so file in build directory and create symlink in current directory
so_file=$(find build -name "*.so" -type f | head -n 1)
if [ -n "$so_file" ]; then
    ln -sf "../$so_file" deep_gemm/
else
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi

# Open users' original directory
cd "$original_dir"
