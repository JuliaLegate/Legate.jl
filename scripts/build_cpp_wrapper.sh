set -e

# Check if exactly one argument is provided
if [[ $# -ne 6 ]]; then
    echo "Usage: $0 <legate-pkg> <legate-jll> <hdf5-jll> <nccl-jll> <build-dir> <nthreads>"
    exit 1
fi
LEGATE_PKG_ROOT_DIR=$1 # this is the repo root of legate.jl
LEGATE_JLL=$2 # location of legate_jll 
HDF5_JLL=$3
NCCL_JLL=$4
BUILD_DIR=$5 # /wrapper/build
NTHREADS=$6

# Check if the provided argument is a valid directory
if [[ ! -d "$LEGATE_PKG_ROOT_DIR" ]]; then
    echo "Error: '$LEGATE_PKG_ROOT_DIR' is not a valid directory."
    exit 1
fi

if [[ ! -d "$LEGATE_JLL" ]]; then
    echo "Error: '$LEGATE_JLL' is not a valid directory."
    exit 1
fi

if [[ ! -d "$HDF5_JLL" ]]; then
    echo "Error: '$HDF5_JLL' is not a valid directory."
    exit 1
fi

if [[ ! -d "$NCCL_JLL" ]]; then
    echo "Error: '$NCCL_JLL' is not a valid directory."
    exit 1
fi

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: '$BUILD_DIR' is not a valid directory."
    exit 1
fi

LEGION_CMAKE_DIR=$LEGATE_JLL/share/Legion/cmake
LEGATE_CMAKE_DIR=$LEGATE_JLL/lib/cmake/legate/

cmake -S $LEGATE_PKG_ROOT_DIR/wrapper -B $BUILD_DIR \
    -D CMAKE_PREFIX_PATH="$LEGATE_CMAKE_DIR;$LEGION_CMAKE_DIR" \
    -D LEGATE_PATH=$LEGATE_JLL \
    -D HDF5_PATH=$HDF5_JLL \
    -D NCCL_PATH=$NCCL_JLL

cmake --build $BUILD_DIR  --parallel $NTHREADS --verbose