set -e

# Check if exactly one argument is provided
if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <legate-pkg> <legate-root> <install-dir> <nthreads>"
    exit 1
fi
LEGATEJL_PKG_ROOT_DIR=$1 # this is the repo root of legate.jl
LEGATE_ROOT=$2 # location of LEGATE_ROOT 
INSTALL_DIR=$3
NTHREADS=$4

# Check if the provided argument is a valid directory
if [[ ! -d "$LEGATEJL_PKG_ROOT_DIR" ]]; then
    echo "Error: '$LEGATEJL_PKG_ROOT_DIR' is not a valid directory."
    exit 1
fi

if [[ ! -d "$LEGATE_ROOT" ]]; then
    echo "Error: '$LEGATE_ROOT' is not a valid directory."
    exit 1
fi


LEGION_CMAKE_DIR=$LEGATE_ROOT/share/Legion/cmake
REALM_CMAKE_DIR=$LEGATE_ROOT/lib/cmake/realm
LEGATE_CMAKE_DIR=$LEGATE_ROOT/lib/cmake/legate

LEGATE_WRAPPER_SOURCE=$LEGATEJL_PKG_ROOT_DIR/lib/legate_jl_wrapper
BUILD_DIR=$LEGATE_WRAPPER_SOURCE/build

if [[ ! -d "$BUILD_DIR" ]]; then
    mkdir -p $BUILD_DIR 
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
    mkdir -p $INSTALL_DIR 
fi

cmake -S $LEGATE_WRAPPER_SOURCE -B $BUILD_DIR \
    -D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -D BINARYBUILDER=OFF \
    -D CMAKE_PREFIX_PATH="$LEGATE_CMAKE_DIR;$LEGION_CMAKE_DIR;$REALM_CMAKE_DIR" \
    -D CMAKE_BUILD_TYPE=Release

cmake --build $BUILD_DIR  --parallel $NTHREADS --verbose
