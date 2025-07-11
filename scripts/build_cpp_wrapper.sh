set -e

# Check if exactly one argument is provided
if [[ $# -ne 6 ]]; then
    echo "Usage: $0 <legate-pkg> <legate-root> <hdf5-root> <nccl-root> <install-dir> <nthreads>"
    exit 1
fi
LEGATEJL_PKG_ROOT_DIR=$1 # this is the repo root of legate.jl
LEGATE_ROOT=$2 # location of LEGATE_ROOT 
HDF5_ROOT=$3
NCCL_ROOT=$4
INSTALL_DIR=$5
NTHREADS=$6

# Check if the provided argument is a valid directory
if [[ ! -d "$LEGATEJL_PKG_ROOT_DIR" ]]; then
    echo "Error: '$LEGATEJL_PKG_ROOT_DIR' is not a valid directory."
    exit 1
fi

if [[ ! -d "$LEGATE_ROOT" ]]; then
    echo "Error: '$LEGATE_ROOT' is not a valid directory."
    exit 1
fi

if [[ ! -d "$HDF5_ROOT" ]]; then
    echo "Error: '$HDF5_ROOT' is not a valid directory."
    exit 1
fi

if [[ ! -d "$NCCL_ROOT" ]]; then
    echo "Error: '$NCCL_ROOT' is not a valid directory."
    exit 1
fi

LEGION_CMAKE_DIR=$LEGATE_ROOT/share/Legion/cmake
LEGATE_CMAKE_DIR=$LEGATE_ROOT/lib/cmake/legate/

GIT_REPO="https://github.com/JuliaLegate/legate_jl_wrapper"
COMMIT_HASH="f00bd063be66b735fc6040b40027669337399a06"
LEGATE_WRAPPER_SOURCE=$LEGATEJL_PKG_ROOT_DIR/deps/legate_jl_wrapper
BUILD_DIR=$LEGATE_WRAPPER_SOURCE/build

if [ ! -d "$LEGATE_WRAPPER_SOURCE" ]; then
    cd $LEGATEJL_PKG_ROOT_DIR/deps
    git clone $GIT_REPO
fi

cd $LEGATE_WRAPPER_SOURCE
git fetch --tags
git checkout $COMMIT_HASH

if [[ ! -d "$BUILD_DIR" ]]; then
    mkdir $BUILD_DIR 
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
    mkdir $INSTALL_DIR 
fi
# patch the cmake for our custom install
diff -u $LEGATE_WRAPPER_SOURCE/CMakeLists.txt $LEGATEJL_PKG_ROOT_DIR/deps/CMakeLists.txt > deps_install.patch  || true
cd $LEGATE_WRAPPER_SOURCE
patch -i $LEGATE_WRAPPER_SOURCE/deps_install.patch

cmake -S $LEGATE_WRAPPER_SOURCE -B $BUILD_DIR \
    -D CMAKE_PREFIX_PATH="$LEGATE_CMAKE_DIR;$LEGION_CMAKE_DIR" \
    -D LEGATE_PATH=$LEGATE_ROOT \
    -D HDF5_PATH=$HDF5_ROOT \
    -D NCCL_PATH=$NCCL_ROOT \
    -D PROJECT_INSTALL_PATH=$INSTALL_DIR

cmake --build $BUILD_DIR  --parallel $NTHREADS --verbose
cmake --install $BUILD_DIR