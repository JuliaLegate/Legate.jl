set -e

# Check if exactly one argument is provided
if [[ $# -ne 7 ]]; then
    echo "Usage: $0 <legate-pkg> <legate-root> <hdf5-root> <nccl-root> <install-dir> <branch> <nthreads>"
    exit 1
fi
LEGATEJL_PKG_ROOT_DIR=$1 # this is the repo root of legate.jl
LEGATE_ROOT=$2 # location of LEGATE_ROOT 
HDF5_ROOT=$3
NCCL_ROOT=$4
INSTALL_DIR=$5
WRAPPER_BRANCH=$6
NTHREADS=$7

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

echo "Checking out wrapper branch: $WRAPPER_BRANCH"
GIT_REPO="https://github.com/JuliaLegate/legate_jl_wrapper"
LEGATE_WRAPPER_SOURCE=$LEGATEJL_PKG_ROOT_DIR/deps/legate_jl_wrapper_src

if [ ! -d "$LEGATE_WRAPPER_SOURCE" ]; then
    git clone $GIT_REPO $LEGATE_WRAPPER_SOURCE

    cd "$LEGATE_WRAPPER_SOURCE" || exit 1
    echo "Current repo: $(basename $(pwd))"
    git remote -v

    git fetch origin "$WRAPPER_BRANCH"
    git checkout "$WRAPPER_BRANCH"
    
    # patch the cmake for our custom install
    diff -u $LEGATE_WRAPPER_SOURCE/CMakeLists.txt $LEGATEJL_PKG_ROOT_DIR/deps/CMakeLists.txt > deps_install.patch  || true
    cd $LEGATE_WRAPPER_SOURCE
    patch -i $LEGATE_WRAPPER_SOURCE/deps_install.patch
fi

BUILD_DIR=$LEGATE_WRAPPER_SOURCE/build

if [[ ! -d "$BUILD_DIR" ]]; then
    mkdir -p $BUILD_DIR 
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
    mkdir -p $INSTALL_DIR 
fi


cmake -S $LEGATE_WRAPPER_SOURCE -B $BUILD_DIR \
    -D CMAKE_PREFIX_PATH="$LEGATE_CMAKE_DIR;$LEGION_CMAKE_DIR" \
    -D LEGATE_PATH=$LEGATE_ROOT \
    -D HDF5_PATH=$HDF5_ROOT \
    -D NCCL_PATH=$NCCL_ROOT \
    -D PROJECT_INSTALL_PATH=$INSTALL_DIR \
    -D CMAKE_BUILD_TYPE=Release

cmake --build $BUILD_DIR  --parallel $NTHREADS --verbose
cmake --install $BUILD_DIR