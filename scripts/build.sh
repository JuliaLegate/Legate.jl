legate_root=`python -c 'import legate.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'`
echo "Using Legate at $legate_root"

REPO_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"

sh $REPO_ROOT/scripts/patch_legion.sh $REPO_ROOT $legate_root 

cd $REPO_ROOT
cmake -S ./wrapper -B build -D legate_ROOT="$legate_root" -D CMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 8
