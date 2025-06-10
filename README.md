# Legate.jl
Julia Bindings for nv-legate

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


## Minimum prereqs
- g++ capable of C++20
- CUDA 12.2
- Python 3.10
- Ubuntu 20.04 or RHEL 8
- Julia 1.10
- CMake 3.26.4 

### 1. Install Julia through [JuliaUp](https://github.com/JuliaLang/juliaup)
```
curl -fsSL https://install.julialang.org | sh -s -- --default-channel 1.10
```

This will install version 1.10 by default since that is what we have tested against. To verify 1.10 is the default run either of the following (you may need to source bashrc):
```bash
juliaup status
julia --version
```

If 1.10 is not your default, please set it to be the default. Other versions of Julia are untested.
```bash
juliaup default 1.10
```

### 2. Download Legate.jl
Legate.jl is not on the general registry yet. To add Legate.jl to your environment run:
```julia
using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/Legate.jl", rev = "main")
```
The `rev` option can be main or any tagged version.  By default, this will use [legate_jll](https://github.com/ejmeitz/legate_jll.jl). In [2b](#2b-use-preinstalled-version-of-legate) and [2c](#2c-use-a-conda-environment-to-install-legatejl), we show different installation methods. Ensure that the enviroment variables are correctly set for custom builds.

To contribute to Legate.jl, we recommend cloning the repository and manually triggering the build process with `Pkg.build` or adding it to one of your existing environments with `Pkg.develop`.
```bash
git clone https://github.com/JuliaLegate/Legate.jl.git
cd Legate.jl
julia -e 'using Pkg; Pkg.activate(".") Pkg.resolve(); Pkg.build()'
```

#### 2b. Use preinstalled version of [Legate](https://github.com/nv-legate/legate)
We support using a custom install version of Legate. See https://docs.nvidia.com/legate/latest/installation.html for details about different install configurations.
```bash
export LEGATE_CUSTOM_INSTALL=1
export LEGATE_CUSTOM_INSTALL_LOCATION="/home/user/path/to/legate-install-dir"
```
```julia
using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/Legate.jl", rev = "main")
```

#### 2c. Use a conda environment to install Legate.jl
Note, you need conda >= 24.1 to install the conda package. More installation details are found [here](https://docs.nvidia.com/legate/latest/installation.html).
```bash
# with a new environment
conda create -n myenv -c conda-forge -c legate legate
# into an existing environment
conda install -c conda-forge -c legate legate
```
Once you have the conda package installed, you can activate here. 
```bash
conda activate [conda-env-with-legate]
export CUNUMERIC_LEGATE_CONDA_INSTALL=1
```
```julia
using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/Legate.jl", rev = "main")
```

## Contact
For technical questions, please either contact 
`krasow(at)u.northwestern.edu` OR
`emeitz(at)andrew.cmu.edu`

If the issue is building the package, please include the `build.log` and `.err` files found in `Legate.jl/pkg/deps/` 

