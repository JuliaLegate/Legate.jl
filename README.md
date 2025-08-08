# Legate.jl
Julia Bindings for nv-legate

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]  
> Leagte.jl and cuNumeric.jl are under active development at the moment. This is a pre-release API and is subject to change. Stability is not guaranteed until the first official release. We are actively working to improve the build experience to be more seamless and Julia-friendly. In parallel, we're developing a comprehensive testing framework to ensure reliability and robustness. Our public beta launch is targeted for Fall 2025.

## Minimum prereqs
- Ubuntu 20.04 or RHEL 8
- Julia 1.11

### 1. Install Julia through [JuliaUp](https://github.com/JuliaLang/juliaup)
```
curl -fsSL https://install.julialang.org | sh -s -- --default-channel 1.11
```

This will install version 1.11 by default since that is what we have tested against. To verify 1.10 is the default run either of the following (you may need to source bashrc):
```bash
juliaup status
julia --version
```

If 1.10 is not your default, please set it to be the default. Other versions of Julia are untested.
```bash
juliaup default 1.11
```

### 2. Download Legate.jl
Legate.jl is not on the general registry yet. To add Legate.jl to your environment run:
```julia
using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/Legate.jl", rev = "main")
```
The `rev` option can be main or any tagged version.  By default, this will use JLLs [legate_jll](https://github.com/JuliaPackaging/Yggdrasil/tree/master/L/legate) and [legate_jl_wrapper_jll](https://github.com/JuliaPackaging/Yggdrasil/tree/master/L/legate_jl_wrapper).

To contribute to Legate.jl, we recommend cloning the repository and manually triggering the build process with `Pkg.build` or adding it to one of your existing environments with `Pkg.develop`. This will install clone[legate_jl_wrapper_jll](https://github.com/JuliaLegate/legate_jl_wrapper) into `/deps`. This will cause the wrapper to be built from source, bypassing the `legate_jl_wrapper_jll` prebuilt binary.

```bash
git clone https://github.com/JuliaLegate/Legate.jl.git
julia --project=. -e 'using Pkg; Pkg.develop(path = "Legate.jl/lib/LegatePreferences")'
julia --project=. -e 'using Pkg; Pkg.develop(path = "Legate.jl")'
julia --project=. -e 'using LegatePreferences; LegatePreferences.use_developer_mode()'
julia --project=. -e 'using Pkg; Pkg.build()'
```

#### 2b. Use preinstalled version of [Legate](https://github.com/nv-legate/legate)
We support using a custom install version of Legate. See https://docs.nvidia.com/legate/latest/installation.html for details about different install configurations.

Following the `use_developer_mode()` instructions above, you can add the following to LegatePreference:
```julia
julia --project=. -e 'using LegatePreferences; LegatePreferences.use_developer_mode(;use_legate_jll=false,  legate_path="/path/to/legate/root")'

```
#### 2c. Use a conda environment to install Legate.jl
> [!WARNING]  
> Installing using conda does not pass our CI. This may break. 

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
```
```julia
using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/Legate.jl", rev = "main")
using LegatePreferences; LegatePreferences.use_conda("conda-env-with-legate");
Pkg.build()
```

## Contact
For technical questions, please either contact 
`krasow(at)u.northwestern.edu` OR
`emeitz(at)andrew.cmu.edu`

If the issue is building the package, please include the `build.log` and `.err` files found in `Legate.jl/deps/` 

