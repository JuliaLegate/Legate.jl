# Build Options

To make customization of the build options easier we have the `LegatePreferences.jl` package to generate the `LocalPreferences.toml` which is read by the build script to determine which build option to use. LegatePreferences.jl will also enforce that Julia is restarted for changes to take effect.

## Default Build (jlls)

```julia
pkg> add Legate
```
If you previously used a custom build or conda build and would like to revert back to using prebuilt JLLs, run the following command in the directory containing the Project.toml of your environment.

```julia
using LegatePreferences; LegatePreferences.use_jll_binary()
```

`LegatePreferences` is a separate module so that it can be used to configure the build settings before `Legate.jl` is added to your environment. To install it separately run

```julia
pkg> add LegatePreferences
```

## Developer mode
> [!TIP]  
> This gives the most flexibility in installs. It is meant for developing on Legate.jl.

We support using a custom install version of Legate. See https://docs.nvidia.com/legate/latest/installation.html for details about different install configurations, or building Legate from source.

We require that you have a g++ capable compiler of C++ 20, and a recent version CMake >= 3.26.

To use developer mode, 
```julia
using LegatePreferences; LegatePreferences.use_developer_mode(; use_legate_jll=true, legate_path=nothing)
```
By default `use_legate_jll` will be set to true. However, you can set a custom branch and/or use a custom path of legate. By using disabling `use_legate_jll`, you can set `legate_path` to your custom install. 
```julia
using LegatePreferences; LegatePreferences.use_developer_mode(;use_legate_jll=false,  legate_path="/path/to/legate/root")

```

> [!WARNING]
> Right now, building Legate with CUDA support requires both the CUDA driver library (libcuda.so) and the CUDA runtime library (libcudart.so) to be discoverable on the system library path.
> Automatic detection and use of CUDA versions provided via JLL packages is not yet implemented and remains a TODO.
>
> In `deps/cpp_wrapper.log`, you will a CUDAToolkit version that is not self-contained with our JLL distributions.


## Link Against Existing Conda Environment

> [!WARNING]  
> This feature is not passing our CI currently. Please use with caution. We are failing to currently match proper versions of .so libraries together. Our hope is to get this functional for users already using Legate within conda. 

Note, you need conda >= 24.1 to install the conda package. More installation details are found [here](https://docs.nvidia.com/legate/latest/installation.html).

```bash
# with a new environment
conda create -n myenv -c conda-forge -c legate
# into an existing environment
conda install -c conda-forge -c legate
```
Once you have the conda package installed, you can activate here. 
```bash
conda activate [conda-env-with-legate]
```

To update `LocalPreferences.toml` so that a local conda environment is used as the binary provider for cupynumeric run the following command. `conda_env` should be the absolute path to the conda environment (e.g., the value of CONDA_PREFIX when your environment is active). For example, this path is: `/home/JuliaLegate/.conda/envs/legate-gpu`.
```julia
using LegatePreferences; LegatePreferences.use_conda("conda-env-with-legate");
Pkg.build()
```