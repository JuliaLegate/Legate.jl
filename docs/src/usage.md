## Setting Hardware Configuration

There is no programatic way to set the hardware configuration used by Legate (as of 26.01). By default, the hardware configuration is set automatically by Legate. This configuration can be manipulated through the following environment variables:

- `LEGATE_SHOW_CONFIG` : When set to 1, the Legate config is printed to stdout
- `LEGATE_AUTO_CONFIG`: When set to 1, Legate will automatically choose the hardware configuration
- `LEGATE_CONFIG`: A string representing the hardware configuration to set

These variables must be set before launching the Julia instance running Legate.jl and/or cuNumeric.jl. We recommend setting `export LEGATE_SHOW_CONFIG=1` so that the hardware configuration will be printed when Legate starts. This output is automatically captured and relayed to the user.

To manually set the hardware configuration, `export LEGATE_AUTO_CONFIG=0`, and then define your own config with something like `export LEGATE_CONFIG="--gpus 1 --cpus 10 --ompthreads 10"`. We recommend using the default memory configuration for your machine and only settings the `gpus`, `cpus` and `ompthreads`. More details about the Legate configuration can be found in the [NVIDIA Legate documentation](https://docs.nvidia.com/legate/latest/usage.html#resource-allocation). If you know where Legate is installed on your computer you can also run `legate --help` for more detailed information.


## Container Build Environments

You can enable CUDA-enabled execution even if there’s no GPU available by telling CUDA.jl to set the runtime version.
This is enough for the package to pick the CUDA-enabled versions of the JLLs, which can be useful in container build environments.

```bash
  export CUDA_VERSION_MAJOR_MINOR=13.0
  julia --project=[yourenv] -e 'using Pkg; \
    Pkg.add("CUDA"); using CUDA; CUDA.set_runtime_version!(VersionNumber(ENV["CUDA_VERSION_MAJOR_MINOR"])) \
    CUDA.precompile_runtime()'
```

Note: Currently in Legate.jl, only CUDA versions ≥ 13.0 are supported. 