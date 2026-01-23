# dev

You can enable CUDA-enabled execution even if there’s no GPU available by telling CUDA.jl to set the runtime version.
This is enough for the package to pick the CUDA-enabled versions of the JLLs, which can be useful in container build environments.

```bash
  export CUDA_VERSION_MAJOR_MINOR=13.0
  julia --project=[yourenv] -e 'using Pkg; \
    Pkg.add("CUDA"); using CUDA; CUDA.set_runtime_version!(VersionNumber(ENV["CUDA_VERSION_MAJOR_MINOR"])) \
    CUDA.precompile_runtime()'
```

Note: Currently in Legate.jl, only CUDA versions ≥ 13.0 are supported. 