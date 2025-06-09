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

This will install version 1.10 by default since that is what we have tested against. To verify 1.10 is the default run either of the following (your may need to source bashrc):
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

## Contact
For technical questions, please either contact 
`krasow(at)u.northwestern.edu` OR
`emeitz(at)andrew.cmu.edu`

If the issue is building the package, please include the `build.log` and `.err` files found in `cuNumeric.jl/pkg/deps/` 

