module CUDAExt

using Random
using CUDA
using Legate
import Legate: wrap_task, create_julia_task

include("ufi.jl")

function __init__()
    if CUDA.functional()
        # in /wrapper/src/cuda.cpp
        # Legate.register_tasks();
    else
        @warn "CUDA.jl is not functional; skipping CUDA Tasking registration."
    end
end

end # module CUDAExt
