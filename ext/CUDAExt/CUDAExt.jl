module CUDAExt

using CUDA
using Legate

using CxxWrap: CxxWrap
import Legate: _execute_julia_task, get_code_type, TaskArgumentGPU

include("ufi.jl")

end # module CUDAExt
