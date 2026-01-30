module CUDAExt

using CUDA
using Legate

using CxxWrap: CxxWrap
import Legate: wrap_task, create_julia_task, SUPPORTED_TYPES, JuliaGPUTask, CxxPtr, Runtime,
    Library, create_task, JULIA_CUSTOM_GPU_TASK, add_scalar, Scalar, register_task_function,
    _execute_julia_task, get_code_type

include("ufi.jl")

end # module CUDAExt
