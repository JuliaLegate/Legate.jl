# GPU task execution for the UFI. The worker dispatches here (Legate._execute_gpu_task)
# for is_gpu tasks. Device pointers are wrapped as CuArrays; the user kernel receives
# them as a single tuple and is launched with @cuda. Phase 1: assumes dense
# (non-partitioned) tiles.

# Compile a GPU task's kernel on the submitting (main) thread so context creation,
# the compiler-version probe, and ptxas all run while the libuv event loop is live.
# The cubin is cached, so the UFI worker's launch is a cache hit and needs no
# subprocess (which would otherwise deadlock against the event loop starved by the
# main thread's blocking Legate call). Dummy device arrays match the exact types the
# worker builds in _gpu_args, so the compilation caches under the right key.
function Legate._gpu_precompile(fun, in_types, out_types, sc_types, arg_dims)
    CUDA.functional() || return nothing
    n_in = length(in_types)
    args = Any[]
    for i in 1:n_in
        push!(args, CUDA.zeros(in_types[i], ntuple(_ -> 1, length(arg_dims[i]))...))
    end
    for j in 1:length(out_types)
        push!(args, CUDA.zeros(out_types[j], ntuple(_ -> 1, length(arg_dims[n_in + j]))...))
    end
    for T in sc_types
        push!(args, zero(T))
    end
    CUDA.@cuda launch = false fun((args...,))
    return nothing
end

# Build the argument tuple (CuArrays for in/out, values for scalars) from the request.
@generated function _gpu_args(
    in_p_ptr::Ptr{Legate.PhysArrPtr},
    out_p_ptr::Ptr{Legate.PhysArrPtr},
    scal_p_ptr::Ptr{Ptr{Cvoid}},
    local_dims::Tuple,
    ::Legate.UfiSignature{InT,OutT,ScT},
) where {InT,OutT,ScT}
    args = []
    for (i, T) in enumerate(InT.parameters)
        E = eltype(T)
        push!(
            args,
            :(unsafe_wrap(
                CUDA.CuArray, reinterpret(CUDA.CuPtr{$E}, unsafe_load(in_p_ptr, $i)), local_dims)),
        )
    end
    for (i, T) in enumerate(OutT.parameters)
        E = eltype(T)
        push!(
            args,
            :(unsafe_wrap(
                CUDA.CuArray, reinterpret(CUDA.CuPtr{$E}, unsafe_load(out_p_ptr, $i)), local_dims)),
        )
    end
    for (i, T) in enumerate(ScT.parameters)
        push!(args, :(unsafe_load(Ptr{$T}(unsafe_load(scal_p_ptr, $i)))))
    end
    return :(($(args...),))
end

# @cuda lives in a normal function (not @generated) to keep the launch out of
# generated code.
function _gpu_launch(f, argt, n::Int)
    threads = min(256, n)
    blocks = cld(n, max(threads, 1))
    CUDA.@cuda threads = threads blocks = blocks f(argt)
    CUDA.synchronize()
    return nothing
end

function Legate._execute_gpu_task(
    meta::Legate.UfiMetadata,
    in_args::Vector{Legate.PhysArrPtr},
    out_args::Vector{Legate.PhysArrPtr},
    scal_args::Vector{Ptr{Cvoid}},
    in_strides::Vector{Int64},
    out_strides::Vector{Int64},
    local_dims::Tuple,
    sig,
)
    GC.@preserve in_args out_args scal_args in_strides out_strides begin
        argt = _gpu_args(
            pointer(in_args), pointer(out_args), pointer(scal_args), local_dims, sig)
        _gpu_launch(meta.fun, argt, prod(local_dims))
    end
    return nothing
end
