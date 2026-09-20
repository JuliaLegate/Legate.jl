# GPU task execution for the UFI. The worker dispatches here (Legate._execute_gpu_task)
# for is_gpu tasks. Legate tiles are strided sub-regions (row-major storage); the user
# kernel expects a plain column-major dense CuArray, so strided tiles are gathered into
# a dense buffer on the device (and outputs scattered back), mirroring the CPU path's
# _strided_to_dense / _dense_to_strided. Contiguous column-major tiles are wrapped
# zero-copy.

@inline function _natural_strides(dims::NTuple{N,Int}) where {N}
    return ntuple(N) do d
        s = 1
        @inbounds for k in 1:(d - 1)
            s *= dims[k]
        end
        s
    end
end

# Linear span (element count) covered by a strided tile.
@inline function _tile_span(dims::NTuple{N,Int}, strides::NTuple{N,Int}) where {N}
    s = 1
    @inbounds for d in 1:N
        s += (dims[d] - 1) * strides[d]
    end
    return s
end

# dense[i] (column-major) <- src[strided offset of i's cartesian index].
function _gather_kernel!(dense::CUDA.CuDeviceArray{T,N}, src::CUDA.CuDeviceArray{T,1},
    dims::NTuple{N,Int}, strides::NTuple{N,Int}) where {T,N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= length(dense)
        rem = i - 1
        off = 0
        for d in 1:N
            c = rem % dims[d]
            rem = rem ÷ dims[d]
            off += c * strides[d]
        end
        dense[i] = src[off + 1]
    end
    return nothing
end

# dst[strided offset] <- dense[i] (column-major).
function _scatter_kernel!(dst::CUDA.CuDeviceArray{T,1}, dense::CUDA.CuDeviceArray{T,N},
    dims::NTuple{N,Int}, strides::NTuple{N,Int}) where {T,N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= length(dense)
        rem = i - 1
        off = 0
        for d in 1:N
            c = rem % dims[d]
            rem = rem ÷ dims[d]
            off += c * strides[d]
        end
        dst[off + 1] = dense[i]
    end
    return nothing
end

@inline function _launch_dims(n::Int)
    threads = min(256, n)
    return threads, cld(n, max(threads, 1))
end

# Input tile -> dense column-major CuArray (zero-copy if already contiguous).
function _gpu_gather_in(::Type{E}, ptr::Legate.PhysArrPtr, dims::NTuple{N,Int},
    strides::NTuple{N,Int}) where {E,N}
    cptr = reinterpret(CUDA.CuPtr{E}, ptr)
    strides == _natural_strides(dims) && return unsafe_wrap(CUDA.CuArray, cptr, dims)
    src = unsafe_wrap(CUDA.CuArray, cptr, _tile_span(dims, strides))
    dense = CUDA.CuArray{E}(undef, dims)
    n = prod(dims)
    th, bl = _launch_dims(n)
    CUDA.@cuda threads = th blocks = bl _gather_kernel!(dense, src, dims, strides)
    return dense
end

# Output buffer. Contiguous: wrap output memory directly (no scatter). Strided: fresh
# dense buffer plus the strided destination array to scatter into afterwards.
function _gpu_out_buffer(::Type{E}, ptr::Legate.PhysArrPtr, dims::NTuple{N,Int},
    strides::NTuple{N,Int}) where {E,N}
    cptr = reinterpret(CUDA.CuPtr{E}, ptr)
    strides == _natural_strides(dims) && return unsafe_wrap(CUDA.CuArray, cptr, dims), nothing
    return CUDA.CuArray{E}(undef, dims), unsafe_wrap(CUDA.CuArray, cptr, _tile_span(dims, strides))
end

function _gpu_scatter_out(dst::CUDA.CuArray, dense::CUDA.CuArray{E,N},
    dims::NTuple{N,Int}, strides::NTuple{N,Int}) where {E,N}
    n = prod(dims)
    th, bl = _launch_dims(n)
    CUDA.@cuda threads = th blocks = bl _scatter_kernel!(dst, dense, dims, strides)
    return nothing
end

# @cuda lives in a normal function (not @generated) to keep the launch out of
# generated code.
function _gpu_launch(f, argt, n::Int)
    threads, blocks = _launch_dims(n)
    CUDA.@cuda threads = threads blocks = blocks f(argt)
    return nothing
end

@generated function _do_call_gpu(
    f,
    in_p_ptr::Ptr{Legate.PhysArrPtr},
    out_p_ptr::Ptr{Legate.PhysArrPtr},
    scal_p_ptr::Ptr{Ptr{Cvoid}},
    in_str_ptr::Ptr{Int64},
    out_str_ptr::Ptr{Int64},
    local_dims::Tuple,
    ::Legate.UfiSignature{InT,OutT,ScT},
) where {InT,OutT,ScT}
    nd = length(local_dims.parameters)
    R = Legate.REALM_MAX_DIM
    instr(base) = Expr(:tuple, [:(Int(unsafe_load(in_str_ptr, $(base + d)))) for d in 1:nd]...)
    outstr(base) = Expr(:tuple, [:(Int(unsafe_load(out_str_ptr, $(base + d)))) for d in 1:nd]...)

    pre = []
    callargs = []
    post = []

    for (i, T) in enumerate(InT.parameters)
        E = eltype(T)
        base = (i - 1) * R
        sym = gensym(:in)
        push!(
            pre, :($sym = _gpu_gather_in(
                $E, unsafe_load(in_p_ptr, $i), local_dims, $(instr(base))))
        )
        push!(callargs, sym)
    end

    for (i, T) in enumerate(OutT.parameters)
        E = eltype(T)
        base = (i - 1) * R
        sym = gensym(:out)
        dst = gensym(:dst)
        st = gensym(:ostr)
        push!(pre, :($st = $(outstr(base))))
        push!(
            pre, :(($sym, $dst) = _gpu_out_buffer(
                $E, unsafe_load(out_p_ptr, $i), local_dims, $st))
        )
        push!(callargs, sym)
        push!(post, :($dst === nothing || _gpu_scatter_out($dst, $sym, local_dims, $st)))
    end

    for (i, T) in enumerate(ScT.parameters)
        push!(callargs, :(unsafe_load(Ptr{$T}(unsafe_load(scal_p_ptr, $i)))))
    end

    return quote
        $(pre...)
        _gpu_launch(f, ($(callargs...),), prod(local_dims))
        $(post...)
        CUDA.synchronize()
        nothing
    end
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
        _do_call_gpu(
            meta.fun,
            pointer(in_args),
            pointer(out_args),
            pointer(scal_args),
            pointer(in_strides),
            pointer(out_strides),
            local_dims,
            sig,
        )
    end
    return nothing
end

# Compile a GPU task's kernels on the submitting (main) thread so context creation,
# the compiler-version probe, and ptxas all run while the libuv event loop is live.
# Cubins are cached, so the UFI worker's launches are cache hits and need no subprocess
# (which would otherwise deadlock against the event loop starved by the main thread's
# blocking Legate call). Precompiles the user kernel plus the gather/scatter kernels
# for each argument's (eltype, ndim). Dummy device arrays match the worker's types.
function Legate._gpu_precompile(fun, in_types, out_types, sc_types, arg_dims)
    CUDA.functional() || return nothing
    n_in = length(in_types)
    args = Any[]
    for i in 1:n_in
        E, N = in_types[i], length(arg_dims[i])
        d = ntuple(_ -> 1, N)
        push!(args, CUDA.zeros(E, d...))
        CUDA.@cuda launch = false _gather_kernel!(
            CUDA.CuArray{E}(undef, d), CUDA.CuArray{E}(undef, 1), d, d)
    end
    for j in 1:length(out_types)
        E, N = out_types[j], length(arg_dims[n_in + j])
        d = ntuple(_ -> 1, N)
        push!(args, CUDA.zeros(E, d...))
        CUDA.@cuda launch = false _scatter_kernel!(
            CUDA.CuArray{E}(undef, 1), CUDA.CuArray{E}(undef, d), d, d)
    end
    for T in sc_types
        push!(args, zero(T))
    end
    CUDA.@cuda launch = false fun((args...,))
    return nothing
end
