# GPU hooks gather and scatter non-contiguous Legate tiles on-device.

@inline function _natural_strides(dims::NTuple{N,Int}) where {N}
    return ntuple(N) do d
        s = 1
        @inbounds for k in 1:(d - 1)
            s *= dims[k]
        end
        s
    end
end

# Element span covered by a strided tile.
@inline function _tile_span(dims::NTuple{N,Int}, strides::NTuple{N,Int}) where {N}
    s = 1
    @inbounds for d in 1:N
        s += (dims[d] - 1) * strides[d]
    end
    return s
end

# Column-major dense[i] <- src[strided Cartesian offset].
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

# dst[strided Cartesian offset] <- column-major dense[i].
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

# Wrap contiguous inputs zero-copy; gather strided inputs into dense storage.
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

# Wrap contiguous outputs directly; stage strided outputs for a later scatter.
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

# Keep @cuda outside the generated dispatcher.
function _gpu_launch(f, argt, n::Int)
    threads, blocks = _launch_dims(n)
    CUDA.@cuda threads = threads blocks = blocks f(argt)
    return nothing
end

function Legate._ufi_prepare_input(
    ::Legate.GPUBackend,
    ::Type{T},
    ptr::Legate.PhysArrPtr,
    dims::NTuple{N,Int},
    strides::NTuple{N,Int},
) where {T,N}
    return _gpu_gather_in(T, ptr, dims, strides)
end

function Legate._ufi_prepare_output(
    ::Legate.GPUBackend,
    ::Type{T},
    ptr::Legate.PhysArrPtr,
    dims::NTuple{N,Int},
    strides::NTuple{N,Int},
) where {T,N}
    return _gpu_out_buffer(T, ptr, dims, strides)
end

function Legate._ufi_commit_output(
    ::Legate.GPUBackend,
    destination,
    output::CUDA.CuArray{T,N},
    dims::NTuple{N,Int},
    strides::NTuple{N,Int},
) where {T,N}
    destination === nothing || _gpu_scatter_out(destination, output, dims, strides)
    return nothing
end

function Legate._ufi_invoke(::Legate.GPUBackend, f, dims::Tuple, args...)
    return _gpu_launch(f, args, prod(dims))
end

Legate._ufi_synchronize(::Legate.GPUBackend) = CUDA.synchronize()

# Compile on the launch thread while libuv can service ptxas; workers only use cached
# cubins. Dummy arrays specialize the task and transfer kernels by element type/rank.
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
