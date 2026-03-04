#= Copyright 2026 Northwestern University, 
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
=#

function attach_external(arr::Array{T,N}; read_only::Bool=true) where {T,N}
    ptr = Base.unsafe_convert(Ptr{Cvoid}, arr)
    shape = collect(UInt64, size(arr))
    lshape = Shape(to_cxx_vector(shape))
    impl = LegateInternal.attach_external_store_sysmem(ptr, lshape, to_legate_type(T), read_only)
    return LogicalStore{T,N}(impl, size(arr))
end

function Base.copyto!(dest::LogicalArray{T,N}, src::Array{T,N}) where {T,N}
    attached = attach_external(src)
    copyto!(dest, attached)
    _detach_nonblocking(attached.handle)
    return dest
end

function Base.copyto!(dest::LogicalStore{T,N}, src::LogicalStore{T,N}) where {T,N}
    _copyto_impl(dest.handle, src.handle)
    return dest
end

function Base.copyto!(dest::LogicalStore{T,N}, src::LogicalArray{T,N}) where {T,N}
    _copyto_impl(dest.handle, LegateInternal.data(src.handle))
    return dest
end

function Base.copyto!(dest::LogicalArray{T,N}, src::LogicalStore{T,N}) where {T,N}
    _copyto_impl(LegateInternal.data(dest.handle), src.handle)
    return dest
end

function Base.copyto!(dest::LogicalArray{T,N}, src::LogicalArray{T,N}) where {T,N}
    _copyto_impl(LegateInternal.data(dest.handle), LegateInternal.data(src.handle))
    return dest
end

function _await_legate_future(future_ptr::Ptr{Cvoid})
    mgr = UFI_MANAGER[]
    while true
        ready = ccall((:legate_is_future_ready, Legate.WRAPPER_LIB_PATH), Bool, (Ptr{Cvoid},), future_ptr)
        if ready
            break
        end
        # Legate is waiting for something (probably Julia tasks). Keep the UFI poller going.
        !isnothing(mgr) && ufi_poll(mgr)
        yield()
        sleep(0.001)
    end
    ccall((:legate_wait_and_destroy_future, Legate.WRAPPER_LIB_PATH), Cvoid, (Ptr{Cvoid},), future_ptr)
end

function _copyto_impl(dest_handle::LogicalStoreImpl, src_handle::LogicalStoreImpl)
    future_ptr = ccall(
        (:legate_start_copy, Legate.WRAPPER_LIB_PATH),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}),
        dest_handle.cpp_object, src_handle.cpp_object
    )
    _await_legate_future(future_ptr)
end

function _detach_nonblocking(store_handle::LogicalStoreImpl)
    future_ptr = ccall(
        (:legate_start_detach, Legate.WRAPPER_LIB_PATH),
        Ptr{Cvoid},
        (Ptr{Cvoid},),
        store_handle.cpp_object
    )
    _await_legate_future(future_ptr)
end

# Conversion from LogicalArray to Base Julia array
function (::Type{<:Array{A}})(arr::LogicalArray{B}) where {A,B}
    dims = Base.size(arr)
    out = Array{A}(undef, dims)
    GC.@preserve out begin
        attached = attach_external(out; read_only=false)
        copyto!(attached, arr)
        _detach_nonblocking(attached.handle)
    end
    return out
end

(::Type{<:Array})(arr::LogicalArray{B}) where {B} = Array{B}(arr)

function (::Type{<:LogicalArray{A}})(arr::Array{B}) where {A,B}
    dims = Base.size(arr)
    out = create_array(collect(dims), A)
    GC.@preserve arr begin
        attached = attach_external(arr)
        copyto!(out, attached)
        _detach_nonblocking(attached.handle)
    end
    return out
end

(::Type{<:LogicalArray})(arr::Array{B}) where {B} = LogicalArray{B}(arr)
