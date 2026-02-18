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

function Base.copyto!(
    dest::Union{LogicalStore{T,N},LogicalArray{T,N}},
    src::Union{LogicalStore{T,N},LogicalArray{T,N}},
) where {T,N}
    dest_handle = (dest isa LogicalArray) ? LegateInternal.data(dest.handle) : dest.handle
    src_handle = (src isa LogicalArray) ? LegateInternal.data(src.handle) : src.handle

    # @threadcall avoids deadlock: issue_copy can block in C++ waiting for
    # Legion events that depend on UFI tasks still in the Julia event loop.
    Base.@threadcall(
        (:legate_issue_copy, Legate.WRAPPER_LIB_PATH),
        Cvoid,
        (Ptr{Cvoid}, Ptr{Cvoid}),
        dest_handle.cpp_object, src_handle.cpp_object
    )
    return dest
end

function Base.copyto!(dest::LogicalArray{T,N}, src::Array{T,N}) where {T,N}
    attached = attach_external(src)
    copyto!(dest, attached)
    _detach_nonblocking(attached.handle)
    return dest
end

# conversion from LogicalArray to Base Julia array
function (::Type{<:Array{A}})(arr::LogicalArray{B}) where {A,B}
    Legate.wait_ufi()
    dims = Base.size(arr)
    out = Array{A}(undef, dims)
    attached = attach_external(out; read_only=false)
    copyto!(attached, arr)
    _detach_nonblocking(attached.handle)
    return out
end

# Detach without blocking Thread 1.
function _detach_nonblocking(store_handle::LogicalStoreImpl)
    pstore = LegateInternal.get_physical_store(store_handle, LegateInternal.StoreTargetOptional{Legate.StoreTarget}())

    # Drive UFI while waiting for pending tasks that may affect this store
    while Legate.ufi_has_pending_work()
        Legate.ufi_poll_sync()
        yield()
    end

    # @threadcall ensures we don't block Julia if C++ does internal sync
    Base.@threadcall((:legate_logical_store_detach, Legate.WRAPPER_LIB_PATH), Cvoid, (Ptr{Cvoid},), store_handle.cpp_object)
end


(::Type{<:Array})(arr::LogicalArray{B}) where {B} = Array{B}(arr)

function (::Type{<:LogicalArray{A}})(arr::Array{B}) where {A,B}
    dims = Base.size(arr)
    out = Legate.create_array(collect(dims), A)
    attached = Legate.attach_external(arr)
    copyto!(out, attached)
    _detach_nonblocking(attached.handle)
    return out
end

(::Type{<:LogicalArray})(arr::Array{B}) where {B} = LogicalArray{B}(arr)
