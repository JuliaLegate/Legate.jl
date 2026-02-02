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

function attach_external(arr::Array{T,N}) where {T,N}
    ptr = Base.unsafe_convert(Ptr{Cvoid}, arr)
    shape = collect(UInt64, size(arr))
    lshape = Shape(to_cxx_vector(shape))
    impl = attach_external_store_sysmem(ptr, lshape, to_legate_type(T))
    return LogicalStore{T,N}(impl, size(arr))
end

function _get_ptr_offloaded(store, target)
    return get_ptr(store, target)
end

function Base.copyto!(
    dest::Union{LogicalStore{T,N},LogicalArray{T,N}},
    src::Union{LogicalStore{T,N},LogicalArray{T,N}},
) where {T,N}
    if Threads.nthreads() > 1
        dest_ptr = Ptr{T}(fetch(Threads.@spawn _get_ptr_offloaded(dest, Legate.SYSMEM)))
        src_ptr = Ptr{T}(fetch(Threads.@spawn _get_ptr_offloaded(src, Legate.SYSMEM)))
    else
        dest_ptr = Ptr{T}(get_ptr(dest, Legate.SYSMEM))
        src_ptr = Ptr{T}(get_ptr(src, Legate.SYSMEM))
    end
    Base.unsafe_copyto!(dest_ptr, src_ptr, prod(size(dest)))
    return dest
end

# conversion from LogicalArray to Base Julia array
# get_ptr is a blocking call that grabs the physical store
# we have not tested across multiple processes or devices yet
function (::Type{<:Array{A}})(arr::LogicalArray{B}) where {A,B}
    # REMOVED wait_ufi() - it can cause deadlocks if a task is queued but not yet running
    # get_ptr naturally blocks until the store is ready.
    dims = Base.size(arr)
    out = Array{A}(undef, dims)
    attached = Legate.attach_external(out)
    copyto!(attached, arr)
    return out
end

function (::Type{<:Array})(arr::LogicalArray{B}) where {B}
    return Array{B}(arr)
end

# conversion from Base Julia array to LogicalArray
function (::Type{<:LogicalArray{A}})(arr::Array{B}) where {A,B}
    dims = Base.size(arr)
    out = Legate.create_array(A, dims)
    attached = Legate.attach_external(arr)
    copyto!(out, attached) # copy elems of attached to resulting out
    return out
end

function (::Type{<:LogicalArray})(arr::Array{B}) where {B}
    dims = Base.size(arr)
    out = Legate.create_array(B, dims)
    attached = Legate.attach_external(arr)
    copyto!(out, attached)
    return out
end
