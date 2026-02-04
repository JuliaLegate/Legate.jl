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
    impl = attach_external_store_sysmem(ptr, lshape, to_legate_type(T), read_only)
    return LogicalStore{T,N}(impl, size(arr))
end

# Helper to get the PhysicalStore wrapper from either LogicalStore or LogicalArray
function _get_physical_store(x::LogicalStore, target)
    return Legate.get_physical_store(x, target)
end

function _get_physical_store(x::LogicalArray, target)
    # LogicalArray -> PhysicalArray -> PhysicalStore
    phys_arr = Legate.get_physical_array(x, target)
    return Legate.data(phys_arr)
end

function Base.copyto!(
    dest::Union{LogicalStore{T,N},LogicalArray{T,N}},
    src::Union{LogicalStore{T,N},LogicalArray{T,N}},
) where {T,N}
    dest_handle = (dest isa LogicalArray) ? Legate.data(dest.handle) : dest.handle
    src_handle = (src isa LogicalArray) ? Legate.data(src.handle) : src.handle

    Legate.issue_copy(dest_handle, src_handle)
    return dest
end

# conversion from LogicalArray to Base Julia array
function (::Type{<:Array{A}})(arr::LogicalArray{B}) where {A,B}
    dims = Base.size(arr)
    out = Array{A}(undef, dims)

    # Wait for all UFI tasks to complete before retrieving results
    # This ensures any tasks writing to 'arr' are done
    Legate.wait_ufi()

    # Use read_only=false so Legate knows it's a mutable allocation
    attached = Legate.attach_external(out; read_only=false)
    copyto!(attached, arr)
    Legate.detach(attached.handle)
    return out
end

# Default: infer element type from LogicalArray
(::Type{<:Array})(arr::LogicalArray{B}) where {B} = Array{B}(arr)

# conversion from Base Julia array to LogicalArray
function (::Type{<:LogicalArray{A}})(arr::Array{B}) where {A,B}
    dims = Base.size(arr)
    out = Legate.create_array(A, dims)
    attached = Legate.attach_external(arr)
    copyto!(out, attached)
    Legate.detach(attached.handle)
    return out
end

# Default type: infer from input array
(::Type{<:LogicalArray})(arr::Array{B}) where {B} = LogicalArray{B}(arr)
