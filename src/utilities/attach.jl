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
    impl = attach_external_store_sysmem(ptr, shape, Legate.to_legate_type(T))
    return LogicalArray{T,N}(impl, size(arr))
end

# Wrap a raw pointer into an Array view
function make_array(::Type{T}, ptr::Ptr{T}, shape::NTuple{N,Int}) where {T,N}
    return unsafe_wrap(Array{T,N}, ptr, shape; own=false)
end

# conversion from LogicalArray to Base Julia array
# get_ptr is a blocking call that grabs the physical store
# we have not tested across multiple processes or devices yet
function (::Type{<:Array{A}})(arr::LogicalArray{B}) where {A,B}
    dims = Base.size(arr)
    ptr = Ptr{A}(get_ptr(arr))
    return make_array(A, ptr, dims)
end

function (::Type{<:Array})(arr::LogicalArray{B}) where {B}
    dims = Base.size(arr)
    ptr = Ptr{B}(get_ptr(arr))
    return make_array(B, ptr, dims)
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
