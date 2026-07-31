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

function _attach_external_sysmem(
    arr::Array{T,N},
    shape::Dims{N},
    attach_fn,
) where {T,N}
    prod(shape) == length(arr) || throw(
        DimensionMismatch(
            "attach shape $(shape) volume $(prod(shape)) != array length $(length(arr))"
        ),
    )
    ptr = Base.unsafe_convert(Ptr{Cvoid}, arr)
    lshape = Shape(to_cxx_vector(collect(UInt64, shape)))
    impl = attach_fn(ptr, lshape, to_legate_type(T))
    return LogicalStore{T,N}(impl, shape)
end

"""
    attach_external_row_major(arr::Array; shape=size(arr))

Attach a Julia `Array` as an external Legate store with **row-major (C-order)**
layout. `shape` may differ from `size(arr)` when `prod(shape) == length(arr)`,
e.g. when attaching a transposed buffer that holds C-order bytes for `shape`.
"""
function attach_external_row_major(arr::Array{T,N}; shape::Dims{N}=size(arr)) where {T,N}
    return _attach_external_sysmem(arr, shape, attach_external_store_sysmem_row_major)
end

"""
    attach_external_col_major(arr::Array; shape=size(arr))

Attach a Julia `Array` as an external Legate store with **col-major (Fortran-order)**
layout. Reserved for future use; prefer `attach_external_row_major` for cuNumeric.
"""
function attach_external_col_major(arr::Array{T,N}; shape::Dims{N}=size(arr)) where {T,N}
    return _attach_external_sysmem(arr, shape, attach_external_store_sysmem_col_major)
end

"""
    attach_external(arr::Array)

Attach with row-major ordering (default for Legate/cuNumeric).
"""
function attach_external(arr::Array{T,N}; shape::Dims{N}=size(arr)) where {T,N}
    return attach_external_row_major(arr; shape)
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
    # PhysicalStore accessors must run on the Legate/Legion toplevel task thread.
    # @threadcall uses a libuv worker thread, which 26.06+ rejects with:
    # "Invalid request to wait until a physical region is valid outside of Toplevel Task".
    phys_dest = _get_physical_store(dest, Legate.SYSMEM)
    phys_src = _get_physical_store(src, Legate.SYSMEM)

    dest_ptr = Ptr{T}(Legate.get_ptr(phys_dest))
    src_ptr = Ptr{T}(Legate.get_ptr(phys_src))

    Base.unsafe_copyto!(dest_ptr, src_ptr, prod(size(dest)))
    return dest
end

# Julia F-order buffer of shape reverse(S) has the same bytes as C-order shape S.
function _julia_to_row_major_buffer(arr::Array{T,0}) where {T}
    return arr, size(arr)
end

function _julia_to_row_major_buffer(arr::Array{T,1}) where {T}
    return arr, size(arr)
end

function _julia_to_row_major_buffer(arr::Array{T,N}) where {T,N}
    tmp = collect(permutedims(arr, reverse(ntuple(identity, Val(N)))))
    return tmp, size(arr)
end

function _row_major_buffer_to_julia(tmp::Array{T,0}, shape::Dims{0}) where {T}
    return reshape(tmp, shape)
end

function _row_major_buffer_to_julia(tmp::Array{T,1}, shape::Dims{1}) where {T}
    return reshape(tmp, shape)
end

function _row_major_buffer_to_julia(tmp::Array{T,N}, shape::Dims{N}) where {T,N}
    return collect(permutedims(tmp, reverse(ntuple(identity, Val(N)))))
end

# LogicalArray -> Array. Eltype must match; no implicit cast.
function (::Type{<:Array{A}})(arr::LogicalArray{A,0}) where {A}
    out = Array{A}(undef, size(arr))
    attached = Legate.attach_external_row_major(out)
    copyto!(attached, arr)
    return out
end

function (::Type{<:Array{A}})(arr::LogicalArray{A,1}) where {A}
    out = Array{A}(undef, size(arr))
    attached = Legate.attach_external_row_major(out)
    copyto!(attached, arr)
    return out
end

function (::Type{<:Array{A}})(arr::LogicalArray{A,N}) where {A,N}
    dims = Base.size(arr)
    if arr.order === :col
        # :col buffer already holds col-major bytes for reverse(dims); copy straight.
        out = Array{A}(undef, reverse(dims))
        attached = Legate.attach_external_col_major(out; shape=dims)
        copyto!(attached, arr)
        return out
    end
    # :row: fill an F-order buffer matching C-order bytes, then permute back.
    tmp = Array{A}(undef, reverse(dims))
    attached = Legate.attach_external_row_major(tmp; shape=dims)
    copyto!(attached, arr)
    return _row_major_buffer_to_julia(tmp, dims)
end

# Bare `Array(arr)` uses the store eltype; `Type{Array}` only so a typed mismatch errors.
function (::Type{Array})(arr::LogicalArray{B,N}) where {B,N}
    return Array{B}(arr)
end

# conversion from Base Julia array to LogicalArray. The Julia buffer is transposed to
# row-major (C-order) before attaching, so the resulting store is row-major (`:row`).
function (::Type{<:LogicalArray{A}})(arr::Array{B}) where {A,B}
    dims = Base.size(arr)
    out = Legate.create_array(collect(Int64, dims), A)
    src = A === B ? arr : convert(Array{A}, arr)
    tmp, shape = _julia_to_row_major_buffer(src)
    attached = Legate.attach_external_row_major(tmp; shape)
    copyto!(out, attached)
    return LogicalArray{A,length(dims)}(out.handle, out.dims, :row)
end

function (::Type{<:LogicalArray})(arr::Array{B}) where {B}
    dims = Base.size(arr)
    out = Legate.create_array(collect(Int64, dims), B)
    tmp, shape = _julia_to_row_major_buffer(arr)
    attached = Legate.attach_external_row_major(tmp; shape)
    copyto!(out, attached)
    return LogicalArray{B,length(dims)}(out.handle, out.dims, :row)
end
