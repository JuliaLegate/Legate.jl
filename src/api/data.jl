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

to_cxx_vector(shape) = CxxWrap.StdVector([UInt64(d) for d in shape])
to_string(ty::LegateType) = code_type_map[code(ty)]

function Base.show(io::IO, ty::LegateType)
    println(io, code_type_map[code(ty)])
end

function Base.print(ty::LegateType)
    Base.show(stdout, ty)
end

"""
    supported_types()

See Legate.jl supported data types
"""
supported_types() = SUPPORTED_TYPES

"""
    string_to_scalar(str::AbstractString) -> Scalar

Convert a string to a `Scalar`.
"""
function string_to_scalar(str::AbstractString)
    LegateInternal.string_to_scalar(str)
end

"""
    create_array(ty::LegateType; dim::Integer=1; 
                 nullable::Bool=false) -> LogicalArray

Create an unbound array.

# Arguments
- `ty`: Element type of the array.
- `dim`: Number of dimensions.
- `nullable`: Whether the array can contain null values.
"""
function create_array(ty::Type{T}; dim::Integer=1, nullable::Bool=false) where {T<:SUPPORTED_TYPES}
    impl = LegateInternal.create_unbound_array(to_legate_type(ty), dim, nullable)
    return LogicalArray{T,dim}(impl, nothing)
end

"""
    create_array(shape::Vector{B}, ty::Type{T};
                 nullable::Bool=false,
                 optimize_scalar::Bool=false) 
    where {T<:SUPPORTED_TYPES, B<:Integer} -> LogicalArray

Create an array with a specified shape.

# Arguments
- `shape`: Shape of the array.
- `ty`: Element type.
- `nullable`: Whether the array can contain null values.
- `optimize_scalar`: Whether to optimize scalar storage.
"""
function create_array(shape::Vector{B}, ty::Type{T};
    nullable::Bool=false,
    optimize_scalar::Bool=false) where {T<:SUPPORTED_TYPES,B<:Integer}
    lshape = Legate.Shape(to_cxx_vector(shape))
    impl = LegateInternal.create_array(lshape, to_legate_type(ty), nullable, optimize_scalar)
    return LogicalArray{T,length(shape)}(impl, Tuple(shape))
end

"""
    create_store(ty::Type{T}; dim::Integer=1) -> LogicalStore

Create an unbound store.

# Arguments
- `ty`: Element type of the store.
- `dim`: Dimensionality of the store.
"""
function create_store(ty::Type{T}; dim::Integer=1) where {T<:SUPPORTED_TYPES}
    impl = LegateInternal.create_unbound_store(to_legate_type(ty), dim)
    return LogicalStore{T,dim}(impl, nothing)
end

"""
    create_store(shape::Vector{B}, ty::Type{T};
                 optimize_scalar::Bool=false) 
    where {T<:SUPPORTED_TYPES, B<:Integer} -> LogicalStore

Create a store with a specified shape.

# Arguments
- `shape`: Shape of the store.
- `ty`: Element type.
- `optimize_scalar`: Whether to optimize scalar storage.
"""
function create_store(shape::Vector{B}, ty::Type{T};
    optimize_scalar::Bool=false) where {T<:SUPPORTED_TYPES,B<:Integer}
    lshape = Legate.Shape(to_cxx_vector(shape))
    impl = LegateInternal.create_store(lshape, to_legate_type(ty), optimize_scalar)
    return LogicalStore{T,length(shape)}(impl, Tuple(shape))
end

"""
    create_store(scalar::T; shape::Vector{B}=[1]) 
    where {T<:SUPPORTED_TYPES, B<:Integer} -> LogicalStore

Create a store from a scalar value.

# Arguments
- `scalar`: Scalar value to store.
- `shape`: Shape of the resulting store.
"""
function create_store(scalar::T; shape::Vector{B}=[1]) where {T<:SUPPORTED_TYPES,B<:Integer}
    lshape = Legate.Shape(to_cxx_vector(shape))
    impl = LegateInternal.store_from_scalar(Legate.Scalar(scalar), lshape)
    return LogicalStore{T,length(shape)}(impl, Tuple(shape))
end

"""
    dim(PhysicalStore) -> Int
    dim(LogicalStore) -> Int
    dim(LogicalArray) -> Int
    dim(PhysicalArray) -> Int

Return the number of dimensions of the array/store.
"""
dim(x::LogicalArray) = LegateInternal.dim(x.handle)
dim(x::LogicalStore) = LegateInternal.dim(x.handle)
dim(x::PhysicalArray) = LegateInternal.dim(x.handle)
dim(x::PhysicalStore) = LegateInternal.dim(x.handle)

"""
    type(PhysicalStore) -> LegateType
    type(LogicalStore) -> LegateType
    type(LogicalArray) -> LegateType
    type(PhysicalArray) -> LegateType
    
Return the data type of elements stored in the array/store.
"""
type(x::LogicalArray) = LegateInternal.type(x.handle)
type(x::LogicalStore) = LegateInternal.type(x.handle)
type(x::PhysicalArray) = LegateInternal.type(x.handle)
type(x::PhysicalStore) = LegateInternal.type(x.handle)

"""
    code(ty::LegateType) -> Int

Return the internal code representing the `LegateType`.
"""
code(ty::LegateType) = LegateInternal.code(ty)

"""
    is_readable(PhysicalStore) -> Bool

Check if the physical store can be read.
"""
is_readable(s::PhysicalStore) = LegateInternal.is_readable(s)

"""
    is_writable(PhysicalStore) -> Bool

Check if the physical store can be written to.
"""
is_writable(s::PhysicalStore) = LegateInternal.is_writable(s)

"""
    is_reducible(PhysicalStore) -> Bool

Check if the physical store supports reduction operations.
"""
is_reducible(s::PhysicalStore) = LegateInternal.is_reducible(s)

"""
    valid(PhysicalStore) -> Bool

Check if the physical store is in a valid state.
"""
valid(s::PhysicalStore) = LegateInternal.valid(s)

"""
    reinterpret_as(LogicalStore, T) -> LogicalStore

Return a view of the logical store reinterpreted as type `T`.
"""
function reinterpret_as(
    store::LogicalStore, ::Type{T}
) where {T<:SUPPORTED_TYPES}
    impl = LegateInternal.reinterpret_as(store.handle, to_legate_type(T))
    return LogicalStore{T,Int(LegateInternal.dim(impl))}(impl, store.dims)
end

"""
    promote(LogicalStore, T) -> LogicalStore

Return a new logical store with elements promoted to type `T`.
"""
function promote(
    store::LogicalStore, ::Type{T}
) where {T<:SUPPORTED_TYPES}
    impl = LegateInternal.promote(store.handle, to_legate_type(T))
    return LogicalStore{T,Int(LegateInternal.dim(impl))}(impl, store.dims)
end

"""
    slice(LogicalStore, indices...) -> LogicalStore

Return a sliced view of the logical store according to the given indices.
"""
function slice(store::LogicalStore, indices...)
    # Convert indices to Slice or appropriate type if needed
    # Assuming LegateInternal.slice handles it
    impl = LegateInternal.slice(store.handle, indices...)
    return LogicalStore{eltype(store),Int(LegateInternal.dim(impl))}(impl, nothing)
end

"""
    get_physical_store(LogicalStore) -> PhysicalStore
    get_physical_store(LogicalArray) -> PhysicalStore

Return the underlying physical store of this logical store or array.
"""
function get_physical_store(x::LogicalStore)
    LegateInternal.get_physical_store(x.handle, StoreTargetOptional{StoreTarget}())
end
function get_physical_store(x::LogicalStore, target::StoreTarget)
    LegateInternal.get_physical_store(x.handle, StoreTargetOptional{StoreTarget}(target))
end

function get_physical_array(x::LogicalArray)
    LegateInternal.get_physical_array(x.handle, StoreTargetOptional{StoreTarget}())
end
function get_physical_array(x::LogicalArray, target::StoreTarget)
    LegateInternal.get_physical_array(x.handle, StoreTargetOptional{StoreTarget}(target))
end

equal_storage(x::LogicalStore, y::LogicalStore) = LegateInternal.equal_storage(x.handle, y.handle)

nullable(x::LogicalArray) = LegateInternal.nullable(x.handle)
nullable(x::PhysicalArray) = LegateInternal.nullable(x.handle)

data(x::LogicalArray) = LegateInternal.data(x.handle)
data(x::PhysicalArray) = LegateInternal.data(x.handle)

unbound(x::LogicalArray) = LegateInternal.unbound(x.handle)

# Delegation for wrappers
Base.eltype(x::Union{LogicalArray{T},LogicalStore{T}}) where {T} = T


"""
    get_ptr(LogicalStore) -> Ptr
    get_ptr(LogicalArray) -> Ptr
    get_ptr(PhysicalArray) -> Ptr
    get_ptr(PhysicalStore) -> Ptr

Return the pointer to the underlying data of the array/store.
"""
get_ptr(arr::LogicalStore) = get_ptr(get_physical_store(arr))
get_ptr(arr::LogicalStore, target::StoreTarget) = get_ptr(get_physical_store(arr, target))

get_ptr(arr::LogicalArray) = get_ptr(data(get_physical_array(arr)))
get_ptr(arr::LogicalArray, target::StoreTarget) = get_ptr(data(get_physical_array(arr, target)))

get_ptr(arr::PhysicalArray) = get_ptr(data(arr))
get_ptr(arr::PhysicalStore) = LegateInternal._get_ptr(CxxWrap.CxxPtr(arr))
