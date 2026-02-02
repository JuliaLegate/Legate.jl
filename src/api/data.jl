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
string_to_scalar

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
    impl = create_unbound_array(to_legate_type(ty), dim, nullable) # cxxwrap call
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
    lshape = Legate.Shape(to_cxx_vector(shape)) # convert to CxxWrap type
    impl = create_array(lshape, to_legate_type(ty), nullable, optimize_scalar) # cxxwrap call
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
    impl = create_unbound_store(to_legate_type(ty), dim) # cxxwrap call
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
    lshape = Legate.Shape(to_cxx_vector(shape)) # convert to CxxWrap type
    impl = create_store(lshape, to_legate_type(ty), optimize_scalar) # cxxwrap call
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
    lshape = Legate.Shape(to_cxx_vector(shape)) # convert to CxxWrap type   
    impl = store_from_scalar(Legate.Scalar(scalar), lshape) # cxxwrap call
    return LogicalStore{T,length(shape)}(impl, Tuple(shape))
end

"""
    dim(PhysicalStore) -> Int
    dim(LogicalStore) -> Int
    dim(LogicalArray) -> Int
    dim(PhysicalArray) -> Int

Return the number of dimensions of the array/store.
"""
dim

"""
    type(PhysicalStore) -> LegateType
    type(LogicalStore) -> LegateType
    type(LogicalArray) -> LegateType
    type(PhysicalArray) -> LegateType
    
Return the data type of elements stored in the array/store.
"""
type

"""
    code(ty::LegateType) -> Int

Return the internal code representing the `LegateType`.
"""
code

"""
    is_readable(PhysicalStore) -> Bool

Check if the physical store can be read.
"""
is_readable

"""
    is_writable(PhysicalStore) -> Bool

Check if the physical store can be written to.
"""
is_writable

"""
    is_reducible(PhysicalStore) -> Bool

Check if the physical store supports reduction operations.
"""
is_reducible

"""
    valid(PhysicalStore) -> Bool

Check if the physical store is in a valid state.
"""
valid

"""
    reinterpret_as(LogicalStore, T) -> LogicalStore

Return a view of the logical store reinterpreted as type `T`.
"""
function reinterpret_as(
    store::LogicalStore, ::Type{T}
) where {T<:SUPPORTED_TYPES}
    impl = reinterpret_as(store.handle, to_legate_type(T)) # cxxwrap call
    return LogicalStore{T,dim(impl)}(impl, store.dims)
end

"""
    promote(LogicalStore, T) -> LogicalStore

Return a new logical store with elements promoted to type `T`.
"""
function promote(
    store::LogicalStore, ::Type{T}
) where {T<:SUPPORTED_TYPES}
    impl = promote(store.handle, to_legate_type(T)) # cxxwrap call
    return LogicalStore{T,dim(impl)}(impl, store.dims)
end

"""
    slice(LogicalStore, indices...) -> LogicalStore

Return a sliced view of the logical store according to the given indices.
"""
slice

"""
    get_physical_store(LogicalStore) -> PhysicalStore
    get_physical_store(LogicalArray) -> PhysicalStore

Return the underlying physical store of this logical store or array.
"""
function get_physical_store(x::LogicalStore)
    get_physical_store(x.handle, StoreTargetOptional{StoreTarget}())
end
function get_physical_store(x::LogicalStore, target::StoreTarget)
    get_physical_store(x.handle, StoreTargetOptional{StoreTarget}(target))
end

function get_physical_array(x::LogicalArray)
    get_physical_array(x.handle, StoreTargetOptional{StoreTarget}())
end
function get_physical_array(x::LogicalArray, target::StoreTarget)
    get_physical_array(x.handle, StoreTargetOptional{StoreTarget}(target))
end

"""
    equal_storage(store1::LogicalStore, store2::LogicalStore) -> Bool

Check if two logical stores refer to the same underlying physical store.
"""
equal_storage(x::LogicalStore, y::LogicalStore) = equal_storage(x.handle, y.handle) # cxxwrap call

"""
    nullable(LogicalArray) -> Bool
    nullable(PhysicalArray) -> Bool

Check if the array supports null values.
"""
nullable(x::LogicalArray) = nullable(x.handle) # cxxwrap call

"""
    data(PhysicalArray) -> PhysicalStore
    data(LogicalArray) -> LogicalStore

Return the underlying store of the array (physical or logical).
"""
data(x::LogicalArray) = data(x.handle) # cxxwrap call

"""
    unbound(LogicalArray) -> Bool

Check if the logical array is unbound (not tied to a physical store).
"""
unbound(x::LogicalArray) = unbound(x.handle) # cxxwrap call

# Delegation for wrappers
Base.eltype(x::Union{LogicalArray{T},LogicalStore{T}}) where {T} = T

"""
    get_ptr(LogicalStore) -> Ptr
    get_ptr(LogicalArray) -> Ptr
    get_ptr(PhysicalArray) -> Ptr
    get_ptr(PhysicalStore) -> Ptr

Return the pointer to the underlying data of the array/store.
"""
function get_ptr(arr::LogicalStore)
    # LogicalStore -> PhysicalStore -> Ptr
    return get_ptr(get_physical_store(arr)) # cxxwrap call  
end

function get_ptr(arr::LogicalStore, target::StoreTarget)
    return get_ptr(get_physical_store(arr, target))
end

function get_ptr(arr::LogicalArray)
    # LogicalArray -> PhysicalArray -> PhysicalStore -> Ptr
    return get_ptr(data(get_physical_array(arr))) # cxxwrap call
end

function get_ptr(arr::LogicalArray, target::StoreTarget)
    return get_ptr(data(get_physical_array(arr, target)))
end

function get_ptr(arr::PhysicalArray)
    # PhysicalArray -> PhysicalStore -> Ptr
    return get_ptr(data(arr)) # cxxwrap call
end

function get_ptr(arr::PhysicalStore)
    # PhysicalStore -> Ptr
    return _get_ptr(CxxWrap.CxxPtr(arr)) # cxxwrap call
end
