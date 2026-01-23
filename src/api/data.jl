"""
    string_to_scalar(str::AbstractString) -> Scalar

Convert a string to a `Scalar`.
"""
string_to_scalar

"""
    create_unbound_array(ty::LegateType;
                         dim::Integer=1; 
                         nullable::Bool=false) -> LogicalArray

Create an unbound array.

# Arguments
- `ty`: Element type of the array.
- `dim`: Number of dimensions.
- `nullable`: Whether the array can contain null values.
"""
function create_unbound_array(ty::LegateType; dim::Integer=1, nullable::Bool=false)
    create_unbound_array(ty, dim, nullable) # cxxwrap call
end

"""
    create_array(shape::Shape, ty::LegateType;
                 nullable::Bool=false,
                 optimize_scalar::Bool=false) -> LogicalArray

Create an array with a specified shape.

# Arguments
- `shape`: Shape of the array.
- `ty`: Element type.
- `nullable`: Whether the array can contain null values.
- `optimize_scalar`: Whether to optimize scalar storage.
"""
function create_array(shape::Shape, ty::LegateType;
    nullable::Bool=false,
    optimize_scalar::Bool=false)
    create_array(shape, ty, nullable, optimize_scalar) # cxxwrap call
end

"""
    create_unbound_store(ty::LegateType;
                         dim::Integer=1) -> LogicalStore

Create an unbound store.

# Arguments
- `ty`: Element type of the store.
- `dim`: Dimensionality of the store.
"""
function create_unbound_store(ty::LegateType; dim::Integer=1)
    create_unbound_store(ty, dim) # cxxwrap call
end

"""
    create_store(shape::Shape, ty::LegateType;
                 optimize_scalar::Bool=false) -> LogicalStore

Create a store with a specified shape.

# Arguments
- `shape`: Shape of the store.
- `ty`: Element type.
- `optimize_scalar`: Whether to optimize scalar storage.
"""
function create_store(shape::Shape, ty::LegateType;
    optimize_scalar::Bool=false)
    create_store(shape, ty, optimize_scalar) # cxxwrap call
end

"""
    store_from_scalar(scalar::Scalar;
                      shape::Shape=Shape(1)) -> LogicalStore

Create a store from a scalar value.

# Arguments
- `scalar`: Scalar value to store.
- `shape`: Shape of the resulting store.
"""
function store_from_scalar(scalar::Scalar; shape::Shape=Shape([1]))
    store_from_scalar(scalar, shape) # cxxwrap call
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
    reinterpret_as(LogicalStore, T::LegateType) -> LogicalStore

Return a view of the logical store reinterpreted as type `T`.
"""
reinterpret_as

"""
    promote(LogicalStore, T::LegateType) -> LogicalStore

Return a new logical store with elements promoted to type `T`.
"""
promote

"""
    slice(LogicalStore, indices...) -> LogicalStore

Return a sliced view of the logical store according to the given indices.
"""
slice

"""
    get_physical_store(LogicalStore) -> PhysicalStore

Return the underlying physical store of this logical store.
"""
get_physical_store

"""
    equal_storage(store1::LogicalStore, store2::LogicalStore) -> Bool

Check if two logical stores refer to the same underlying physical store.
"""
equal_storage

"""
    nullable(LogicalArray) -> Bool
    nullable(PhysicalArray) -> Bool

Check if the array supports null values.
"""
nullable

"""
    data(PhysicalArray) -> Ptr{T}

Return a pointer to the raw data of the physical array.
"""
data

"""
    unbound(LogicalArray) -> Bool

Check if the logical array is unbound (not tied to a physical store).
"""
unbound