"""
    start_legate()

Start the Legate runtime.

This function initializes the Legate runtime and must be called
before creating tasks or data objects.
"""
start_legate

"""
    legate_finish() -> Int32

Finalize the Legate runtime.

Returns an integer status code from the runtime shutdown procedure.
"""
legate_finish

"""
    get_runtime() -> Runtime

Return the current Legate runtime instance.

This returns a handle to the singleton `Runtime` object managed by Legate.
"""
get_runtime

"""
    has_started() -> Bool

Check whether the Legate runtime has started.
"""
has_started

"""
    has_finished() -> Bool

Check whether the Legate runtime has finished.
"""
has_finished

"""
    create_auto_task(rt::Runtime, lib::Library, id::LocalTaskID) -> AutoTask

Create an auto task in the runtime.

# Arguments
- `rt`: The current runtime instance.
- `lib`: The library to associate with the task.
- `id`: The local task identifier.
"""
create_auto_task

"""
    submit_auto_task(rt::Runtime, AutoTask)

Submit an auto task to the runtime.
"""
submit_auto_task

"""
    submit_manual_task(rt::Runtime, ManualTask)

Submit a manual task to the runtime.
"""
submit_manual_task

"""
    string_to_scalar(str::AbstractString) -> Scalar

Convert a string to a `Scalar`.
"""
string_to_scalar

"""
    align(a::Variable, b::Variable) -> Constraint

Align two variables.

Returns a new constraint representing the alignment of `a` and `b`.
"""
align

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
    time_microseconds() -> UInt64

Measure time in microseconds.
"""
time_microseconds

"""
    time_nanoseconds() -> UInt64

Measure time in nanoseconds.
"""
time_nanoseconds

"""
    Library

Represents a computational or data library. Serves as a container for tasks, arrays, and stores.
"""
Library

"""
    Variable

Represents a variable in the task system, typically produced or consumed by tasks.
"""
Variable

"""
    Constraint

Represents a dependency or restriction for a task, such as ordering or memory constraints.
"""
Constraint

"""
    LocalTaskID

A unique identifier for a task within a single process or node.
"""
LocalTaskID

"""
    GlobalTaskID

A globally unique identifier for a task across processes or nodes.
"""
GlobalTaskID

"""
    StoreTarget

Represents the target storage type or location for a store in the mapping layer.
"""
StoreTarget

"""
    Shape

Represents the dimensions of an array or store. Can be constructed from a vector of `UInt64`.
"""
Shape

"""
    Scalar

Represents a scalar value used in tasks. Can be constructed from `Float32`, `Float64`, or `Int`.
"""
Scalar

"""
    Slice

Represents a slice of an array or store. Can be constructed from optional start and stop indices.
"""
Slice

"""
    PhysicalStore

Represents a physical storage container. Provides methods to query its dimensions, type, and accessibility.
"""
PhysicalStore

"""
    dim(PhysicalStore) -> Int

Return the number of dimensions of the physical store.
"""
dim

"""
    type(PhysicalStore) -> DataType

Return the data type of elements stored in the physical store.
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
    LogicalStore

Represents a logical view over a physical store. Supports reinterpretation, promotion, slicing, and storage queries.
"""
LogicalStore

"""
    dim(LogicalStore) -> Int

Return the number of dimensions of the logical store.
"""
dim

"""
    type(LogicalStore) -> LegateType

Return the data type of elements in the logical store.
"""
type

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
    PhysicalArray

A physical array container. Provides access to dimensions, type, and raw data pointer.
"""
PhysicalArray

"""
    nullable(PhysicalArray) -> Bool

Check if the array supports null values.
"""
nullable

"""
    dim(PhysicalArray) -> Int

Return the number of dimensions of the physical array.
"""
dim

"""
    type(PhysicalArray) -> LegateType

Return the data type of the physical array elements.
"""
type

"""
    data(PhysicalArray) -> Ptr{T}

Return a pointer to the raw data of the physical array.
"""
data

"""
    LogicalArray

A logical view over a physical array. Supports unbound views and nullability checks.
"""
LogicalArray

"""
    dim(LogicalArray) -> Int

Return the number of dimensions of the logical array.
"""
dim

"""
    type(LogicalArray) -> DataType

Return the data type of elements in the logical array.
"""
type

"""
    unbound(LogicalArray) -> Bool

Check if the logical array is unbound (not tied to a physical store).
"""
unbound

"""
    nullable(LogicalArray) -> Bool

Check if the logical array supports null values.
"""
nullable

"""
    AutoTask

Represents an automatically scheduled task. Supports adding inputs, outputs, scalars, and constraints.
"""
AutoTask

"""
    add_input(AutoTask, LogicalArray) -> Variable

Add a logical array as an input to the task.
"""
add_input

"""
    add_output(AutoTask, LogicalArray) -> Variable

Add a logical array as an output of the task.
"""
add_output

"""
    add_scalar(AutoTask, scalar::Scalar)

Add a scalar argument to the task.
"""
add_scalar

"""
    add_constraint(AutoTask, c::Constraint)

Add a constraint to the task.
"""
add_constraint

"""
    ManualTask

Represents a manually scheduled task. Supports adding inputs, outputs, and scalars.
"""
ManualTask

"""
    add_input(ManualTask, LogicalStore)

Add a logical store as an input to the task.
"""
add_input

"""
    add_output(ManualTask, LogicalStore)

Add a logical store as an output of the task.
"""
add_output

"""
    add_scalar(ManualTask, scalar::Scalar)

Add a scalar argument to the task.
"""
add_scalar
