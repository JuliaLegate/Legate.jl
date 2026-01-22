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
    submit_auto_task(rt::Runtime, task::AutoTask)

Submit an auto task to the runtime.
"""
submit_auto_task

"""
    submit_manual_task(rt::Runtime, task::ManualTask)

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
    create_unbound_array(ty::Type;
                         dim::Integer=1,
                         nullable::Bool=false) -> LogicalArray

Create an unbound array.

# Arguments
- `ty`: Element type of the array.
- `dim`: Number of dimensions.
- `nullable`: Whether the array can contain null values.
"""
create_unbound_array

"""
    create_array(shape::Shape, ty::Type;
                 nullable::Bool=false,
                 optimize_scalar::Bool=false) -> LogicalArray

Create an array with a specified shape.

# Arguments
- `shape`: Shape of the array.
- `ty`: Element type.
- `nullable`: Whether the array can contain null values.
- `optimize_scalar`: Whether to optimize scalar storage.
"""
create_array

"""
    create_unbound_store(ty::Type;
                         dim::Integer=1) -> LogicalStore

Create an unbound store.

# Arguments
- `ty`: Element type of the store.
- `dim`: Dimensionality of the store.
"""
create_unbound_store

"""
    create_store(shape::Shape, ty::Type;
                 optimize_scalar::Bool=false) -> LogicalStore

Create a store with a specified shape.

# Arguments
- `shape`: Shape of the store.
- `ty`: Element type.
- `optimize_scalar`: Whether to optimize scalar storage.
"""
create_store

"""
    store_from_scalar(scalar::Scalar;
                      shape::Shape=Shape(1)) -> LogicalStore

Create a store from a scalar value.

# Arguments
- `scalar`: Scalar value to store.
- `shape`: Shape of the resulting store.
"""
store_from_scalar

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
