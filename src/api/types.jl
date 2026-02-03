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
    AutoTask

Represents an automatically scheduled task. Supports adding inputs, outputs, scalars, and constraints.
"""
AutoTask

"""
    ManualTask

Represents a manually scheduled task. Supports adding inputs, outputs, and scalars.
"""
ManualTask

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
    PhysicalArray

A physical array container. Provides access to dimensions, type, and raw data pointer.
"""
PhysicalArray

"""
    LogicalStore{T,N}

Represents a logical view over a physical store. Supports reinterpretation, promotion, slicing, and storage queries.
Wraps the underlying C++ `LogicalStoreImpl`.
"""
struct LogicalStore{T,N}
    handle::LogicalStoreImpl
    dims::Union{Nothing,NTuple{N,Int}}
end

Base.size(s::LogicalStore) = s.dims
Base.size(s::LogicalStore, i::Integer) = size(s)[i]

"""
    LogicalArray{T,N}

A logical view over a physical array. Supports unbound views and nullability checks.
Wraps the underlying C++ `LogicalArrayImpl`.
"""
struct LogicalArray{T,N}
    handle::LogicalArrayImpl
    dims::Union{Nothing,NTuple{N,Int}}
end

Base.size(a::LogicalArray) = a.dims
Base.size(a::LogicalArray, i::Integer) = size(a)[i]

"""
    LegateType

Datatype of object within Legate. See `Legate.supported_types()` to see supported types.
"""
LegateType
