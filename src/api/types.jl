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

abstract type TaskBackend end
struct CPUBackend <: TaskBackend end
struct GPUBackend <: TaskBackend end

struct JuliaTask{B<:TaskBackend,F}
    fun::F
    task_id::UInt32
end

function wrap_task(f, ::Type{CPUBackend})
    assert_experimental()
    return JuliaTask{CPUBackend,typeof(f)}(f, 0)
end

function wrap_task(f, ::Type{GPUBackend})
    assert_experimental()
    return JuliaTask{GPUBackend,typeof(f)}(f, 0)
end

mutable struct LegateTask{I,F}
    impl::I
    fun::F
    task_id::UInt32
    input_types::Vector{DataType}
    output_types::Vector{DataType}
    scalar_types::Vector{DataType}
    arg_dims::Vector{Union{Nothing,NTuple}}
    is_gpu::Bool
end

function LegateTask(impl::I, fun::F) where {I,F}
    return LegateTask{I,F}(
        impl, fun, UInt32(0), DataType[], DataType[], DataType[], Union{Nothing,NTuple}[], false
    )
end

const AutoTask = LegateTask{AutoTaskImpl}
const ManualTask = LegateTask{ManualTaskImpl}

function AutoTask(impl::LegateInternal.AutoTaskAllocated)
    @debug "IMPL: Creating auto task $(impl)"
    return LegateTask{AutoTaskImpl}(impl)
end

function ManualTask(impl::LegateInternal.ManualTaskAllocated)
    @debug "IMPL: Creating manual task $(impl)"
    return LegateTask{ManualTaskImpl}(impl)
end

struct UfiSignature{InT,OutT,ScT} end

struct UfiMetadata{F,S,D}
    fun::F
    sig::S
    dims::D
end

struct Scalar{T}
    impl::ScalarImpl
end

function Scalar(x::T) where {T<:SUPPORTED_TYPES}
    r = Ref(x)
    impl = GC.@preserve r begin
        ptr = Base.unsafe_convert(Ptr{Cvoid}, r)
        LegateInternal.make_scalar(ptr, to_legate_type(T))
    end
    return Scalar{T}(impl)
end

"""
    Shape

Represents the dimensions of an array or store. Can be constructed from a vector of `UInt64`.
"""
Shape

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
Wraps the underlying C++ `LogicalArrayImpl`. `order` is the store's buffer layout: `:row`
(C, cuNumeric-native) or `:col` (Fortran); `Array` uses it to convert back to Julia.
"""
struct LogicalArray{T,N}
    handle::LogicalArrayImpl
    dims::Union{Nothing,NTuple{N,Int}}
    order::Symbol
end

function LogicalArray{T,N}(handle::LogicalArrayImpl, dims) where {T,N}
    return LogicalArray{T,N}(handle, dims, :row)
end

Base.size(a::LogicalArray) = a.dims
Base.size(a::LogicalArray, i::Integer) = size(a)[i]

"""
    LegateType

Datatype of object within Legate. See `Legate.supported_types()` to see supported types.
"""
LegateType

"""
    LogicalStorePartition{T,N}
Represents a tiled partition of a `LogicalStore`. Created via `partition_by_tiling`.
Wraps the underlying C++ `LogicalStorePartitionImpl`.
"""
struct LogicalStorePartition{T,N}
    handle::CxxWrap.StdLib.SharedPtr{LogicalStorePartitionImpl}
end
