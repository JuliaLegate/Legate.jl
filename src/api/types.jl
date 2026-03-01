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

struct JuliaCPUTask
    fun::Function
    task_id::UInt32
end

struct JuliaGPUTask
    fun::Function
    task_id::UInt32
end

JuliaTask = Union{JuliaCPUTask, JuliaGPUTask}

mutable struct LegateTask{I}
    impl::I
    input_types::Vector{DataType}
    output_types::Vector{DataType}
    scalar_types::Vector{DataType}
    arg_dims::Vector{Union{Nothing, NTuple}}
end

LegateTask(impl::I) where I = LegateTask{I}(impl, DataType[], DataType[], DataType[], Union{Nothing, NTuple}[])

const AutoTask   = LegateTask{AutoTaskImpl}
const ManualTask = LegateTask{ManualTaskImpl}

function AutoTask(impl::LegateInternal.AutoTaskAllocated)
    @debug "IMPL: Creating auto task $(impl)"
    return LegateTask{AutoTaskImpl}(impl)
end

function ManualTask(impl::LegateInternal.ManualTaskAllocated)
    @debug "IMPL: Creating manual task $(impl)"
    return LegateTask{ManualTaskImpl}(impl)
end


"""
    UfiSignature{InT, OutT, ScT, Dims}

A type-level representation of a task's full signature, including argument types
and dimension values. Encoding dimensions as type parameters ensures perfect
type stability and zero-allocation dispatch in background threads.
"""
struct UfiSignature{InT, OutT, ScT, Dims} end

"""
    UfiMetadata

Stores the task function and its static signature.
"""
struct UfiMetadata
    fun::Function
    sig::Any # UfiSignature{...}
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

struct LogicalStore{T,N}
    handle::LogicalStoreImpl
    dims::Union{Nothing,NTuple{N,Int}}
end

Base.size(s::LogicalStore) = s.dims
Base.size(s::LogicalStore, i::Integer) = size(s)[i]

struct LogicalArray{T,N}
    handle::LogicalArrayImpl
    dims::Union{Nothing,NTuple{N,Int}}
end

Base.size(a::LogicalArray) = a.dims
Base.size(a::LogicalArray, i::Integer) = size(a)[i]
