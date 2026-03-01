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

using CxxWrap

const REGISTRY_LOCK = ReentrantLock()
const GLOBAL_TASK_REGISTRY = Dict{UInt32, UfiMetadata}()

const SUBMITTED_COUNT = Threads.Atomic{Int}(0)
const NEXT_TASK_ID = Threads.Atomic{UInt32}(50000)

function create_task(rt::CxxPtr{Runtime}, lib::Library, id::LocalTaskID)
    impl = LegateInternal.create_auto_task(rt, lib, id)
    @debug "Creating auto task $(impl)"
    task = AutoTask(impl)
    return task
end

function create_task(rt::CxxPtr{Runtime}, lib::Library, id::LocalTaskID, domain::Domain)
    impl = LegateInternal.create_manual_task(rt, lib, id, domain)
    @debug "Creating manual task $(impl)"
    task = ManualTask(impl)
    return task
end

function add_input(task::LegateTask, array::LogicalArray{T, N}) where {T, N}
    push!(task.input_types, T)
    push!(task.arg_dims, array.dims)
    LegateInternal.add_input(task.impl, array.handle)
end

function add_output(task::LegateTask, array::LogicalArray{T, N}) where {T, N}
    push!(task.output_types, T)
    push!(task.arg_dims, array.dims)
    LegateInternal.add_output(task.impl, array.handle)
end

function add_scalar(task::LegateTask, scalar::Scalar{T}) where {T}
    push!(task.scalar_types, T)
    LegateInternal.add_scalar(task.impl, scalar.impl)
end

function align(a::Variable, b::Variable)
    LegateInternal.align(a, b)
end

function default_alignment(task::LegateTask, inputs::Vector{<:Variable}, outputs::Vector{<:Variable})
    for i in 2:length(inputs)
        add_constraint(task, align(inputs[i], inputs[1]))
    end
    for i in 1:length(outputs)
        add_constraint(task, align(outputs[i], outputs[1]))
    end
    if !isempty(inputs) && !isempty(outputs)
        add_constraint(task, align(outputs[1], inputs[1]))
    end
end

"""
    add_constraint(AutoTask, c::Constraint)

Add a constraint to the task.
"""
function add_constraint(task::LegateTask, c::Constraint)
    LegateInternal.add_constraint(task.impl, c)
end


function create_julia_task(rt, lib, task_obj::JuliaTask{CPUBackend})
    create_julia_task_impl(rt, lib, task_obj, 0)
end

function create_julia_task(rt, lib, task_obj::JuliaTask{GPUBackend})
    create_julia_task_impl(rt, lib, task_obj, 1)
end


function create_julia_task_impl(rt, lib, task_obj, backend_flag::Cint)
    # returns an Legate AutoTask object ptr
    impl_ptr = ccall((:legate_create_julia_task_wrapper, Legate.WRAPPER_LIB_PATH), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Cint), rt.cpp_object, lib.cpp_object, backend_flag)
    task = LegateTask(CxxWrap.CxxPtr{LegateInternal.AutoTask}(impl_ptr), task_obj.fun)
    task.task_id = Threads.atomic_add!(NEXT_TASK_ID, UInt32(1))
    # Prepend internal task_id as scalar 0 on cpp Legate side
    LegateInternal.add_scalar(task.impl, Scalar(UInt32(task.task_id)).impl)
    return task
end

function _submit_task(t::CxxPtr{Runtime}, task::LegateTask{<:CxxPtr{<:LegateInternal.AutoTask}})
    LegateInternal.submit_auto_task(t, task.impl[])
end

function _submit_task(t::CxxPtr{Runtime}, task::LegateTask{<:LegateInternal.AutoTask})
    LegateInternal.submit_auto_task(t, task.impl)
end

function _submit_task(t::CxxPtr{Runtime}, task::ManualTask)
    LegateInternal.submit_manual_task(t, task.impl)
end


function submit_task(rt::CxxPtr{Runtime}, task::LegateTask)
    if !isnothing(task.fun)
        in_t = Tuple{task.input_types...}
        out_t = Tuple{task.output_types...}
        sc_t = Tuple{task.scalar_types...}
        
        sig = UfiSignature{in_t, out_t, sc_t}()
        meta = UfiMetadata(task.fun, sig, Tuple(task.arg_dims))
        
        lock(REGISTRY_LOCK) do
            GLOBAL_TASK_REGISTRY[task.task_id] = meta
        end
        
        # Principled warmup: Force JIT compilation safely on submission thread
        # 1. Precompile the internal statically-typed dispatcher
        precompile(Legate._do_call, (typeof(task.fun), Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, typeof(meta.dims), typeof(sig)))
        precompile(Legate._extract_and_call, (typeof(meta), Vector{Ptr{Cvoid}}, Vector{Ptr{Cvoid}}, Vector{Ptr{Cvoid}}, typeof(sig)))
        
        # 2. Precompile the user-provided function with exact types
        user_arg_types = Any[]
        for (T, d) in zip(task.input_types, task.arg_dims)
            push!(user_arg_types, Array{T, length(d)})
        end
        for (T, d) in zip(task.output_types, task.arg_dims)
            push!(user_arg_types, Array{T, length(d)})
        end
        for T in task.scalar_types
            push!(user_arg_types, T)
        end
        precompile(task.fun, (user_arg_types...,))
        
        Threads.atomic_add!(SUBMITTED_COUNT, 1)
    end
    _submit_task(rt, task)
end
