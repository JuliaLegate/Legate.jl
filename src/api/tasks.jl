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

function wrap_task(f::Function; is_gpu=false)
    if is_gpu
        return JuliaGPUTask(f, 0)
    else
        return JuliaCPUTask(f, 0)
    end
end

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
    push!(task.input_types, Array{T, N})
    push!(task.arg_dims, size(array))
    return LegateInternal.add_input(task.impl, array.handle)
end

function add_output(task::LegateTask, array::LogicalArray{T, N}) where {T, N}
    push!(task.output_types, Array{T, N})
    push!(task.arg_dims, size(array))
    return LegateInternal.add_output(task.impl, array.handle)
end

function add_scalar(task::LegateTask, scalar::Scalar{T}) where T
    push!(task.scalar_types, T)
    LegateInternal.add_scalar(task.impl, scalar.impl)
end

function align(a::Variable, b::Variable)
    LegateInternal.align(a, b)
end

function default_alignment(task::AutoTask, inputs::Vector{<:Variable}, outputs::Vector{<:Variable})
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
function add_constraint(task::AutoTask, c::Constraint)
    LegateInternal.add_constraint(task.impl, c)
end


function create_julia_task(rt::CxxPtr{Runtime}, lib::Library, task_obj::JuliaTask)
    id = (task_obj isa JuliaCPUTask) ? LegateInternal.JULIA_CUSTOM_TASK : LegateInternal.JULIA_CUSTOM_GPU_TASK
    task = create_task(rt, lib, id)
    task.fun = task_obj.fun

    # ONLY Julia tasks get the task_id scalar and tracking
    task.task_id = Threads.atomic_add!(NEXT_TASK_ID, UInt32(1))
    
    # Prepend internal task_id as scalar 0 on cpp side
    LegateInternal.add_scalar(task.impl, Scalar(UInt32(task.task_id)).impl)
    return task
end


function _submit_task(t::CxxPtr{Runtime}, task::AutoTask)
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
        dims_val = Tuple(task.arg_dims)
        
        sig = UfiSignature{in_t, out_t, sc_t, dims_val}()
        meta = UfiMetadata(task.fun, sig)
        
        lock(REGISTRY_LOCK) do
            GLOBAL_TASK_REGISTRY[task.task_id] = meta
        end

        # Track submitted count ONLY for Julia tasks
        Threads.atomic_add!(SUBMITTED_COUNT, 1)
        
        # Precompile the dispatch engine for this signature
        try
            precompile(Legate._extract_and_call, (UfiMetadata, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, typeof(sig)))
        catch e; end
    end

    @debug "Submitting task $(task.task_id)"
    _submit_task(rt, task)
end
