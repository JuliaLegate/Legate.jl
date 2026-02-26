"""
    create_task(rt::Runtime, lib::Library, id::LocalTaskID) -> AutoTask

Create an auto task in the runtime.

# Arguments
- `rt`: The current runtime instance.
- `lib`: The library to associate with the task.
- `id`: The local task identifier.
"""
function create_task(rt::CxxPtr{Runtime}, lib::Library, id::LocalTaskID)
    impl = LegateInternal.create_auto_task(rt, lib, id)
    return AutoTask(impl)
end
function create_task(rt::CxxPtr{Runtime}, lib::Library, id::LocalTaskID, domain::Domain)
    impl = LegateInternal.create_manual_task(rt, lib, id, domain)
    return ManualTask(impl)
end

"""
    submit_task(rt::Runtime, AutoTask)
    submit_task(rt::Runtime, ManualTask)

Submit an manual/auto task to the runtime.
"""
function submit_task(rt::CxxPtr{Runtime}, task::AutoTask)
    # Update High Water Mark for UFI tracking
    Threads.atomic_max!(MAX_SUBMITTED_TASK_ID, Int(LAST_CREATED_TASK_ID[]))

    @threadcall((:legate_submit_auto_task, Legate.WRAPPER_LIB_PATH), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), rt.cpp_object, task.impl.cpp_object)
end
function submit_task(rt::CxxPtr{Runtime}, task::ManualTask)
    Threads.atomic_max!(MAX_SUBMITTED_TASK_ID, Int(LAST_CREATED_TASK_ID[]))
    
    @threadcall((:legate_submit_manual_task, Legate.WRAPPER_LIB_PATH), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), rt.cpp_object, task.impl.cpp_object)
end

"""
    align(a::Variable, b::Variable) -> Constraint

Align two variables.

Returns a new constraint representing the alignment of `a` and `b`.
"""
function align(a::Variable, b::Variable)
    LegateInternal.align(a, b)
end

"""
    default_alignment(task::AutoTask, inputs::Vector{Variable}, outputs::Vector{Variable})

Add default alignment constraints to the task. All inputs and outputs are aligned to the first input.
"""
function default_alignment(
    task::Legate.AutoTask, inputs::Vector{<:Legate.Variable}, outputs::Vector{<:Legate.Variable}
)
    # Align all inputs to the first input
    for i in 2:length(inputs)
        Legate.add_constraint(task, Legate.align(inputs[i], inputs[1]))
    end
    # Align all outputs to the first output
    for i in 2:length(outputs)
        Legate.add_constraint(task, Legate.align(outputs[i], outputs[1]))
    end
    # Align first output with first input
    if !isempty(inputs) && !isempty(outputs)
        Legate.add_constraint(task, Legate.align(outputs[1], inputs[1]))
    end
end

"""
    add_constraint(AutoTask, c::Constraint)

Add a constraint to the task.
"""
function add_constraint(task::AutoTask, c::Constraint)
    LegateInternal.add_constraint(task.impl, c)
end

"""
    add_input(AutoTask, LogicalArray) -> Variable
    add_input(ManualTask, LogicalStore) -> Variable

Add a logical array/store as an input to the task.
"""
function add_input(
    task::Union{AutoTask,ManualTask},
    item::LogicalArray{T, N},
) where {T, N}
    push!(task.arg_types, Array{T, N})
    LegateInternal.add_input(task.impl, item.handle)
end

function add_output(
    task::Union{AutoTask,ManualTask},
    item::LogicalArray{T, N},
) where {T, N}
    push!(task.arg_types, Array{T, N})
    LegateInternal.add_output(task.impl, item.handle)
end

function add_scalar(task::Union{AutoTask,ManualTask}, scalar::ScalarImpl)
    # Note: We don't easily have the Julia type here unless we wrap Scalar.
    # For now, we rely on the MTW in ufi_poll if this is missing.
    # But often Julia tasks are created via wrap_task which does MTW.
    LegateInternal.add_scalar(task.impl, scalar)
end

# Specialized add_scalar to capture type for precompile
function add_scalar(task::Union{AutoTask,ManualTask}, x::T) where {T<:SUPPORTED_TYPES}
    push!(task.arg_types, T)
    LegateInternal.add_scalar(task.impl, Scalar(x))
end
