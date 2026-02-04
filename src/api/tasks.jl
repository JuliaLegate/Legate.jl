"""
    create_task(rt::Runtime, lib::Library, id::LocalTaskID) -> AutoTask

Create an auto task in the runtime.

# Arguments
- `rt`: The current runtime instance.
- `lib`: The library to associate with the task.
- `id`: The local task identifier.
"""
create_task(rt::CxxPtr{Runtime}, lib::Library, id::LocalTaskID) = create_auto_task(rt, lib, id)
function create_task(rt::CxxPtr{Runtime}, lib::Library, id::LocalTaskID, domain::Domain)
    create_manual_task(rt, lib, id, domain)
end

"""
    submit_task(rt::Runtime, AutoTask)
    submit_task(rt::Runtime, ManualTask)

Submit an manual/auto task to the runtime.
"""
submit_task(rt::CxxPtr{Runtime}, task::AutoTask) = submit_auto_task(rt, task)
submit_task(rt::CxxPtr{Runtime}, task::ManualTask) = submit_manual_task(rt, task)

"""
    align(a::Variable, b::Variable) -> Constraint

Align two variables.

Returns a new constraint representing the alignment of `a` and `b`.
"""
align

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
add_constraint

"""
    add_input(AutoTask, LogicalArray) -> Variable
    add_input(ManualTask, LogicalStore) -> Variable

Add a logical array/store as an input to the task.
"""
function add_input(
    task::Union{AutoTask,ManualTask},
    item::Union{LogicalArray,LogicalStore},
)
    add_input(task, item.handle)
end

"""
    add_output(AutoTask, LogicalArray) -> Variable
    add_output(ManualTask, LogicalStore) -> Variable

Add a logical array/store as an output of the task.
"""
function add_output(
    task::Union{AutoTask,ManualTask},
    item::Union{LogicalArray,LogicalStore},
)
    add_output(task, item.handle)
end

"""
    add_scalar(AutoTask, scalar::Scalar)
    add_scalar(ManualTask, scalar::Scalar)

Add a scalar argument to the task.
"""
add_scalar
