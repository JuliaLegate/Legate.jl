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
    align(a::Variable, b::Variable) -> Constraint

Align two variables.

Returns a new constraint representing the alignment of `a` and `b`.
"""
align

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
    add_constraint(AutoTask, c::Constraint)

Add a constraint to the task.
"""
add_constraint

"""
    add_input(AutoTask, LogicalArray) -> Variable
    add_input(ManualTask, LogicalStore) -> Variable

Add a logical array/store as an input to the task.
"""
add_input

"""
    add_output(AutoTask, LogicalArray) -> Variable
    add_output(ManualTask, LogicalStore) -> Variable

Add a logical array/store as an output of the task.
"""
add_output

"""
    add_scalar(AutoTask, scalar::Scalar)
    add_scalar(ManualTask, scalar::Scalar)

Add a scalar argument to the task.
"""
add_scalar
