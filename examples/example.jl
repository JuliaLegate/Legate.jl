using Legate

# Element-wise operation: c[i] = a[i] * scalar + b[i]
function cpu_task_kernel(a, b, c, scalar)
    @inbounds @simd for i in eachindex(a)
        c[i] = a[i] * scalar + b[i]
    end
end

rt = Legate.get_runtime()
lib = Legate.create_library("cpu_lib")

# 1. Register Task
task_id = Legate.wrap_task(cpu_task_kernel)

# 2. Setup Data
shape = [10, 10]

a = Legate.create_array(shape, Float32)
b = Legate.create_array(shape, Float32)
c = Legate.create_array(shape, Float32)

# 3. Create Task
task = Legate.create_julia_task(rt, lib, task_id)

# Add Inputs (a, b), Output (c), Scalar (2.0)
inputs = [Legate.add_input(task, a), Legate.add_input(task, b)]
outputs = [Legate.add_output(task, c)]
Legate.add_scalar(task, Legate.Scalar(2.0f0))

Legate.default_alignment(task, inputs, outputs)
Legate.submit_task(task) # Submit the task to the runtime
