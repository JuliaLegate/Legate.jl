# Legate Usage Examples

## CPU Tasking

```julia
using Legate

# Element-wise operation: c[i] = a[i] * scalar + b[i]
function cpu_task_kernel(args)
    a, b, c, scalar = args
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

# You will need to initialize (a, b) since they are inputs
# Legate will throw an error if they are not initialized
# For more information about initializing arrays, look at /examples/tasking.jl
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
Legate.submit_task(rt, task) # Submit the task to the runtime
```


## GPU Tasking

```julia
using Legate
using CUDA

# CUDA Kernel: c[i] = a[i] + b[i]
function gpu_add_kernel(args)
    a, b, c = args
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= length(a)
        @inbounds c[idx] = a[idx] + b[idx]
    end
    return nothing
end

rt = Legate.get_runtime()
lib = Legate.create_library("gpu_lib")

# 1. Register GPU Task
task_id = Legate.wrap_task(gpu_add_kernel; task_type=:gpu)

# 2. Setup Data.
N = [128, 128] 

# Similar to CPU tasking, you will need to initialize (a, b) since they are inputs
# For more information about initializing arrays, look at /examples/tasking.jl
a = Legate.create_array(N, Float32)
b = Legate.create_array(N, Float32)
c = Legate.create_array(N, Float32)

# 3. Create Task
task = Legate.create_julia_task(rt, lib, task_id)
inputs = [Legate.add_input(task, a), Legate.add_input(task, b)]
outputs = [Legate.add_output(task, c)]

Legate.default_alignment(task, inputs, outputs)
Legate.submit_task(rt, task) # Submit the task to the runtime
```