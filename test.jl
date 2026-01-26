using FunctionWrappers
import FunctionWrappers: FunctionWrapper
using Legate

const TaskFunType = FunctionWrapper{Nothing,Tuple{Vector{Float32},Vector{Float32},Vector{Float32}}}

struct JuliaTask
    fun::TaskFunType
end

const JULIA_TASKS = Dict{UInt32,FunctionWrapper}()

function register_task(task_id::UInt32, fw::FunctionWrapper)
    JULIA_TASKS[task_id] = fw
    println("Registered Julia task $task_id in Julia registry")
end

# Example of wrapping your original task_test
function task_test(a::Vector{Float32}, b::Vector{Float32}, c::Vector{Float32})
    @inbounds @simd for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
end

function task_init(a::Vector{Float32}, b::Vector{Float32}, c::Vector{Float32})
    @inbounds @simd for i in eachindex(a)
        a[i] = rand(Float32)
        b[i] = rand(Float32)
        c[i] = 0.0f0
    end
end

# Store the wrapped function
const my_task = JuliaTask(
    FunctionWrapper{Nothing,Tuple{Vector{Float32},Vector{Float32},Vector{Float32}}}(task_test)
)
const my_init_task = JuliaTask(
    FunctionWrapper{Nothing,Tuple{Vector{Float32},Vector{Float32},Vector{Float32}}}(task_init)
)

function register_task()
    # Use Ref to get a pointer to Julia object
    ptr = Base.unsafe_convert(Ptr{Nothing}, Ref(my_init_task))
    Legate.register_julia_task(50001, ptr)  # for init task

    ptr = Base.unsafe_convert(Ptr{Nothing}, Ref(my_task))
    Legate.register_julia_task(50002, ptr)

    println("Registered Julia tasks with C++ dispatcher")
end

function test_driver()
    rt = Legate.get_runtime()
    lib = Legate.create_library("test")
    Legate.ufi_interface_register(lib)
    println("Registered library with C++ runtime")

    task = Legate.create_task(rt, lib, Legate.JULIA_CUSTOM_TASK)

    a = Legate.create_array([100], Float32)
    b = Legate.create_array([100], Float32)
    c = Legate.create_array([100], Float32)

    init_output_vars = Vector{Legate.Variable}()

    push!(init_output_vars, Legate.add_output(task, a))
    push!(init_output_vars, Legate.add_output(task, b))
    push!(init_output_vars, Legate.add_output(task, c))
    Legate.add_scalar(task, Legate.Scalar(UInt32(50001))) # init task id
    Legate.default_alignment(task, Vector{Legate.Variable}(), init_output_vars)
    Legate.submit_task(rt, task)

    input_vars = Vector{Legate.Variable}()
    output_vars = Vector{Legate.Variable}()

    push!(input_vars, Legate.add_input(task, a))
    push!(input_vars, Legate.add_input(task, b))
    push!(output_vars, Legate.add_output(task, c))

    Legate.add_scalar(task, Legate.Scalar(UInt32(50002))) # main task id
    Legate.default_alignment(task, input_vars, output_vars)
    Legate.submit_task(rt, task)

    println("Task executed successfully")
end

if abspath(PROGRAM_FILE) == @__FILE__
    register_task()
    test_driver()
end
