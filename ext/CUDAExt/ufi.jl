const TaskArgumentGPU = Union{CUDA.CuArray,SUPPORTED_TYPES}

function create_julia_task(
    rt::CxxPtr{Runtime}, lib::Library, task_obj::JuliaGPUTask
)
    task = create_task(rt, lib, JULIA_CUSTOM_GPU_TASK)
    add_scalar(task, Scalar(task_obj.task_id))
    register_task_function(task_obj.task_id, task_obj.fun)
    return task
end

function launch_gpu_task(fun, args, threads, blocks)
    # unfortunately, we have to convert to a tuple for @cuda
    @cuda threads=threads blocks=blocks fun(Tuple(args))
    CUDA.synchronize()
end

function _execute_julia_task(::Val{:gpu}, req, task_fun)
    args = Vector{TaskArgumentGPU}()
    sizehint!(args, req.num_inputs + req.num_outputs + req.num_scalars)

    dims = ntuple(i -> req.dims[i], Int(req.ndim))
    N = prod(dims)
    threads = 256
    blocks = cld(N, threads)

    # Process Inputs
    for i in 1:req.num_inputs
        type_code = unsafe_load(req.inputs_types, i) # get type code
        T = get_code_type(type_code) # get type from code
        ptr_val = unsafe_load(req.inputs_ptr, i) # get value storage
        cu_ptr = reinterpret(CUDA.CuPtr{T}, ptr_val) # convert to CuPtr with proper type
        push!(args, unsafe_wrap(CuArray, cu_ptr, dims))
    end

    # Process Outputs
    for i in 1:req.num_outputs
        type_code = unsafe_load(req.outputs_types, i)
        T = get_code_type(type_code)
        ptr_val = unsafe_load(req.outputs_ptr, i)
        cu_ptr = reinterpret(CUDA.CuPtr{T}, ptr_val)
        push!(args, unsafe_wrap(CuArray, cu_ptr, dims))
    end

    # Process Scalars
    for i in 1:req.num_scalars
        type_code = Int(unsafe_load(req.scalar_types, i))
        T = get_code_type(type_code)
        val_ptr = unsafe_load(req.scalars_ptr, i) # get value storage
        scalar_val = unsafe_load(Ptr{T}(val_ptr)) # load value with proper type
        push!(args, scalar_val)
    end

    # Launch Kernel via invokelatest to handle world age issues
    Base.invokelatest(launch_gpu_task, task_fun, args, threads, blocks)

    @debug "Legate UFI: GPU task completed successfully!" req.task_id
end
