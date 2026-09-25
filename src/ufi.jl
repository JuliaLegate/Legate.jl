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

const REALM_MAX_DIM = 6
const MAX_UFI_SLOTS_VAL = 32
const PhysArrPtr = Ptr{Cvoid}
const SLOT_REQUEST_PTRS = Vector{Ptr{Cvoid}}(undef, MAX_UFI_SLOTS_VAL)
const SLOT_INPUT_DIMS_PTRS = fill(Ptr{Int64}(C_NULL), MAX_UFI_SLOTS_VAL)
const SLOT_OUTPUT_DIMS_PTRS = fill(Ptr{Int64}(C_NULL), MAX_UFI_SLOTS_VAL)

const UFI_INIT_LOCK = ReentrantLock()

const PENDING_JOBS = Threads.Atomic{Int}(0)

const UFI_ERROR = 217

struct TaskRequest
    is_gpu::Int32
    task_id::UInt32
    inputs_ptr::Ptr{PhysArrPtr}
    outputs_ptr::Ptr{PhysArrPtr}
    scalars_ptr::Ptr{Ptr{Cvoid}}
    input_strides_ptr::Ptr{Int64}
    output_strides_ptr::Ptr{Int64}
    ndim::Int32
    dims::NTuple{REALM_MAX_DIM,Int64}
end

@inline function _stride_offset(strides::NTuple{N,Int}, I::CartesianIndex{N}) where {N}
    off = 0
    @inbounds for d in 1:N
        off += (I[d] - 1) * strides[d]
    end
    return off
end

# Materialize a strided Legate tile as a dense Julia array.
function _strided_to_dense(::Type{T}, ptr::Ptr{T}, dims::NTuple{N,Int},
    strides::NTuple{N,Int}) where {T,N}
    out = Array{T,N}(undef, dims)
    @inbounds for I in CartesianIndices(dims)
        out[I] = unsafe_load(ptr, _stride_offset(strides, I) + 1)
    end
    return out
end

# Commit a dense result to a strided Legate tile.
function _dense_to_strided(ptr::Ptr{T}, src::Array{T,N},
    strides::NTuple{N,Int}) where {T,N}
    @inbounds for I in CartesianIndices(size(src))
        unsafe_store!(ptr, src[I], _stride_offset(strides, I) + 1)
    end
    return nothing
end

# Holds copied request metadata; B selects CPU or GPU hooks.
struct TaskJob{B<:TaskBackend,M<:UfiMetadata,D<:Tuple}
    slot_id::Int
    backend::B
    in_args::Vector{PhysArrPtr}
    out_args::Vector{PhysArrPtr}
    scal_args::Vector{Ptr{Cvoid}}
    in_strides::Vector{Int64}   # flat [arg][REALM_MAX_DIM] element strides
    out_strides::Vector{Int64}
    in_dims::Vector{Int64}      # flat [arg][REALM_MAX_DIM] tile dimensions
    out_dims::Vector{Int64}
    local_dims::D
    meta::M
end

function _unsupported_backend(backend::TaskBackend)
    return error("$(nameof(typeof(backend))) tasking requires its package extension to be loaded.")
end
function _unsupported_backend(::GPUBackend)
    return error("GPU tasking requires CUDA.jl to be loaded (`using CUDA`).")
end

# CUDAExt adds GPU methods for these shared dispatcher hooks.
_ufi_prepare_input(backend::TaskBackend, args...) = _unsupported_backend(backend)
_ufi_prepare_output(backend::TaskBackend, args...) = _unsupported_backend(backend)
_ufi_commit_output(backend::TaskBackend, args...) = _unsupported_backend(backend)
_ufi_invoke(backend::TaskBackend, args...) = _unsupported_backend(backend)
_ufi_synchronize(backend::TaskBackend) = _unsupported_backend(backend)

function _ufi_prepare_input(
    ::CPUBackend, ::Type{T}, ptr::PhysArrPtr, dims::NTuple{N,Int}, strides::NTuple{N,Int}
) where {T,N}
    return _strided_to_dense(T, Ptr{T}(ptr), dims, strides)
end

function _ufi_prepare_output(
    ::CPUBackend, ::Type{T}, ptr::PhysArrPtr, dims::NTuple{N,Int}, ::NTuple{N,Int}
) where {T,N}
    return Array{T,N}(undef, dims), Ptr{T}(ptr)
end

function _ufi_commit_output(
    ::CPUBackend,
    destination::Ptr{T},
    output::Array{T,N},
    ::NTuple{N,Int},
    strides::NTuple{N,Int},
) where {T,N}
    return _dense_to_strided(destination, output, strides)
end

_ufi_invoke(::CPUBackend, f, ::Tuple, args...) = f(args...)
_ufi_synchronize(::CPUBackend) = nothing

mutable struct UfiManager
    job_queue::Channel{TaskJob}
    poller_task::Task
    worker_tasks::Vector{Task}
    shutdown::Threads.Atomic{Bool}
    shutdown_done::Threads.Atomic{Bool}

    function UfiManager(num_workers::Int)
        mgr = new(
            Channel{TaskJob}(128),
            Task(() -> nothing), # poller_task placeholder
            Vector{Task}(), # worker_tasks
            Threads.Atomic{Bool}(false),
            Threads.Atomic{Bool}(false),
        )

        # Default-pool tasks keep running while the interactive launch thread blocks.
        mgr.poller_task = errormonitor(Threads.@spawn _ufi_poller_loop(mgr))

        for _ in 1:num_workers
            push!(mgr.worker_tasks, errormonitor(Threads.@spawn _ufi_worker_loop(mgr)))
        end

        return mgr
    end
end

const UFI_MANAGER = Ref{Union{Nothing,UfiManager}}(nothing)

function ufi_initialized()
    return !isnothing(UFI_MANAGER[])
end

# Signature types unroll argument decoding; backend hooks own storage and invocation.
@generated function _do_call(
    backend::B,
    f,
    in_p_ptr::Ptr{PhysArrPtr},
    out_p_ptr::Ptr{PhysArrPtr},
    scal_p_ptr::Ptr{Ptr{Cvoid}},
    in_str_ptr::Ptr{Int64},
    out_str_ptr::Ptr{Int64},
    in_dim_ptr::Ptr{Int64},
    out_dim_ptr::Ptr{Int64},
    local_dims::Tuple,
    ::UfiSignature{InT,OutT,ScT},
) where {B<:TaskBackend,InT,OutT,ScT}
    pre = []       # prepare input and output buffers
    callargs = []  # args passed to the user task
    post = []      # commit outputs to their destination tiles

    metadata(ptr, base, nd) =
        Expr(:tuple, [:(Int(unsafe_load($ptr, $(base + d)))) for d in 1:nd]...)

    for (i, T) in enumerate(InT.parameters)
        E = eltype(T)
        nd = ndims(T)
        base = (i - 1) * REALM_MAX_DIM
        sym = gensym(:in)
        dsym = gensym(:indims)
        ssym = gensym(:instrides)
        push!(pre, :($dsym = $(metadata(:in_dim_ptr, base, nd))))
        push!(pre, :($ssym = $(metadata(:in_str_ptr, base, nd))))
        push!(
            pre,
            :(
                $sym = _ufi_prepare_input(
                    backend, $E, unsafe_load(in_p_ptr, $i), $dsym, $ssym)
            ),
        )
        push!(callargs, sym)
    end

    for (i, T) in enumerate(OutT.parameters)
        E = eltype(T)
        nd = ndims(T)
        base = (i - 1) * REALM_MAX_DIM
        sym = gensym(:out)
        state = gensym(:state)
        dsym = gensym(:outdims)
        ssym = gensym(:outs)
        push!(pre, :($dsym = $(metadata(:out_dim_ptr, base, nd))))
        push!(pre, :($ssym = $(metadata(:out_str_ptr, base, nd))))
        push!(
            pre,
            :(
                ($sym, $state) = _ufi_prepare_output(
                    backend, $E, unsafe_load(out_p_ptr, $i), $dsym, $ssym)
            ),
        )
        push!(callargs, sym)
        push!(post, :(_ufi_commit_output(backend, $state, $sym, $dsym, $ssym)))
    end

    for (i, T) in enumerate(ScT.parameters)
        push!(callargs, :(unsafe_load(Ptr{$T}(unsafe_load(scal_p_ptr, $i)))))
    end

    return quote
        $(pre...)
        _ufi_invoke(backend, f, local_dims, $(callargs...))
        $(post...)
        _ufi_synchronize(backend)
        nothing
    end
end

function _execute_task(job::TaskJob)
    in_args = job.in_args
    out_args = job.out_args
    scalar_args = job.scal_args
    in_strides = job.in_strides
    out_strides = job.out_strides
    in_dims = job.in_dims
    out_dims = job.out_dims
    GC.@preserve in_args out_args scalar_args in_strides out_strides in_dims out_dims begin
        _do_call(
            job.backend,
            job.meta.fun,
            pointer(in_args),
            pointer(out_args),
            pointer(scalar_args),
            pointer(in_strides),
            pointer(out_strides),
            pointer(in_dims),
            pointer(out_dims),
            job.local_dims,
            job.meta.sig,
        )
    end
    return nothing
end

function ufi_has_pending_work(drain_slots::Bool=true)
    active_calls = Int(ccall((:legate_get_active_call_count, Legate.WRAPPER_LIB_PATH), Cint, ()))
    active_slots = if drain_slots
        Int(ccall((:legate_get_active_slot_count, Legate.WRAPPER_LIB_PATH), Cint, ()))
    else
        0
    end
    return active_calls > 0 || active_slots > 0 || PENDING_JOBS[] > 0
end

function wait_ufi(drain_slots::Bool=true)
    while ufi_has_pending_work(drain_slots)
        yield()
        sleep(0.001)
    end
end

_task_backend(is_gpu::Int32) = is_gpu == 0 ? CPUBackend() : GPUBackend()

function _copy_pointer_args(ptr::Ptr{T}, count::Int) where {T}
    args = Vector{T}(undef, count)
    @inbounds for i in 1:count
        args[i] = unsafe_load(ptr, i)
    end
    return args
end

function _copy_strides(ptr::Ptr{Int64}, count::Int)
    strides = Vector{Int64}(undef, count * REALM_MAX_DIM)
    count > 0 && unsafe_copyto!(pointer(strides), ptr, length(strides))
    return strides
end

function _copy_dims(ptr::Ptr{Int64}, count::Int, common_dims::Tuple)
    dims = zeros(Int64, count * REALM_MAX_DIM)
    if ptr == C_NULL
        for arg in 0:(count - 1), dim in eachindex(common_dims)
            dims[arg * REALM_MAX_DIM + dim] = common_dims[dim]
        end
    elseif count > 0
        unsafe_copyto!(pointer(dims), ptr, length(dims))
    end
    return dims
end

# C++ keeps the referenced tile storage valid until the completion callback.
function _make_job(slot_id::Int, req::TaskRequest, meta::UfiMetadata)
    sig_type = typeof(meta.sig)
    in_count = length(sig_type.parameters[1].parameters)
    out_count = length(sig_type.parameters[2].parameters)
    scalar_count = length(sig_type.parameters[3].parameters)
    local_dims = ntuple(i -> Int(max(0, req.dims[i])), Int(req.ndim))
    in_dims_ptr = SLOT_INPUT_DIMS_PTRS[slot_id + 1]
    out_dims_ptr = SLOT_OUTPUT_DIMS_PTRS[slot_id + 1]

    return TaskJob(
        slot_id,
        _task_backend(req.is_gpu),
        _copy_pointer_args(req.inputs_ptr, in_count),
        _copy_pointer_args(req.outputs_ptr, out_count),
        _copy_pointer_args(req.scalars_ptr, scalar_count),
        _copy_strides(req.input_strides_ptr, in_count),
        _copy_strides(req.output_strides_ptr, out_count),
        _copy_dims(in_dims_ptr, in_count, local_dims),
        _copy_dims(out_dims_ptr, out_count, local_dims),
        local_dims,
        meta,
    )
end

function ufi_poll(mgr::UfiManager)
    if mgr.shutdown[]
        return false
    end

    slot_id = Int(ccall((:legate_pop_pending_slot_nonblocking, Legate.WRAPPER_LIB_PATH), Cint, ()))
    slot_id == -1 && return false

    base_ptr = SLOT_REQUEST_PTRS[slot_id + 1]
    req = unsafe_load(Ptr{TaskRequest}(base_ptr))
    task_id = req.task_id

    lock(REGISTRY_LOCK)
    meta = try
        get(GLOBAL_TASK_REGISTRY, task_id, nothing)
    finally
        unlock(REGISTRY_LOCK)
    end

    if isnothing(meta)
        if !mgr.shutdown[]
            println(stderr, "[UFI Error] Task ID $task_id not found in registry!")
            exit(UFI_ERROR)
        end
        return false
    end

    Threads.atomic_add!(PENDING_JOBS, 1)
    try
        put!(mgr.job_queue, _make_job(slot_id, req, meta))
    catch e
        Threads.atomic_sub!(PENDING_JOBS, 1) # Decrement if put! fails
        if e isa InvalidStateException && e.state == :closed
            return false
        end
        rethrow(e)
    end
    return true
end

function _ufi_poller_loop(mgr::UfiManager)
    _is_precompiling() && return nothing
    while !mgr.shutdown[]
        if !ufi_poll(mgr)
            yield()
        end
    end
    return mgr.shutdown_done[] = true
end

function _ufi_worker_loop(mgr::UfiManager)
    _is_precompiling() && return nothing
    while !mgr.shutdown[]
        job = try
            take!(mgr.job_queue)
        catch ex
            if ex isa InvalidStateException && ex.state == :closed
                return nothing
            end
            @error "Error in UFI worker loop" exception=(ex, catch_backtrace())
            break
        end

        try
            # Workers must see methods compiled on the submitting thread.
            Base.invokelatest(_execute_task, job)
        catch e
            println(stderr, "[UFI Worker Error] Slot $(job.slot_id): $e")
            Base.display_error(stderr, e, catch_backtrace())
            exit(UFI_ERROR)
        finally
            Threads.atomic_sub!(PENDING_JOBS, 1)
            ccall(
                (:completion_callback_from_julia, Legate.WRAPPER_LIB_PATH),
                Cvoid,
                (Cint,),
                Cint(job.slot_id),
            )
        end
    end
end

function init_ufi()
    lock(UFI_INIT_LOCK) do
        ufi_initialized() && return nothing
        _is_precompiling() && return nothing

        max_slots = ccall((:legate_get_max_slots, Legate.WRAPPER_LIB_PATH), Cint, ())
        if max_slots <= 0
            exit(UFI_ERROR)
        end

        has_arg_dims =
            isdefined(LegateInternal, :legate_get_slot_input_dims_ptr) &&
            isdefined(LegateInternal, :legate_get_slot_output_dims_ptr)
        for i in 1:max_slots
            SLOT_REQUEST_PTRS[i] = ccall(
                (:legate_get_slot_request_ptr, Legate.WRAPPER_LIB_PATH),
                Ptr{Cvoid},
                (Cint,),
                Cint(i-1),
            )
            if has_arg_dims
                SLOT_INPUT_DIMS_PTRS[i] =
                    LegateInternal.legate_get_slot_input_dims_ptr(Cint(i - 1)).cpp_object
                SLOT_OUTPUT_DIMS_PTRS[i] =
                    LegateInternal.legate_get_slot_output_dims_ptr(Cint(i - 1)).cpp_object
            end
        end

        LegateInternal._initialize_async_system()
        if isdefined(LegateInternal, :JULIA_CUSTOM_GPU_TASK)
            JULIA_CUSTOM_GPU_TASK[] = LegateInternal.JULIA_CUSTOM_GPU_TASK
        end

        precompile(_ufi_poller_loop, (UfiManager,))
        precompile(_ufi_worker_loop, (UfiManager,))

        # Reserve one default thread for the caller.
        num_workers = max(1, Threads.nthreads(:default) - 1)
        UFI_MANAGER[] = UfiManager(num_workers)

        yield()

        return println(
            stderr,
            "[UFI] System Initialized (Concurrent Count-Sync Mode) with $(num_workers) workers\n",
        )
    end
end

# Filled from the C++ wrapper during UFI initialization.
const JULIA_CUSTOM_GPU_TASK = Ref{Any}(nothing)
export JULIA_CUSTOM_GPU_TASK

function shutdown_ufi(mgr::UfiManager=UFI_MANAGER[])
    isnothing(mgr) && return nothing
    # The caller must drain work with wait_ufi() first.
    mgr.shutdown[] = true

    if isopen(mgr.job_queue)
        close(mgr.job_queue)
    end

    # Closing the queue lets workers exit before runtime teardown.
    for t in (mgr.poller_task, mgr.worker_tasks...)
        try
            wait(t)
        catch e
            @error "UFI task errored during shutdown" exception = (e, catch_backtrace())
        end
    end

    if mgr === UFI_MANAGER[]
        UFI_MANAGER[] = nothing
    end
    return nothing
end

function ufi_has_shutdown_done()
    mgr = UFI_MANAGER[]
    isnothing(mgr) && return true
    return mgr.shutdown_done[]
end
