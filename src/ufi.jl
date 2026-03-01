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

# Realm-defined max dimension
const REALM_MAX_DIM = 6
const MAX_UFI_SLOTS_VAL = 32
const SLOT_REQUEST_PTRS = Vector{Ptr{Cvoid}}(undef, MAX_UFI_SLOTS_VAL)

const UFI_SHUTDOWN = Threads.Atomic{Bool}(false)
const UFI_INITIALIZED = Ref{Bool}(false)
const UFI_SHUTDOWN_DONE = Ref{Bool}(false)
const UFI_INIT_LOCK = ReentrantLock()
const DISPATCH_LOCK = ReentrantLock()

const IN_POLL = Threads.Atomic{Int}(0)

struct TaskJob
    slot_id::Int
    in_p::Ptr{Ptr{Cvoid}}
    out_p::Ptr{Ptr{Cvoid}}
    scal_p::Ptr{Ptr{Cvoid}}
    meta::UfiMetadata
end

const JOB_QUEUE = Ref{Channel{TaskJob}}()
const UFI_POLLER_TASK = Ref{Task}()
const UFI_WORKER_TASKS = Vector{Task}()

"""
    _extract_and_call(meta, in_p, out_p, scal_p, sig)

The type-stable JIT engine. Reconstructs arguments from raw pointers using the 
Signature's type parameters (including dimensions). Perfectly type-stable and 
zero-allocation in the hot path.
"""
@generated function _extract_and_call(meta::UfiMetadata, in_p::Ptr{Ptr{Cvoid}}, out_p::Ptr{Ptr{Cvoid}}, scal_p::Ptr{Ptr{Cvoid}}, ::UfiSignature{InT, OutT, ScT, Dims}) where {InT, OutT, ScT, Dims}
    exprs = []
    cursor = 1

    # Inputs
    for (i, T) in enumerate(InT.parameters)
        d = Dims[cursor]
        push!(exprs, quote
            ptr = unsafe_load(in_p, $i)
            unsafe_wrap(Array, Ptr{eltype($T)}(ptr), $d)
        end)
        cursor += 1
    end

    # Outputs
    for (i, T) in enumerate(OutT.parameters)
        d = Dims[cursor]
        push!(exprs, quote
            ptr = unsafe_load(out_p, $i)
            unsafe_wrap(Array, Ptr{eltype($T)}(ptr), $d)
        end)
        cursor += 1
    end

    # Scalars
    for (i, T) in enumerate(ScT.parameters)
        push!(exprs, quote
            ptr = unsafe_load(scal_p, $i)
            unsafe_load(Ptr{$T}(ptr))
        end)
    end

    return quote
        args = ($(exprs...),)
        # Static argument reconstruction is now complete.
        Base.invokelatest(meta.fun, args...)
    end
end

function ufi_has_pending_work()
    # Robust synchronization using submitted/started counts
    submitted = SUBMITTED_COUNT[]
    started = Int(ccall((:legate_get_started_count, Legate.WRAPPER_LIB_PATH), Cint, ()))
    
    active_calls = Int(ccall((:legate_get_active_call_count, Legate.WRAPPER_LIB_PATH), Cint, ()))
    active_slots = Int(ccall((:legate_get_active_slot_count, Legate.WRAPPER_LIB_PATH), Cint, ()))
    
    # Optional stderr logging to track synchronization state
    # if submitted > started
    #     println(stderr, "[UFI Sync] Pending Start: sub=$submitted, start=$started, active=$active_calls, slots=$active_slots")
    # end

    return submitted > started || active_calls > 0 || active_slots > 0
end

function wait_ufi()
    # Legate.issue_execution_fence(blocking=false)
    while ufi_has_pending_work()
        if Threads.threadid() == 1
            ufi_poll()
        end
        yield()
        sleep(0.001)
    end
end

function ufi_poll()
    if !UFI_INITIALIZED[]; return false; end
    if Threads.atomic_cas!(IN_POLL, 0, 1) != 0; return false; end
    
    try
        slot_id = Int(ccall((:legate_pop_pending_slot_nonblocking, Legate.WRAPPER_LIB_PATH), Cint, ()))
        if slot_id != -1
            base_ptr = SLOT_REQUEST_PTRS[slot_id + 1]
            task_id = unsafe_load(Ptr{UInt32}(base_ptr + 4))
            
            meta = lock(REGISTRY_LOCK) do
                get(GLOBAL_TASK_REGISTRY, task_id, nothing)
            end
            
            if isnothing(meta)
                # This should NOT happen now with correct metadata alignment
                println(stderr, "[UFI Error] Task ID $task_id not found in registry!")
                ccall((:completion_callback_from_julia, Legate.WRAPPER_LIB_PATH), Cvoid, (Cint,), Cint(slot_id))
                return true
            end

            # void** extraction with fixed indirection
            in_p    = unsafe_load(Ptr{Ptr{Ptr{Cvoid}}}(base_ptr + 8))
            out_p   = unsafe_load(Ptr{Ptr{Ptr{Cvoid}}}(base_ptr + 16))
            scal_p  = unsafe_load(Ptr{Ptr{Ptr{Cvoid}}}(base_ptr + 24))

            put!(JOB_QUEUE[], TaskJob(slot_id, in_p, out_p, scal_p, meta))
            return true
        end
    finally
        IN_POLL[] = 0
    end
    return false
end

function _ufi_worker_loop()
    (ccall(:jl_generating_output, Cint, ()) != 0) && return
    while !UFI_SHUTDOWN[]
        try
            job = take!(JOB_QUEUE[])
            try
                _extract_and_call(job.meta, job.in_p, job.out_p, job.scal_p, job.meta.sig)
            catch e
                println(stderr, "[UFI Worker Error] Slot $(job.slot_id): $e")
                Base.display_error(stderr, e, catch_backtrace())
            finally
                ccall((:completion_callback_from_julia, Legate.WRAPPER_LIB_PATH), Cvoid, (Cint,), Cint(job.slot_id))
            end
        catch e; end
    end
end

function _ufi_poller_loop()
    (ccall(:jl_generating_output, Cint, ()) != 0) && return
    while !UFI_SHUTDOWN[]
        if !ufi_poll()
            yield()
            # sleep(0.001)
        end
    end
    UFI_SHUTDOWN_DONE[] = true
end


function init_ufi()
    lock(UFI_INIT_LOCK) do
        UFI_INITIALIZED[] && return
        _is_precompiling() && return
        
        JOB_QUEUE[] = Channel{TaskJob}(1000)
        
        max_slots = ccall((:legate_get_max_slots, Legate.WRAPPER_LIB_PATH), Cint, ())
        for i in 1:max_slots
            SLOT_REQUEST_PTRS[i] = ccall((:legate_get_slot_request_ptr, Legate.WRAPPER_LIB_PATH), Ptr{Cvoid}, (Cint,), Cint(i-1))
        end

        LegateInternal._initialize_async_system()
        UFI_INITIALIZED[] = true
        
        precompile(_ufi_poller_loop, ())
        precompile(_ufi_worker_loop, ())

        UFI_POLLER_TASK[] = errormonitor(@async _ufi_poller_loop())

        empty!(UFI_WORKER_TASKS)
        for i in 1:Threads.nthreads()-1
            push!(UFI_WORKER_TASKS, errormonitor(Threads.@spawn _ufi_worker_loop()))
        end
        yield()

        println(stderr, "[UFI] System Initialized (Concurrent Count-Sync Mode)")
    end
end

function shutdown_ufi()
    UFI_SHUTDOWN[] = true
    UFI_INITIALIZED[] = false
    if isassigned(JOB_QUEUE) && isopen(JOB_QUEUE[])
        close(JOB_QUEUE[])
    end
end