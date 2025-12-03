#= Copyright 2025 Northwestern University, 
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

module LegatePreferences
using Preferences
using Libdl

include("PreferenceBackend.jl")
using .PrefBackend

@make_preferences("legate_")

const _libcuda_names = (
    "libcuda.so.1",  # Linux 
    "libcuda.so",    # Linux
    "nvcuda.dll",    # Windows
    "libcuda.dylib", # macOS
)

const CUresult     = Cint
const CUDA_SUCCESS = CUresult(0)

# Assumes the driver is visible, if its only installed
# via a JLL this will probably return false
function has_cuda_gpu()::Bool
    for name in _libcuda_names
        # dlopen_e: no throw on failure, returns C_NULL
        handle = Libdl.dlopen_e(name)
        handle == C_NULL && continue

        try
            # dlsym_e: no throw on failure, returns C_NULL
            cuInit_ptr = Libdl.dlsym_e(handle, :cuInit)
            cuInit_ptr == C_NULL && continue

            cuDeviceGetCount_ptr = Libdl.dlsym_e(handle, :cuDeviceGetCount)
            cuDeviceGetCount_ptr == C_NULL && continue

            # Call cuInit
            res = ccall(cuInit_ptr, CUresult, (Cuint,), 0)
            res == CUDA_SUCCESS || continue

            # Call cuDeviceGetCount
            n = Ref{Cint}()
            res = ccall(cuDeviceGetCount_ptr, CUresult, (Ref{Cint},), n)
            return res == CUDA_SUCCESS && n[] > 0
        finally
            Libdl.dlclose(handle)
        end
    end

    return false
end

function __init__()
    if MODE == MODE_JLL
        if has_cuda_gpu()
            @info "Detected CUDA GPU"
        else
            @warn "Detected no CUDA GPU will download CPU only JLL."
        end
    end

    #! FOR CONDA/LOCAL INSTALLS WE NEED TO  
    #! SET SOME THINGS IN LOCALPREFERENCES.TOML TO 
    #! TELL CUDA.JL TO USE THE RIGHT CUDA
end

end # module LegatePreferences
