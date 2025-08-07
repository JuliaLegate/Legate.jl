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

const PREFS_CHANGED = Ref(false)
const DEPS_LOADED = Ref(false)

# default
const MODE_JLL = "jll"
# will compile wrappers from src
const MODE_DEVELOPER = "developer"
# not well tested, allows conda env install
const MODE_CONDA = "conda"

# Store what the values were when module loaded
const mode = @load_preference("mode")
# used for developer mode
const wrapper_branch = @load_preference("wrapper_branch")
const use_legate_jll = @load_preference("use_legate_jll")
const legate_path = @load_preference("legate_path")

# used for conda mode
const conda_env = @load_preference("conda_env")

# default developer options
const DEVEL_DEFAULT_JLL_WRAP_CONFIG = false
const DEVEL_DEFAULT_WRAPPER_BRANCH = "main"
const DEVEL_DEFAULT_JLL_CONFIG = true
const DEVEL_DEFAULT_LEGATE_PATH = nothing

# from MPIPreferences.jl
"""
    LegatePreferences.check_unchanged()

Throws an error if the preferences have been modified in the current Julia
session, or if they are modified after this function is called.

This is should be called from the `__init__()` function of any package which
relies on the values of LegatePreferences.
"""
function check_unchanged()
    if PREFS_CHANGED[]
        error("LegatePreferences have changed, you will need to restart Julia for the changes to take effect")
    end
    DEPS_LOADED[] = true
    return nothing
end


"""
    LegatePreferences.use_conda(conda_env::String; export_prefs = false, force = true)

Tells Legate.jl to use existing conda install. We make no gurantees of compiler compatability at this time.

Expects `conda_env` to be the absolute path to the root of the environment.
For example, `/home/julialegate/.conda/envs/cunumeric-gpu`
"""
function use_conda(conda_env::String; export_prefs = false, force = true)
    set_preferences!(LegatePreferences,
        "conda_env" =>  conda_env,
        "mode" => MODE_CONDA,
        export_prefs = export_prefs,
        force = force
    )

    if conda_env == LegatePreferences.conda_env && LegatePreferences.mode == MODE_CONDA
        @info "LegatePreferences found no differences."
    else
        PREFS_CHANGED[] = true
        @info "LegatePreferences set to use local conda env at:" conda_env

        if DEPS_LOADED[]
            error("You will need to restart Julia for the changes to take effect. You may need Pkg.build() for Legion patch.")
        end
    end
end

"""
    LegatePreferences.use_jll_binary(; export_prefs = false, force = true)

Tells Legate.jl to use JLLs. This is the default option. 
"""
function use_jll_binary(; export_prefs = false, force = true)
    set_preferences!(LegatePreferences,
        "mode" => MODE_JLL,
        export_prefs = export_prefs,
        force = force
    )

    if LegatePreferences.mode == MODE_JLL
        @info "LegatePreferences found no differences. Using JLLs."
    else
        PREFS_CHANGED[] = true
        @info "LegatePreferences set to use JLLs."

        if DEPS_LOADED[]
            error("You will need to restart Julia for the changes to take effect. JLLs do not require building.")
        end
    end
end

"""
    LegatePreferences.use_developer_mode(; wrapper_branch="main", use_legate_jll=true, legate_path=nothing, export_prefs = false, force = true)

Tells Legate.jl to enable developer mode. This will clone legate_jl_wrapper into Legate.jl/deps. 

To specify a legate_jl_wrapper branch: ```wrapper_branch="some-branch"```
To disable using legate_jll: ```use_legate_jll=false``` 
If you disable legate_jll, then you need to set a path to Legate with ```legate_path="/path/to/legate"```
"""
function use_developer_mode(; wrapper_branch=DEVEL_DEFAULT_WRAPPER_BRANCH, use_legate_jll=DEVEL_DEFAULT_JLL_CONFIG, legate_path=DEVEL_DEFAULT_LEGATE_PATH, export_prefs = false, force = true)
    if (use_legate_jll == false)
        if (legate_path == nothing) 
            error("You must set a legate_path if you are disabling use_legate_jll")
        end
    end

    set_preferences!(LegatePreferences,
        "mode" => MODE_DEVELOPER,
        "wrapper_branch" => wrapper_branch,
        "use_legate_jll" => use_legate_jll,
        "legate_path" => legate_path,
        export_prefs = export_prefs,
        force = force
    )
    same_branch = wrapper_branch == LegatePreferences.wrapper_branch
    same_jll_conf = use_legate_jll == LegatePreferences.use_legate_jll
    same_legate_path = legate_path == LegatePreferences.legate_path
    same_mode = LegatePreferences.mode == MODE_DEVELOPER
    
    if same_branch && same_jll_conf && same_legate_path && same_mode
        @info "LegatePreferences found no differences. Using Developer mode."
    else
        PREFS_CHANGED[] = true
        @info "LegatePreferences set to use developer mode.."

        if DEPS_LOADED[]
            error("You will need to restart Julia for the changes to take effect. You need to call Pkg.build()")
        end
    end
end

end # module LegatePreferences
