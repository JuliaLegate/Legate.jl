module PrefBackend

export @make_preferences

"""
    @make_preferences(; prefix, default_mode="jll",
                        default_use_jll=true,
                        default_path=nothing)
Creates a namespaced set of preferences for Legate.jl or cuNumeric.jl.  
The generated preferences and functions will be prefixed with the provided
`prefix` string.    
# Arguments
- `prefix::String`: The prefix to use for all preference keys and generated functions.
- `default_mode::String`: The default mode to use. One of "jll", "developer", or "conda". Default is "jll".
- `default_use_jll::Bool`: The default value for whether to use JLLs in developer mode. Default is `true`.
- `default_path::Union{Nothing,String}`: The default path to Legate or cuNumeric in developer mode. Default is `nothing`.
""" 
macro make_preferences(prefix, default_mode="jll",
                        default_use_jll=true,
                        default_path=nothing)

esc(quote
    abstract type Mode end
    struct JLL <: Mode end # default
    struct Developer <: Mode end # will compile wrappers from src
    struct Conda <: Mode end # not well tested, allows conda env install

    function to_mode(m::String)
        m_lower = lowercase(m)
        if m_lower == "jll"
            return JLL()
        elseif m_lower == "developer"
            return Developer()
        elseif m_lower == "conda"
            return Conda()
        else
            error("Unknown mode: $(m). Must be one of 'jll', 'developer', or 'conda'.")
        end
    end

    const MODE_JLL = "jll"
    const MODE_DEVELOPER = "developer"
    const MODE_CONDA = "conda"

    const _PREFS_CHANGED = Ref(false)
    const _DEPS_LOADED = Ref(false)

    # load preferences with namespaced keys
    const MODE = @load_preference($prefix * "mode", $default_mode)

    const _use_jll = @load_preference($prefix * "use_jll",
                                            $default_use_jll)
    const _path = @load_preference($prefix * "path", $default_path)
    const _conda_env = @load_preference($prefix * "conda_env")

    function _set(pairs::Pair...; export_prefs=false, force=true)
        prefixed_pairs = Pair[]
        for (k, v) in pairs
            push!(prefixed_pairs, $prefix * k => v)
        end

        set_preferences!(@__MODULE__, prefixed_pairs...; export_prefs, force)
    end

    """
        LegatePreferences.check_unchanged()

    Throws an error if the preferences have been modified in the current Julia
    session, or if they are modified after this function is called.

    This is should be called from the `__init__()` function of any package which
    relies on the values of LegatePreferences.
    """
    function check_unchanged()
        if _PREFS_CHANGED[]
            error("Preferences changed; restart Julia")
        end
        _DEPS_LOADED[] = true
        nothing
    end


    """
        LegatePreferences.use_jll_binary(; export_prefs = false, force = true)

    Tells Legate.jl | cuNumeric.jl to use JLLs. This is the default option. 
    """
    function use_jll_binary(; export_prefs=false, force=true)
        if MODE == MODE_JLL
            @info "Already using JLL mode"
        else
            _PREFS_CHANGED[] = true
            @info "Switched to JLL mode"
            _set("mode" => MODE_JLL; export_prefs, force)
            if _DEPS_LOADED[]
                error("JLL mode: Restart Julia for changes to take effect.")
            end
        end
    end

    """
        LegatePreferences.use_conda(conda_env::String; export_prefs = false, force = true)

    Tells Legate.jl | cuNumeric.jl to use existing conda install. We make no gurantees of compiler compatability at this time.

    Expects `conda_env` to be the absolute path to the root of the environment.
    For example, `/home/julialegate/.conda/envs/cunumeric-gpu`
    """
    function use_conda(env; export_prefs=false, force=true)
        same_mode = MODE == MODE_CONDA
        same_env = env == _conda_env
        if same_mode && same_env
            @info "Already using Conda mode"
        else
            _PREFS_CHANGED[] = true
            @info "Conda mode enabled with env=$env"
            _set("mode" => MODE_CONDA, "conda_env" => env; export_prefs, force)
            if _DEPS_LOADED[]
                error("Conda mode: Restart Julia for changes to take effect. You will need Pkg.build().")
            end
        end
    end

    """
    LegatePreferences.use_developer_mode(; use_jll=true, path=nothing, export_prefs = false, force = true)

    Tells Legate.jl | cuNumeric.jl to enable developer mode. Developer mode allows you to build from source.

    To disable using legate_jll or cupynumeric_jll: ```use_jll=false``` 
    If you disable legate_jll or cupynumeric_jll, then you need to set a path to Legate|cuPyNumeric with ```path="/path/to/Legate|cuPyNumeric"```
    """
    function use_developer_mode(;use_jll=$default_use_jll,
                                 path=$default_path,
                                 export_prefs=false, force=true)

        if !use_jll && isnothing(path)
            error("Must provide path when not using JLL")
        end

        same_mode = MODE == MODE_DEVELOPER
        same_jll_usage = use_jll == _use_jll
        same_path = path == _path

        if same_mode && same_jll_usage && same_path
            @info "Already using Developer mode with the same settings"
            return
        end

        _PREFS_CHANGED[] = true
        _set("mode" => MODE_DEVELOPER,
             "use_jll" => use_jll,
             "path" => path;
             export_prefs, force)

        @info "Developer mode enabled"
        if _DEPS_LOADED[]
            error("Developer Mode: Restart Julia for changes to take effect. You will need to run Pkg.build().")
        end
    end
end)

end

end # module
