function load_jll_lib(jll, lib)
    dir = joinpath(jll.artifact_dir, "lib")
    libpath = joinpath(dir, lib)
    try
        Libdl.dlopen(libpath, Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW)
    catch e
        @warn "Failed to open $(lib)" path=libpath exception=e
    end
    push!(Base.DL_LOAD_PATH, dir)
    return dir
end

function find_gpu_libs()
    CUDA_DRIVER_LIB = load_jll_lib(CUDA_Driver_jll, "libcuda.so")
    CUDA_RUNTIME_LIB = load_jll_lib(CUDA_Runtime_jll, "libcudart.so")
    return CUDA_DRIVER_LIB, CUDA_RUNTIME_LIB
end

function is_legate_installed(legate_dir::String; throw_errors::Bool = false)
    include_dir = joinpath(legate_dir, "include")
    if !isdir(joinpath(include_dir, "legate/legate"))
        throw_errors && @error "Legate.jl: Cannot find include/legate/legate in $(legate_dir)"
        return false
    end 
    return true
end

function parse_legate_version(legate_dir)
    version_file = joinpath(legate_dir, "include", "legate/legate", "version.h")

    version = nothing
    open(version_file, "r") do f
        data = readlines(f)
        major = parse(Int, split(data[end-2])[end])
        minor = lpad(split(data[end-1])[end], 2, '0')
        patch = lpad(split(data[end])[end], 2, '0')
        version = "$(major).$(minor).$(patch)"
    end

    if isnothing(version)
        error("Legate.jl: Failed to parse version")
    end
    return version
end

function check_if_patch(legate_dir)
    patch = joinpath(legate_dir, "include", "legate/legate", "patch")
    if isfile(patch)
        return true
    end
    return false
end

function check_legate_install(legate_dir)
    is_legate_installed(legate_dir; throw_errors=true)

    installed_version = parse_legate_version(legate_dir)
    if installed_version ∉ SUPPORTED_LEGATE_VERSIONS
        error("Legate.jl: $(legate_dir) detected unsupported version $(installed_version)")
    end

    patch = check_if_patch(legate_dir)
    if patch == false
        error("Legate.jl: legate does not have patch. Please run Pkg.build()")
    end

    @info "Legate.jl: Found a valid install in: $(legate_dir)"
    return true
end

function get_library_root(jll_module, env_var::String)
    if haskey(ENV, env_var)
        return get(ENV, env_var, "0")
    elseif jll_module.is_available()
        return joinpath(jll_module.artifact_dir, "lib")
    else
        error("$env_var not found via environment or JLL.")
    end
end

function find_preferences()
    pkg_root = abspath(joinpath(@__DIR__, "../"))

    cuda_driver_lib, cuda_runtime_lib = find_gpu_libs()
    CUDA.precompile_runtime()

    mpi_lib = get_library_root(MPICH_jll, "JULIA_LEGATE_MPI_PATH")
    hdf5_lib = get_library_root(HDF5_jll, "JULIA_LEGATE_HDF5_PATH")
    set_preferences!(LegatePreferences, "HDF5_LIB" => hdf5_lib, force=true)
    set_preferences!(LegatePreferences, "MPI_LIB" => mpi_lib, force=true)

    nccl_lib = get_library_root(NCCL_jll, "JULIA_LEGATE_NCCL_PATH")
    legate_wrapper_lib = joinpath(legate_jl_wrapper_jll.artifact_dir, "lib")
    
    legate_path = legate_jll.artifact_dir

    mode = load_preference(LegatePreferences, "mode", LegatePreferences.MODE_JLL)

    # if developer mode
    if mode == LegatePreferences.MODE_DEVELOPER
        use_legate_jll = load_preference(LegatePreferences, "use_legate_jll", LegatePreferences.DEVEL_DEFAULT_JLL_CONFIG)
        if use_legate_jll == false
            legate_path = load_preference(LegatePreferences, "legate_path", LegatePreferences.DEVEL_DEFAULT_LEGATE_PATH)
            check_legate_install(legate_path)
        end 
        legate_wrapper_lib = joinpath(pkg_root, "deps", "legate_jl_wrapper", "lib")
    # if conda
    elseif mode == LegatePreferences.MODE_CONDA
        @warn "mode = conda may break. We are using a subset of libraries from conda."
        conda_env = load_preference(LegatePreferences, "conda_env", nothing)
        check_legate_install(conda_env)
        # so right now, we just use nccl and legate from the conda env
        # mpi is not available, hdf5 throws symbol errors
        legate_path = conda_env
        nccl_lib = joinpath(conda_env, "lib")
    end

    legate_lib = joinpath(legate_path, "lib")

    set_preferences!(LegatePreferences, "CUDA_DRIVER_LIB" => cuda_driver_lib, force=true)
    set_preferences!(LegatePreferences, "CUDA_RUNTIME_LIB" => cuda_runtime_lib, force=true)
    set_preferences!(LegatePreferences, "NCCL_LIB" =>  nccl_lib, force=true)
    set_preferences!(LegatePreferences, "LEGATE_LIB" => legate_lib, force=true)
    set_preferences!(LegatePreferences, "LEGATE_WRAPPER_LIB" => legate_wrapper_lib, force=true)
end