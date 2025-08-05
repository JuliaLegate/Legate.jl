function update_project(version::String)
    Pkg.compat("legate_jl_wrapper_jll", version)
    Pkg.compat("legate_jll", version)

    path = "Project.toml"
    project = TOML.parsefile(path)
    project["version"] = version

    open(path, "w") do io
        TOML.print(io, project)
    end
end

function get_install_liblegate()
    return LEGATE_LIB
end

function get_install_libnccl()
    return NCCL_LIB
end

function get_install_libmpi()
    return MPI_LIB
end

function get_install_libhdf5()
    return HDF5_LIB
end

function get_install_libcuda()
    return CUDA_DRIVER_LIB
end

function get_install_libcudart()
    return CUDA_RUNTIME_LIB
end
