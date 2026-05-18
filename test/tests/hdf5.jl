using HDF5

# Write a Julia array to HDF5 using HDF5.jl, read it back with Legate, and compare.
function test_hdf5_read(T::Type, shape::Tuple)
    path = tempname() * ".h5"
    dataset = "data"
    original = rand(T, shape)

    h5open(path, "w") do f
        write(f, dataset, original)
    end

    legate_arr = Legate.read_hdf5(path, dataset)
    result = Array(legate_arr)

    rm(path; force=true)
    return result == original
end

# Write a LogicalArray with Legate.write_hdf5, read it back with HDF5.jl, and compare.
function test_hdf5_write(T::Type, shape::Tuple)
    path = tempname() * ".h5"
    dataset = "data"
    original = rand(T, shape...)

    legate_arr = Legate.LogicalArray(original)
    Legate.write_hdf5(legate_arr, path, dataset)
    Legate.runtime_sync()

    result = h5open(path, "r") do f
        read(f, dataset)
    end

    rm(path; force=true)
    return result == original
end

# Write with Legate, read back with Legate (roundtrip).
function test_hdf5_roundtrip(T::Type, shape::Tuple)
    path = tempname() * ".h5"
    dataset = "data"
    original = rand(T, shape...)

    legate_arr = Legate.LogicalArray(original)
    Legate.write_hdf5(legate_arr, path, dataset)
    Legate.runtime_sync()

    result_arr = Legate.read_hdf5(path, dataset)
    result = Array(result_arr)

    rm(path; force=true)
    return result == original
end

@testset verbose = true "HDF5 Interoperability" begin
    for T in Base.uniontypes(Legate.SUPPORTED_NUMERIC_TYPES)
        @testset "Type: $T" begin
            for shape in [(10,), (4, 5), (3, 4, 5)]
                @testset "Shape: $shape" begin
                    @testset "HDF5.jl write → Legate read" begin
                        @test test_hdf5_read(T, shape)
                    end
                    @testset "Legate write → HDF5.jl read" begin
                        @test test_hdf5_write(T, shape)
                    end
                    @testset "Legate roundtrip" begin
                        @test test_hdf5_roundtrip(T, shape)
                    end
                end
            end
        end
    end
end
