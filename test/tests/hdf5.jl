using HDF5

# Write a Julia array to HDF5 using HDF5.jl, read it back with Legate, and compare.
# HDF5.jl stores column-major with reversed dimensions, so read it back with layout=:col.
function test_hdf5_read(T::Type, shape::Tuple)
    path = tempname() * ".h5"
    dataset = "data"
    original = rand(T, shape)

    h5open(path, "w") do f
        return write(f, dataset, original)
    end

    legate_arr = Legate.h5read(path, dataset; layout=:col)
    result = Array(legate_arr)

    rm(path; force=true)
    return result == original
end

# Legate write -> HDF5.jl read: Legate stores row-major, HDF5.jl reads column-major, so the
# dataset comes back transposed (reversed axes).
function test_hdf5_write(T::Type, shape::Tuple)
    path = tempname() * ".h5"
    dataset = "data"
    original = rand(T, shape...)

    legate_arr = Legate.LogicalArray(original)
    Legate.h5write(path, dataset, legate_arr)
    Legate.runtime_sync()

    result = h5open(path, "r") do f
        return read(f, dataset)
    end

    rm(path; force=true)
    return result == permutedims(original, reverse(1:length(shape)))
end

# Legate roundtrip: h5write stores row-major, so read it back with the default layout=:row.
function test_hdf5_roundtrip(T::Type, shape::Tuple)
    path = tempname() * ".h5"
    dataset = "data"
    original = rand(T, shape...)

    legate_arr = Legate.LogicalArray(original)
    Legate.h5write(path, dataset, legate_arr)
    Legate.runtime_sync()

    result_arr = Legate.h5read(path, dataset)
    result = Array(result_arr)

    rm(path; force=true)
    return result == original
end

# numpy/h5py row-major file; HDF5.jl (column-major) is the reference reader.
const ROW_MAJOR_FILE = joinpath(@__DIR__, "..", "data", "row_major.h5")

# :row read == reference with axes reversed.
function test_hdf5_read_row_major_row(dataset::String)
    row = Array(Legate.h5read(ROW_MAJOR_FILE, dataset))
    ref = HDF5.h5read(ROW_MAJOR_FILE, dataset)
    return eltype(row) == eltype(ref) && row == permutedims(ref, reverse(1:ndims(ref)))
end

# :col read == reference directly.
function test_hdf5_read_row_major_col(dataset::String)
    col = Array(Legate.h5read(ROW_MAJOR_FILE, dataset; layout=:col))
    ref = HDF5.h5read(ROW_MAJOR_FILE, dataset)
    return eltype(col) == eltype(ref) && col == ref
end

@testset verbose = true "HDF5 Interoperability" begin
    @testset "numpy/h5py row-major file → Legate read" begin
        for dataset in ("vec1d", "mat2d", "mat3d")
            @testset "$dataset" begin
                @test test_hdf5_read_row_major_row(dataset)
                @test test_hdf5_read_row_major_col(dataset)
            end
        end
    end

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
