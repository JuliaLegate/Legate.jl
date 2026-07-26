const ROW_MAJOR_FILE_STAB = joinpath(@__DIR__, "..", "data", "row_major.h5")

@testset verbose = true "Stability Tests" begin
    # Array(::LogicalArray) preserves eltype; inferrable for both :row and :col stores.
    row1 = Legate.LogicalArray(rand(5))                              # :row 1-D
    row2 = Legate.LogicalArray(rand(3, 4))                           # :row 2-D
    col2 = Legate.h5read(ROW_MAJOR_FILE_STAB, "mat2d"; layout=:col)  # :col 2-D
    col3 = Legate.h5read(ROW_MAJOR_FILE_STAB, "mat3d"; layout=:col)  # :col 3-D

    @test @inferred(Array(row1)) !== nothing
    @test @inferred(Array(row2)) !== nothing
    @test @inferred(Array(col2)) !== nothing
    @test @inferred(Array(col3)) !== nothing

    @test @inferred(Array{Float64}(row2)) !== nothing
    @test @inferred(Array{Int64}(col3)) !== nothing
end
