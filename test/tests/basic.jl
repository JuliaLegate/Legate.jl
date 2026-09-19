

@testset verbose=true "Enums" begin

    @test isdefined(@__MODULE__, :LegionPrivilegeMode) 
    @test isdefined(@__MODULE__, :TypeCode)

    @test Legate.to_legate_type(Int8) == Legate.TypeCode.Int8
    @test Legate.to_legate_type(Int16) == Legate.TypeCode.Int16
    @test Legate.to_legate_type(Int32) == Legate.TypeCode.Int32
    @test Legate.to_legate_type(Int64) == Legate.TypeCode.Int64
    @test Legate.to_legate_type(UInt8) == Legate.TypeCode.UInt8
    @test Legate.to_legate_type(UInt16) == Legate.TypeCode.UInt16
    @test Legate.to_legate_type(UInt32) == Legate.TypeCode.UInt32
    @test Legate.to_legate_type(UInt64) == Legate.TypeCode.UInt64
    @test Legate.to_legate_type(Float32) == Legate.TypeCode.Float32
    @test Legate.to_legate_type(Float64) == Legate.TypeCode.Float64

end