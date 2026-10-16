
@testset verbose=true "Enums" begin
    @test isdefined(Legate.LegateInternal, :LegionPrivilegeMode)
    @test isdefined(Legate.LegateInternal, :TypeCode)

    @test Legate.to_legate_type(Int8) isa Legate.LegateInternal.LegateTypeAllocated
end
