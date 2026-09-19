

@testset verbose=true "Enums" begin

    @test isdefined(Legate, :LegionPrivilegeMode) 
    @test isdefined(Legate, :TypeCode)

    @test Legate.to_legate_type(Int8) isa Legate.LegateTypeAllocated

end