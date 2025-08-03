 # add these dummy functions for documentation
const code_type_map = Dict{Int, DataType}(
        1 => Float32,
        2 => Float64,
        3 => Int64,
    )
function to_legate_type(::Type{T}) where T
    return T
end