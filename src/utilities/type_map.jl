#= Copyright 2026 Northwestern University, 
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
=#

export LType, to_legate_type
# is legion Complex128 same as ComplexF64 in julia? 
# These are methods that return a Legate::Type
global const type_map = Dict{Type,Function}(
    Bool => LegateInternal.bool_,
    Int8 => LegateInternal.int8,
    Int16 => LegateInternal.int16,
    Int32 => LegateInternal.int32,
    Int64 => LegateInternal.int64,
    UInt8 => LegateInternal.uint8,
    UInt16 => LegateInternal.uint16,
    UInt32 => LegateInternal.uint32,
    UInt64 => LegateInternal.uint64,
    Float16 => LegateInternal.float16,
    Float32 => LegateInternal.float32,
    Float64 => LegateInternal.float64,
    # ComplexF16 => LegateInternal.complex32,  #COMMENTED OUT IN WRAPPER
    ComplexF32 => LegateInternal.complex64,
    ComplexF64 => LegateInternal.complex128,
)

# hate this but casting to Int gets around 
# a bug where TypeCode can't be used at compile time 
global const code_type_map = Dict{Int,Type}(
    Int(LegateInternal.BOOL) => Bool,
    Int(LegateInternal.INT8) => Int8,
    Int(LegateInternal.INT16) => Int16,
    Int(LegateInternal.INT32) => Int32,
    Int(LegateInternal.INT64) => Int64,
    Int(LegateInternal.UINT8) => UInt8,
    Int(LegateInternal.UINT16) => UInt16,
    Int(LegateInternal.UINT32) => UInt32,
    Int(LegateInternal.UINT64) => UInt64,
    Int(LegateInternal.FLOAT16) => Float16,
    Int(LegateInternal.FLOAT32) => Float32,
    Int(LegateInternal.FLOAT64) => Float64,
    Int(LegateInternal.COMPLEX64) => ComplexF32,
    Int(LegateInternal.COMPLEX128) => ComplexF64,
    Int(LegateInternal.STRING) => String, # CxxString?
)

to_legate_type(T::Type) = type_map[T]()

# This is the same function as the above. 
# TODO, check if anycode depends on LType calls.
LType(T::Type) = to_legate_type(T)


function get_code_type(type_code; throw=true)
    val = if haskey(code_type_map, type_code)
        code_type_map[type_code]
    else
        throw && error("Unknown type code: $type_code")
        nothing
    end
    return val
end
