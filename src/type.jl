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
    Bool => Legate.bool_,
    Int8 => Legate.int8,
    Int16 => Legate.int16,
    Int32 => Legate.int32,
    Int64 => Legate.int64,
    UInt8 => Legate.uint8,
    UInt16 => Legate.uint16,
    UInt32 => Legate.uint32,
    UInt64 => Legate.uint64,
    Float16 => Legate.float16,
    Float32 => Legate.float32,
    Float64 => Legate.float64,
    # ComplexF16 => Legate.complex32,  #COMMENTED OUT IN WRAPPER
    ComplexF32 => Legate.complex64,
    ComplexF64 => Legate.complex128,
)

# hate this but casting to Int gets around 
# a bug where TypeCode can't be used at compile time 
global const code_type_map = Dict{Int,Type}(
    Int(Legate.BOOL) => Bool,
    Int(Legate.INT8) => Int8,
    Int(Legate.INT16) => Int16,
    Int(Legate.INT32) => Int32,
    Int(Legate.INT64) => Int64,
    Int(Legate.UINT8) => UInt8,
    Int(Legate.UINT16) => UInt16,
    Int(Legate.UINT32) => UInt32,
    Int(Legate.UINT64) => UInt64,
    Int(Legate.FLOAT16) => Float16,
    Int(Legate.FLOAT32) => Float32,
    Int(Legate.FLOAT64) => Float64,
    Int(Legate.COMPLEX64) => ComplexF32,
    Int(Legate.COMPLEX128) => ComplexF64,
    Int(Legate.STRING) => String, # CxxString?
)

to_legate_type(T::Type) = Legate.type_map[T]()

function LType(T::Type)
    return Legate.type_map[T]()
end
