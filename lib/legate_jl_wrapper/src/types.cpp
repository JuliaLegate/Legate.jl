/* Copyright 2026 Northwestern University,
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
 */

#include "types.h"
#include "legate.h"
#include "legion/api/config.h"

#include <vector>

// forced to return a std::int32_t instead of enum code
inline std::int32_t code(legate::Type ty) { return (int32_t)ty.code(); }

void wrap_type_enums(jlcxx::Module& mod) {
  auto lt = mod.add_type<legate::Type>("LegateType");

  mod.add_enum<legate::Type::Code>("TypeCode",
    std::vector<const char*>({
      "BOOL",
      "INT8",
      "INT16",
      "INT32",
      "INT64",
      "UINT8",
      "UINT16",
      "UINT32",
      "UINT64",
      "FLOAT16",
      "FLOAT32",
      "FLOAT64",
      "COMPLEX64",
      "COMPLEX128",
      "NIL",
      "BINARY",
      "FIXED_ARRAY",
      "STRUCT",
      "STRING",
      "LIST"
    }),
    std::vector<int>({
      static_cast<int>(legate::Type::Code::BOOL),
      static_cast<int>(legate::Type::Code::INT8),
      static_cast<int>(legate::Type::Code::INT16),
      static_cast<int>(legate::Type::Code::INT32),
      static_cast<int>(legate::Type::Code::INT64),
      static_cast<int>(legate::Type::Code::UINT8),
      static_cast<int>(legate::Type::Code::UINT16),
      static_cast<int>(legate::Type::Code::UINT32),
      static_cast<int>(legate::Type::Code::UINT64),
      static_cast<int>(legate::Type::Code::FLOAT16),
      static_cast<int>(legate::Type::Code::FLOAT32),
      static_cast<int>(legate::Type::Code::FLOAT64),
      static_cast<int>(legate::Type::Code::COMPLEX64),
      static_cast<int>(legate::Type::Code::COMPLEX128),
      static_cast<int>(legate::Type::Code::NIL),
      static_cast<int>(legate::Type::Code::BINARY),
      static_cast<int>(legate::Type::Code::FIXED_ARRAY),
      static_cast<int>(legate::Type::Code::STRUCT),
      static_cast<int>(legate::Type::Code::STRING),
      static_cast<int>(legate::Type::Code::LIST)
    })
  );

  mod.method("code", &code);
}

void wrap_type_getters(jlcxx::Module& mod) {
  mod.method("bool_", &legate::bool_);
  mod.method("int8", &legate::int8);
  mod.method("int16", &legate::int16);
  mod.method("int32", &legate::int32);
  mod.method("int64", &legate::int64);
  mod.method("uint8", &legate::uint8);
  mod.method("uint16", &legate::uint16);
  mod.method("uint32", &legate::uint32);
  mod.method("uint64", &legate::uint64);
  mod.method("float16", &legate::float16);
  mod.method("float32", &legate::float32);
  mod.method("float64", &legate::float64);
  // mod.method("complex32", &legate::complex32);
  mod.method("complex64", &legate::complex64);
  mod.method("complex128", &legate::complex128);
}

void wrap_privilege_modes(jlcxx::Module& mod) {

  mod.add_enum<legion_privilege_mode_t>("LegionPrivilegeMode",
    std::vector<const char*>({
      "LEGION_READ_ONLY",
      "LEGION_WRITE_DISCARD"
    }),
    std::vector<int>({
      static_cast<int>(legion_privilege_mode_t::LEGION_READ_ONLY),
      static_cast<int>(legion_privilege_mode_t::LEGION_WRITE_DISCARD)
    })
  );

}
