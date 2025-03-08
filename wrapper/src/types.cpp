#include "legate.h"
#include "types.h"

void wrap_type_enums(jlcxx::Module& mod) {

    auto lt = mod.add_type<legate::Type>("LegateType");
  
    mod.add_bits<legate::Type::Code>("TypeCode", jlcxx::julia_type("CppEnum"));
    mod.set_const("BOOL", legate::Type::Code::BOOL); //legion_type_id_t::LEGION_TYPE_BOOL
    mod.set_const("INT8", legate::Type::Code::INT8); //legion_type_id_t::LEGION_TYPE_INT8
    mod.set_const("INT16", legate::Type::Code::INT16); // legion_type_id_t::LEGION_TYPE_INT16);
    mod.set_const("INT32", legate::Type::Code::INT32); // legion_type_id_t::LEGION_TYPE_INT32);
    mod.set_const("INT64", legate::Type::Code::INT64); // legion_type_id_t::LEGION_TYPE_INT64);
    mod.set_const("UINT8", legate::Type::Code::UINT8); // legion_type_id_t::LEGION_TYPE_UINT8);
    mod.set_const("UINT16", legate::Type::Code::UINT16); // legion_type_id_t::LEGION_TYPE_UINT16);
    mod.set_const("UINT32", legate::Type::Code::UINT32); // legion_type_id_t::LEGION_TYPE_UINT32);
    mod.set_const("UINT64", legate::Type::Code::UINT64); //legion_type_id_t::LEGION_TYPE_UINT64);
    mod.set_const("FLOAT16", legate::Type::Code::FLOAT16); //legion_type_id_t::LEGION_TYPE_FLOAT16);
    mod.set_const("FLOAT32", legate::Type::Code::FLOAT32); // legion_type_id_t::LEGION_TYPE_FLOAT32);
    mod.set_const("FLOAT64", legate::Type::Code::FLOAT64); // legion_type_id_t::LEGION_TYPE_FLOAT64);
    mod.set_const("COMPLEX64", legate::Type::Code::COMPLEX64); // legion_type_id_t::LEGION_TYPE_COMPLEX64);
    mod.set_const("COMPLEX128", legate::Type::Code::COMPLEX128); // legion_type_id_t::LEGION_TYPE_COMPLEX128);
    mod.set_const("NIL", legate::Type::Code::NIL);
    mod.set_const("BINARY", legate::Type::Code::BINARY);
    mod.set_const("FIXED_ARRAY", legate::Type::Code::FIXED_ARRAY);
    mod.set_const("STRUCT", legate::Type::Code::STRUCT);
    mod.set_const("STRING", legate::Type::Code::STRING);
    mod.set_const("LIST", legate::Type::Code::LIST);
  
    lt.method("code", &legate::Type::code);
    // lt.method("to_string", &legate::Type::to_string); // ABI issue :)
  }