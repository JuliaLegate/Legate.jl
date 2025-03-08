#pragma once

#include "jlcxx/jlcxx.hpp"

//Wraps the enums which define how legate
// and cupynumeric types map to legion types
void wrap_type_enums(jlcxx::Module&);