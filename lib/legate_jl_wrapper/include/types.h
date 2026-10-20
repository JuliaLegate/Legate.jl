#pragma once

#include <legate.h>

#include "jlcxx/jlcxx.hpp"

namespace legate_util {
template <legate::Type::Code CODE>
struct code_to_cxx;

#define DEFINE_CODE_TO_CXX(code_enum, cxx_type)       \
  template <>                                         \
  struct code_to_cxx<legate::Type::Code::code_enum> { \
    using type = cxx_type;                            \
  };

DEFINE_CODE_TO_CXX(BOOL, bool)
DEFINE_CODE_TO_CXX(INT8, int8_t)
DEFINE_CODE_TO_CXX(INT16, int16_t)
DEFINE_CODE_TO_CXX(INT32, int32_t)
DEFINE_CODE_TO_CXX(INT64, int64_t)
DEFINE_CODE_TO_CXX(UINT8, uint8_t)
DEFINE_CODE_TO_CXX(UINT16, uint16_t)
DEFINE_CODE_TO_CXX(UINT32, uint32_t)
DEFINE_CODE_TO_CXX(UINT64, uint64_t)
#ifdef HAVE_CUDA
DEFINE_CODE_TO_CXX(FLOAT16, __half)
#else
// Dummy type for FLOAT16 when CUDA is not available
// This allows compilation but we throw an error if actually used
struct __half_dummy {};
DEFINE_CODE_TO_CXX(FLOAT16, __half_dummy)
#endif
DEFINE_CODE_TO_CXX(FLOAT32, float)
DEFINE_CODE_TO_CXX(FLOAT64, double)
DEFINE_CODE_TO_CXX(COMPLEX64, std::complex<float>)
DEFINE_CODE_TO_CXX(COMPLEX128, std::complex<double>)
#undef DEFINE_CODE_TO_CXX
}  // namespace legate_util

// Wraps the enums which define how legate
//  and cupynumeric types map to legion types
void wrap_type_enums(jlcxx::Module&);

// Wraps the legate functions which return the
// specified legate::Type. (e.g. legate::int8())
void wrap_type_getters(jlcxx::Module&);

// Wraps the privilege modes used in
// FieldAccessor (AcessorRO, AccessorWO)
void wrap_privilege_modes(jlcxx::Module&);
