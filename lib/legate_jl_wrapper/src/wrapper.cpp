/* Copyright 2025 Northwestern University,
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

#include "legate.h"
#include "legion.h"

#include "legate/timing/timing.h"
#include "legate/mapping/machine.h"
#include "legate/runtime/runtime.h"
#include "legion/legion_config.h"

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"

#include <type_traits>
#include <vector>

#include "types.h"

using namespace legate;

legate::Type type_from_code(legate::Type::Code type_id) {
  switch (type_id) {
    case legate::Type::Code::BOOL:       return legate::bool_();
    case legate::Type::Code::INT8:       return legate::int8();
    case legate::Type::Code::INT16:      return legate::int16();
    case legate::Type::Code::INT32:      return legate::int32();
    case legate::Type::Code::INT64:      return legate::int64();
    case legate::Type::Code::UINT8:      return legate::uint8();
    case legate::Type::Code::UINT16:     return legate::uint16();
    case legate::Type::Code::UINT32:     return legate::uint32();
    case legate::Type::Code::UINT64:     return legate::uint64();
    case legate::Type::Code::FLOAT16:    return legate::float16();
    case legate::Type::Code::FLOAT32:    return legate::float32();
    case legate::Type::Code::FLOAT64:    return legate::float64();
    case legate::Type::Code::COMPLEX64:  return legate::complex64();
    case legate::Type::Code::COMPLEX128: return legate::complex128();
    default:
      throw std::invalid_argument("Unsupported legate::Type::Code enum value.");
  }
}

struct WrapDefault {
    template <typename TypeWrapperT>
    void operator()(TypeWrapperT&& wrapped) {
      typedef typename TypeWrapperT::type WrappedT;
      wrapped.template constructor<typename WrappedT::value_type>();
    }
  };

legate::Scalar string_to_scalar(std::string str) {
   return legate::Scalar(str);
}


JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    using jlcxx::ParameterList;
    using jlcxx::Parametric;
    using jlcxx::TypeVar;

    wrap_privilege_modes(mod);
    wrap_type_enums(mod);
    wrap_type_getters(mod);

    using privilege_modes = ParameterList<
      std::integral_constant<legion_privilege_mode_t, LEGION_WRITE_DISCARD>,
      std::integral_constant<legion_privilege_mode_t, LEGION_READ_ONLY>>;

    mod.method("start_legate", [] { legate::start(); });  // in legate/runtime.h
    mod.method("legate_finish", &legate::finish);  // in legate/runtime.h

    mod.add_bits<LocalTaskID>("LocalTaskID");
    mod.add_bits<GlobalTaskID>("GlobalTaskID");

    mod.add_type<Shape>("Shape")
        .constructor<std::vector<std::uint64_t>>();

    mod.add_type<Scalar>("Scalar")
        .constructor<float>()
        .constructor<double>()
        .constructor<int>(); // julia lets me make with ints???
    
    mod.add_type<std::vector<legate::Scalar>>("VectorScalar")
        .method("push_back", [](std::vector<legate::Scalar>& v, const legate::Scalar& x) {
        v.push_back(x);
        })
        .method("size", [](const std::vector<legate::Scalar>& v) {
        return v.size();
        })
        .method("get", [](const std::vector<legate::Scalar>& v, std::size_t i) {
        return v.at(i);
        })
        .method("set", [](std::vector<legate::Scalar>& v, std::size_t i, const legate::Scalar& x) {
        v.at(i) = x;
        });
    mod.method("string_to_scalar", &string_to_scalar);

    mod.add_type<Parametric<TypeVar<1>>>("StdOptional")
      .apply<std::optional<legate::Type>, std::optional<int64_t>>(WrapDefault());

    // mod.add_type<legate::Slice>("Slice")
    //   .constructor<std::optional<int64_t>, std::optional<int64_t>>(jlcxx::kwarg("_start") = Slice::OPEN, jlcxx::kwarg("_stop") = Slice::OPEN);
    
    mod.add_type<legate::Slice>("Slice")
      .constructor<std::optional<int64_t>, std::optional<int64_t>>();


    mod.add_type<std::vector<legate::Slice>>("VectorSlice")
      .method("push", [](std::vector<legate::Slice>& v, legate::Slice s) {
        v.push_back(s);
      });

    mod.add_bits<legate::mapping::StoreTarget>("StoreTarget");
    
    mod.add_type<Parametric<TypeVar<1>>>("StoreTargetOptional")
      .apply<std::optional<legate::mapping::StoreTarget>>(WrapDefault());
    
    mod.add_type<Library>("Library");
   
    // This has all the accessor methods
    mod.add_type<PhysicalStore>("PhysicalStore")
        .method("dim", &PhysicalStore::dim)
        .method("type", &PhysicalStore::type)
        .method("is_readable", &PhysicalStore::is_readable)
        .method("is_writable", &PhysicalStore::is_writable)
        .method("is_reducible", &PhysicalStore::is_reducible)
        .method("valid", &PhysicalStore::valid);

    mod.add_type<LogicalStore>("LogicalStore")
        .method("dim", &LogicalStore::dim)
        .method("type", &LogicalStore::type)
        .method("reinterpret_as", &LogicalStore::reinterpret_as)
        .method("promote", &LogicalStore::promote)
        .method("slice", &LogicalStore::slice)
        .method("get_physical_store", &LogicalStore::get_physical_store)
        .method("equal_storage", &LogicalStore::equal_storage);

    mod.add_type<PhysicalArray>("PhysicalArray")
        .method("nullable", &PhysicalArray::nullable)
        .method("dim", &PhysicalArray::dim)
        .method("type", &PhysicalArray::type)
        .method("data", &PhysicalArray::data);

    mod.add_type<LogicalArray>("LogicalArray")
        .method("dim", &LogicalArray::dim)
        .method("type", &LogicalArray::type)
        .method("unbound", &LogicalArray::unbound)
        .method("nullable", &LogicalArray::nullable);

    mod.add_type<Variable>("Variable");
    mod.add_type<Constraint>("Constraint");

    mod.add_type<AutoTask>("AutoTask")
        .method("add_input", static_cast<Variable (AutoTask::*)(LogicalArray)>(&AutoTask::add_input))
        .method("add_output", static_cast<Variable (AutoTask::*)(LogicalArray)>(&AutoTask::add_output))
        .method("add_scalar", static_cast<void (AutoTask::*)(const Scalar&)>(&AutoTask::add_scalar_arg))
        .method("add_constraint", static_cast<void (AutoTask::*)(const Constraint&)>(&AutoTask::add_constraint)); 
              
    mod.add_type<ManualTask>("ManualTask")
        .method("add_input", static_cast<void (ManualTask::*)(LogicalStore)>(&ManualTask::add_input))
        .method("add_output", static_cast<void (ManualTask::*)(LogicalStore)>(&ManualTask::add_output))
        .method("add_scalar", static_cast<void (ManualTask::*)(const Scalar&)>(&ManualTask::add_scalar_arg));

    mod.add_type<Runtime>("Runtime")
      .method("create_auto_task", [](Runtime* rt, Library lib, LocalTaskID id) { return rt->create_task(lib, id); })
      .method("submit_auto_task", [](Runtime* rt, AutoTask& task) { return rt->submit(std::move(task));})
      .method("submit_manual_task", [](Runtime* rt, ManualTask& task) {return rt->submit(std::move(task));});

    mod.method("align", [](const Variable& a, const Variable& b) {
      return legate::align(a, b);
    });


    mod.method("get_runtime", [] { return Runtime::get_runtime(); });
    mod.method("create_unbound_array",
      [](const Type& ty, std::uint32_t dim = 1, bool nullable = false) {
        // Type ty = type_from_code(Type::Code(type_id));
        return Runtime::get_runtime()->create_array(ty, dim, nullable);
      } //,
      // jlcxx::kwarg("dim") = 1,
      // jlcxx::kwarg("nullable") = false
    );

    mod.method("create_array",
      [](const Shape& shape, const Type& ty, bool nullable = false, bool optimize_scalar = false) {
        // Type ty = type_from_code(Type::Code(type_id));
        return Runtime::get_runtime()->create_array(shape, ty, nullable, optimize_scalar);
      } //,
      // jlcxx::kwarg("nullable") = false,
      // jlcxx::kwarg("optimize_scalar") = false
    );

    // create_unbound_store with Type, uint32_t
    mod.method("create_unbound_store",
      [](const Type& ty, std::uint32_t dim = 1) {
        // Type ty = type_from_code(Type::Code(type_id));
        return Runtime::get_runtime()->create_store(ty, dim);
      } //,
      // jlcxx::kwarg("dim") = 1
    );

    // // create_store with Shape, Type, bool
    mod.method("create_store",
      [](const Shape& shape, const Type& ty, bool optimize_scalar = false) {
        // Type ty = type_from_code(Type::Code(type_id));
        return Runtime::get_runtime()->create_store(shape, ty, optimize_scalar);
      } //,
      // jlcxx::kwarg("optimize_scalar") = false
    );

    // store_from_scalar (Shape as kwarg)
    mod.method("store_from_scalar",
      [](const Scalar& scalar, const Shape& shape = Shape{1}) {
        return Runtime::get_runtime()->create_store(scalar, shape);
      } //,
      // jlcxx::kwarg("shape") = Shape{1}
    );

    // intialization & cleanup
    // TODO catch the (Auto)ConfigurationError and make the Julia error nicer
    mod.method("start", static_cast<void (*)()>(&legate::start), jlcxx::calling_policy::std_function);
    mod.method("has_started", &legate::has_started);
    mod.method("finish", &legate::finish);
    mod.method("has_finished", &legate::has_finished);


    // timing methods
    mod.add_type<timing::Time>("Time").method(
        "value", &timing::Time::value);
    mod.method("time_microseconds", &timing::measure_microseconds);
    mod.method("time_nanoseconds", &timing::measure_nanoseconds);

}
