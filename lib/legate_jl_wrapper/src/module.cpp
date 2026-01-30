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

#include <type_traits>
#include <vector>

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"
#include "task.h"
#include "types.h"
#include "wrapper.inl"

legate::Type type_from_code(legate::Type::Code type_id) {
  switch (type_id) {
    case legate::Type::Code::BOOL:
      return legate::bool_();
    case legate::Type::Code::INT8:
      return legate::int8();
    case legate::Type::Code::INT16:
      return legate::int16();
    case legate::Type::Code::INT32:
      return legate::int32();
    case legate::Type::Code::INT64:
      return legate::int64();
    case legate::Type::Code::UINT8:
      return legate::uint8();
    case legate::Type::Code::UINT16:
      return legate::uint16();
    case legate::Type::Code::UINT32:
      return legate::uint32();
    case legate::Type::Code::UINT64:
      return legate::uint64();
    case legate::Type::Code::FLOAT16:
      return legate::float16();
    case legate::Type::Code::FLOAT32:
      return legate::float32();
    case legate::Type::Code::FLOAT64:
      return legate::float64();
    case legate::Type::Code::COMPLEX64:
      return legate::complex64();
    case legate::Type::Code::COMPLEX128:
      return legate::complex128();
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

  mod.add_type<Library>("Library");
  mod.add_type<Variable>("Variable");
  mod.add_type<Constraint>("Constraint");

  mod.add_bits<LocalTaskID>("LocalTaskID");
  mod.add_bits<GlobalTaskID>("GlobalTaskID");

  mod.add_bits<legate::mapping::StoreTarget>("StoreTarget");

  mod.add_type<Shape>("Shape").constructor<std::vector<std::uint64_t>>();

  // TODO: add DomainPoint and Domain for manual tasking interface
  // mod.add_type<DomainPoint>("DomainPoint").constructor<Point>();
  // mod.add_type<Domain>("Domain").constructor<DomainPoint, DomainPoint>();

  mod.add_type<Scalar>("Scalar")
      .constructor<float>()
      .constructor<double>()
      .constructor<int32_t>()
      .constructor<void*>();

  mod.add_type<Parametric<TypeVar<1>>>("StdOptional")
      .apply<std::optional<legate::Type>, std::optional<int64_t>>(
          WrapDefault());

  mod.add_type<legate::Slice>("Slice")
      .constructor<std::optional<int64_t>, std::optional<int64_t>>();

  mod.add_type<Parametric<TypeVar<1>>>("StoreTargetOptional")
      .apply<std::optional<legate::mapping::StoreTarget>>(WrapDefault());

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

  mod.add_type<AutoTask>("AutoTask")
      .method("add_input", static_cast<Variable (AutoTask::*)(LogicalArray)>(
                               &AutoTask::add_input))
      .method("add_output", static_cast<Variable (AutoTask::*)(LogicalArray)>(
                                &AutoTask::add_output))
      .method("add_scalar", static_cast<void (AutoTask::*)(const Scalar&)>(
                                &AutoTask::add_scalar_arg))
      .method("add_constraint",
              static_cast<void (AutoTask::*)(const Constraint&)>(
                  &AutoTask::add_constraint));

  mod.add_type<ManualTask>("ManualTask")
      .method("add_input", static_cast<void (ManualTask::*)(LogicalStore)>(
                               &ManualTask::add_input))
      .method("add_output", static_cast<void (ManualTask::*)(LogicalStore)>(
                                &ManualTask::add_output))
      .method("add_scalar", static_cast<void (ManualTask::*)(const Scalar&)>(
                                &ManualTask::add_scalar_arg));

  /* runtime */
  mod.add_type<Runtime>("Runtime");
  mod.method("start_legate", &legate_wrapper::runtime::start_legate);
  mod.method("legate_finish", &legate_wrapper::runtime::legate_finish);
  mod.method("get_runtime", &legate_wrapper::runtime::get_runtime);
  mod.method("has_started", &legate_wrapper::runtime::has_started);
  mod.method("has_finished", &legate_wrapper::runtime::has_finished);
  /* tasking */
  mod.method("align", &legate_wrapper::tasking::align);
  mod.method("create_auto_task", &legate_wrapper::tasking::create_auto_task);
  mod.method("submit_auto_task", &legate_wrapper::tasking::submit_auto_task);
  mod.method("submit_manual_task",
             &legate_wrapper::tasking::submit_manual_task);
  /* array management */
  mod.method("create_unbound_array",
             &legate_wrapper::data::create_unbound_array);
  mod.method("create_array", &legate_wrapper::data::create_array);
  mod.method("create_unbound_store",
             &legate_wrapper::data::create_unbound_store);
  mod.method("create_store", &legate_wrapper::data::create_store);
  mod.method("store_from_scalar", &legate_wrapper::data::store_from_scalar);
  /* type management */
  mod.method("string_to_scalar", &legate_wrapper::data::string_to_scalar);
  /* timing */
  mod.method("time_microseconds", &legate_wrapper::time::time_microseconds);
  mod.method("time_nanoseconds", &legate_wrapper::time::time_nanoseconds);

  wrap_ufi(mod);
}
