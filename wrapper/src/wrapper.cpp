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
#include "legion/legion_config.h"

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"

#include <type_traits>
#include <vector>

#include "types.h"

using namespace legate;


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
    
    mod.add_type<Library>("LegateLibrary");
   
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

    mod.add_type<Variable>("LegateVariable");

    mod.add_type<AutoTask>("AutoTask")
        .method("add_input", static_cast<Variable (AutoTask::*)(LogicalArray)>(&AutoTask::add_input))
        .method("add_output", static_cast<Variable (AutoTask::*)(LogicalArray)>(&AutoTask::add_output));
    mod.add_type<ManualTask>("ManualTask")
        .method("add_input", static_cast<void (ManualTask::*)(LogicalStore)>(&ManualTask::add_input))
        .method("add_output", static_cast<void (ManualTask::*)(LogicalStore)>(&ManualTask::add_output));

    mod.add_type<Runtime>("LegateRuntime");
    // mod.method("create_auto_task", static_cast<AutoTask (Runtime::*)(Library, LocalTaskID)>(&Runtime::create_task));
    // mod.method("submit_auto_task", [](Runtime& self, AutoTask& task) {self.submit(std::move(task));});
    // mod.method("submit_manual_task", [](Runtime& self, ManualTask& task) {self.submit(std::move(task));});
    // mod.method("create_unbound_array", static_cast<LogicalArray (Runtime::*)(const Type&, std::uint32_t, bool)>(&Runtime::create_array),
    //              jlcxx::kwarg("dim") = 1, jlcxx::kwarg("nullable") = false);
    // mod.method("create_array", static_cast<LogicalArray (Runtime::*)(const Shape&, const Type&, bool, bool)>(&Runtime::create_array),
    //              jlcxx::kwarg("nullable") = false, jlcxx::kwarg("optimize_scalar") = false);
    // mod.method("create_unbound_store", static_cast<LogicalStore (Runtime::*)(const Type&, std::uint32_t)>(&Runtime::create_store), 
    //              jlcxx::kwarg("dim") = 1);
    // mod.method("create_store", static_cast<LogicalStore (Runtime::*)(const Shape&, const Type&, bool)>(&Runtime::create_store),
    //              jlcxx::kwarg("optimize_scalar") = false);
    // mod.method("store_from_scalar", static_cast<LogicalStore (Runtime::*)(const Scalar&, const Shape&)>(&Runtime::create_store),
    //              jlcxx::kwarg("shape") = Shape{1});

    // intialization & cleanup
    // TODO catch the (Auto)ConfigurationError and make the Julia error nicer.
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