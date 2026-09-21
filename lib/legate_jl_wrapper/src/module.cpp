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

#include <complex>
#include <cstdint>
#include <functional>
#include <mutex>
#include <type_traits>
#include <vector>

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"
#include "task.h"
#include "types.h"
#include "wrapper.inl"

// Deferred free: destroying wrapped Legate handles off-thread (Julia's
// multi-threaded GC finalizers, e.g. 1.12's interactive thread) corrupts the
// runtime — some destructors call it (unmap_region) and it is only valid on the
// launch thread. Every wrapped object's finalizer instead enqueues its deleter
// here; the launch thread runs them via legate_drain_frees.
namespace {
std::mutex g_deferred_free_mutex;
std::vector<std::function<void()>> g_deferred_deleters;

void drain_deferred_frees() {
  std::vector<std::function<void()>> local;
  {
    std::lock_guard<std::mutex> lk(g_deferred_free_mutex);
    local.swap(g_deferred_deleters);
  }
  for (auto& del : local) del();
}
}  // namespace

namespace jlcxx {
// Defer destruction of ALL wrapped types to the launch thread (see above).
template <typename T>
struct Finalizer<T, SpecializedFinalizer> {
  static void finalize(T* p) {
    if (p == nullptr) return;
    std::lock_guard<std::mutex> lk(g_deferred_free_mutex);
    g_deferred_deleters.emplace_back([p] { delete p; });
  }
};
}  // namespace jlcxx

struct WrapDefault {
  template <typename TypeWrapperT>
  void operator()(TypeWrapperT&& wrapped) {
    typedef typename TypeWrapperT::type WrappedT;
    wrapped.template constructor<typename WrappedT::value_type>();
  }
};

// Register Scalar(StrictlyTypedNumber<T>) for each numeric element type.
// apply_combination is for Parametric types; for_each_type walks a
// ParameterList and adds constructors on a single non-parametric TypeWrapper.
struct WrapScalarStrictCtors {
  jlcxx::TypeWrapper<Scalar> wrapped;

  template <typename T>
  void operator()() {
    wrapped.constructor(
        [](jlcxx::StrictlyTypedNumber<T> v) { return new Scalar(v.value); });
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

  // Bool/int/uint/float Scalar element types (complex is special-cased below).
  using scalar_strict_types =
      ParameterList<bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                    uint32_t, uint64_t, float, double>;

  mod.add_type<Library>("Library");
  mod.add_type<Variable>("Variable");
  mod.add_type<Constraint>("Constraint");

  mod.add_bits<LocalTaskID>("LocalTaskID");
  mod.add_bits<GlobalTaskID>("GlobalTaskID");

  mod.add_bits<legate::mapping::StoreTarget>("StoreTarget");

  mod.set_const("SYSMEM", legate::mapping::StoreTarget::SYSMEM);
  mod.set_const("SOCKETMEM", legate::mapping::StoreTarget::SOCKETMEM);
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  mod.set_const("FBMEM", legate::mapping::StoreTarget::FBMEM);
  mod.set_const("ZCMEM", legate::mapping::StoreTarget::ZCMEM);
#endif

  mod.add_type<Shape>("Shape").constructor<std::vector<std::uint64_t>>();
  mod.add_type<Domain>("Domain");

  // Scalar constructors for cuNumeric-style element types.
  // StrictlyTypedNumber keeps each integer/float as its own Julia method
  // (avoids collapse to Scalar(::Integer) / last-wins UInt64).
  // Bool maps to Union{Bool,CxxBool}; std::complex is mirrored to Complex{T}.
  auto scalar = mod.add_type<Scalar>("Scalar");
  jlcxx::for_each_type<scalar_strict_types>(WrapScalarStrictCtors{scalar});
  scalar.constructor([](std::complex<float> v) { return new Scalar(v); })
      .constructor([](std::complex<double> v) { return new Scalar(v); })
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

  mod.add_type<LogicalStore>("LogicalStoreImpl");
  mod.add_type<LogicalStorePartition>("LogicalStorePartitionImpl");

  mod.method("dim", [](LogicalStore& s) { return s.dim(); });
  mod.method("type", [](LogicalStore& s) { return s.type(); });
  mod.method("reinterpret_as", [](LogicalStore& s, legate::Type t) {
    return s.reinterpret_as(t);
  });
  mod.method("promote",
             [](LogicalStore& s, int32_t extra_dim, size_t dim_size) {
               return s.promote(extra_dim, dim_size);
             });
  mod.method("slice", [](LogicalStore& s, int32_t dim, legate::Slice sl) {
    return s.slice(dim, sl);
  });
  mod.method(
      "get_physical_store",
      [](LogicalStore& s, std::optional<legate::mapping::StoreTarget> target) {
        return s.get_physical_store(target);
      });
  mod.method("equal_storage", [](LogicalStore& s, LogicalStore& other) {
    return s.equal_storage(other);
  });
  mod.method("detach", [](LogicalStore& s) { return s.detach(); });
  mod.method("partition_by_tiling", [](LogicalStore& store,
                                       std::vector<uint64_t> tile_shape) {
    return legate_wrapper::data::partition_by_tiling(store, tile_shape);
  });

  mod.method("partition_by_tiling",
             [](LogicalStore& store, std::vector<uint64_t> tile_shape,
                std::vector<uint64_t> color_shape) {
               return legate_wrapper::data::partition_by_tiling(
                   store, tile_shape, color_shape);
             });
  mod.method("color_shape", [](std::shared_ptr<LogicalStorePartition> p) {
    auto s = p->color_shape();
    std::vector<uint64_t> result(s.begin(), s.end());
    return result;
  });
  mod.method("store", [](std::shared_ptr<LogicalStorePartition> p) {
    return p->store();
  });

  mod.add_type<PhysicalArray>("PhysicalArray")
      .method("dim", &PhysicalArray::dim)
      .method("type", &PhysicalArray::type)
      .method("nullable", &PhysicalArray::nullable)
      .method("data", &PhysicalArray::data);  // returns PhysicalStore

  mod.add_type<LogicalArray>("LogicalArrayImpl")
      .method("dim", &LogicalArray::dim)
      .method("type", &LogicalArray::type)
      .method("nullable", &LogicalArray::nullable)
      .method("data", &LogicalArray::data)  // returns LogicalStore
      .method("get_physical_array",
              &LogicalArray::get_physical_array)  // return PhysicalArray
      .method("unbound", &LogicalArray::unbound)
      .method("shape", [](const LogicalArray& arr) {
        auto s = arr.data().shape();
        std::vector<uint64_t> result;
        for (int i = 0; i < arr.dim(); i++) result.push_back(s[i]);
        return result;
      });

  mod.add_type<AutoTask>("AutoTask")
      .method("add_input", static_cast<Variable (AutoTask::*)(LogicalArray)>(
                               &AutoTask::add_input))
      .method("add_output", static_cast<Variable (AutoTask::*)(LogicalArray)>(
                                &AutoTask::add_output))
      .method("add_scalar", static_cast<void (AutoTask::*)(const Scalar&)>(
                                &AutoTask::add_scalar_arg))
      .method("add_constraint",
              static_cast<void (AutoTask::*)(const Constraint&)>(
                  &AutoTask::add_constraint))
      .method("add_communicator",
              [](AutoTask& t, const std::string& name) {
                t.add_communicator(std::string_view{name});
              })
      .method("get_obj_ptr", [](AutoTask& t) { return static_cast<void*>(&t); })
      .method("find_or_declare_partition",
              static_cast<Variable (AutoTask::*)(const LogicalArray&)>(
                  &AutoTask::find_or_declare_partition))
      .method("declare_partition", static_cast<Variable (AutoTask::*)()>(
                                       &AutoTask::declare_partition))
      .method("broadcast", [](Variable& v) { return legate::broadcast(v); })
      .method("broadcast",
              [](Variable& v, std::vector<uint32_t> axes) {
                return legate::broadcast(
                    v, legate::Span<const uint32_t>{axes.data(), axes.size()});
              })
      .method("provenance",
              [](AutoTask& t) { return std::string{t.provenance()}; });

  mod.add_type<ManualTask>("ManualTask")
      .method("add_input", static_cast<void (ManualTask::*)(LogicalStore)>(
                               &ManualTask::add_input))
      .method("add_output", static_cast<void (ManualTask::*)(LogicalStore)>(
                                &ManualTask::add_output))
      .method("add_input",
              [](ManualTask& t, std::shared_ptr<LogicalStorePartition> p) {
                t.add_input(*p);
              })
      .method("add_output",
              [](ManualTask& t, std::shared_ptr<LogicalStorePartition> p) {
                t.add_output(*p);
              })
      .method("add_scalar", static_cast<void (ManualTask::*)(const Scalar&)>(
                                &ManualTask::add_scalar_arg))
      .method("add_communicator",
              [](ManualTask& t, const std::string& name) {
                t.add_communicator(std::string_view{name});
              })
      .method("get_obj_ptr",
              [](ManualTask& t) { return static_cast<void*>(&t); });

  /* runtime */
  mod.add_type<Runtime>("Runtime");

  mod.method("start_legate", &legate_wrapper::runtime::start_legate);
  mod.method("legate_finish", &legate_wrapper::runtime::legate_finish);
  // Drain GC-enqueued LogicalStore/LogicalArray frees; call on the launch
  // thread.
  mod.method("legate_drain_frees", []() { drain_deferred_frees(); });
  mod.method("get_runtime", &legate_wrapper::runtime::get_runtime);
  mod.method("has_started", &legate_wrapper::runtime::has_started);
  mod.method("has_finished", &legate_wrapper::runtime::has_finished);
  mod.method("runtime_sync", &legate_wrapper::runtime::runtime_sync);
  /* tasking */
  mod.method("align", &legate_wrapper::tasking::align);
  mod.method("domain_from_shape", &legate_wrapper::tasking::domain_from_shape);
  mod.method("create_manual_task",
             &legate_wrapper::tasking::create_manual_task);
  mod.method("create_auto_task", &legate_wrapper::tasking::create_auto_task);
  mod.method("submit_auto_task", &legate_wrapper::tasking::submit_auto_task);
  mod.method("submit_manual_task",
             &legate_wrapper::tasking::submit_manual_task);

  mod.add_type<Scope>("Scope")
      .constructor<std::int32_t>()
      .constructor<std::string>()
      .method("set_priority", &Scope::set_priority)
      .method("set_provenance",
              [](Scope& s, std::string p) { s.set_provenance(std::move(p)); });

  mod.method("scope_priority", []() { return Scope::priority(); });
  mod.method("scope_provenance",
             []() { return std::string{Scope::provenance()}; });
  mod.method("destroy_scope", &legate_wrapper::tasking::destroy_scope);
  mod.method("add_task_provenance",
             &legate_wrapper::tasking::add_task_provenance);

  /* array management */
  mod.method("create_unbound_array",
             &legate_wrapper::data::create_unbound_array);
  mod.method("create_array", &legate_wrapper::data::create_array);
  mod.method("create_unbound_store",
             &legate_wrapper::data::create_unbound_store);
  mod.method("create_store", &legate_wrapper::data::create_store);
  mod.method("store_from_scalar", &legate_wrapper::data::store_from_scalar);
  mod.method("attach_external_store_sysmem_row_major",
             &legate_wrapper::data::attach_external_store_sysmem_row_major);
  mod.method("attach_external_store_sysmem_col_major",
             &legate_wrapper::data::attach_external_store_sysmem_col_major);
  mod.method("attach_external_store_sysmem",
             &legate_wrapper::data::attach_external_store_sysmem);
  mod.method("attach_external_store_fbmem_row_major",
             &legate_wrapper::data::attach_external_store_fbmem_row_major);
  mod.method("attach_external_store_fbmem_col_major",
             &legate_wrapper::data::attach_external_store_fbmem_col_major);
  mod.method("attach_external_store_fbmem",
             &legate_wrapper::data::attach_external_store_fbmem);
  mod.method("_get_ptr", &legate_wrapper::data::get_ptr);
  mod.method("set_gpu_tasking_active",
             &legate_wrapper::data::set_gpu_tasking_active);
  mod.method("make_scalar", &legate_wrapper::data::make_scalar);
  /* type management */
  mod.method("string_to_scalar", &legate_wrapper::data::string_to_scalar);
  /* timing */
  mod.method("time_microseconds", &legate_wrapper::time::time_microseconds);
  mod.method("time_nanoseconds", &legate_wrapper::time::time_nanoseconds);

  /* hdf5 */
  mod.method("_read_h5", &legate_wrapper::hdf5::read_h5);
  mod.method("_write_h5", &legate_wrapper::hdf5::write_h5);
  mod.method("num_procs", &legate_wrapper::runtime::num_procs);
  mod.method("num_gpus", &legate_wrapper::runtime::num_gpus);
  // `block` is required — do not default at the C++ binding layer.
  mod.method("issue_execution_fence",
             &legate_wrapper::runtime::issue_execution_fence);
  mod.method("issue_mapping_fence",
             &legate_wrapper::runtime::issue_mapping_fence);

  wrap_ufi(mod);
}

extern "C" void legate_logical_store_detach(void* store_ptr) {
  reinterpret_cast<legate::LogicalStore*>(store_ptr)->detach();
}

extern "C" void legate_issue_copy(void* dest_ptr, void* src_ptr) {
  auto& dest = *reinterpret_cast<legate::LogicalStore*>(dest_ptr);
  auto& src = *reinterpret_cast<const legate::LogicalStore*>(src_ptr);
  legate::Runtime::get_runtime()->issue_copy(dest, src);
}

extern "C" void legate_issue_execution_fence_blocking() {
  legate::Runtime::get_runtime()->issue_execution_fence(/*block=*/true);
}
