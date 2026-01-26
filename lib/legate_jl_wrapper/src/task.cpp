#include "task.h"

#include <dlfcn.h>

#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "legate.h"
#include "types.h"

namespace ufi {

enum class AccessMode {
  READ,
  WRITE,
};

#define BOB(MODE, ACCESSOR_CALL)                                              \
  template <                                                                  \
      typename T, int D,                                                      \
      typename std::enable_if<(D >= 1 && D <= REALM_MAX_DIM), int>::type = 0> \
  void bob_##MODE(std::uintptr_t& p, const legate::PhysicalArray& rf) {       \
    auto shp = rf.shape<D>();                                                 \
    auto acc = rf.data().ACCESSOR_CALL<T, D>();                               \
    p = reinterpret_cast<std::uintptr_t>(/*.lo to ensure multiple GPU         \
                                            support*/                         \
                                         static_cast<const void*>(acc.ptr(    \
                                             Realm::Point<D>(shp.lo))));      \
  }

BOB(read, read_accessor);    // cuda_device_array_arg_read
BOB(write, write_accessor);  // cuda_device_array_arg_write

struct ufiFunctor {
  template <legate::Type::Code CODE, int DIM>
  void operator()(AccessMode mode, std::uintptr_t& p,
                  const legate::PhysicalArray& arr) {
    using CppT = typename legate_util::code_to_cxx<CODE>::type;
    if (mode == AccessMode::READ)
      bob_read<CppT, DIM>(p, arr);
    else
      bob_write<CppT, DIM>(p, arr);
  }
};

inline legate::Library create_library(legate::Runtime* rt,
                                      std::string library_name) {
  // leverage default resource config and default mapper
  // TODO have the mapper configurable by users depending on their library
  // workload
  return rt->create_library(library_name, legate::ResourceConfig{});
}

using julia_task_fn_t = void (*)(void** inputs, void** outputs, int64_t n);

struct TaskEntry {
  julia_task_fn_t cpu_fn;
};

static std::unordered_map<uint32_t, TaskEntry> task_table;

void register_julia_task(uint32_t task_id, void* fn) {
  auto cast = reinterpret_cast<julia_task_fn_t>(fn);  // just to verify the type
  auto [it, inserted] = task_table.emplace(task_id, TaskEntry{cast});
  assert(inserted && "task_id already registered");
}

/*static*/ void JuliaCustomTask::cpu_variant(legate::TaskContext context) {
  std::int32_t task_id = context.scalar(0).value<std::int32_t>();
  auto it = task_table.find(task_id);
  assert(it != task_table.end());
  julia_task_fn_t fn = reinterpret_cast<julia_task_fn_t>(it->second.cpu_fn);

  const std::size_t num_inputs = context.num_inputs();
  const std::size_t num_outputs = context.num_outputs();

  std::vector<void*> inputs;
  std::vector<void*> outputs;

  for (std::size_t i = 0; i < num_inputs; ++i) {
    auto ps = context.input(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    // store the raw pointer
    std::uintptr_t p;
    legate::double_dispatch(dim, code, ufiFunctor{}, ufi::AccessMode::READ, p,
                            ps);
    inputs[i] = reinterpret_cast<void*>(p);
  }

  for (std::size_t i = 0; i < num_outputs; ++i) {
    auto ps = context.output(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    // store the raw pointer
    std::uintptr_t p;
    legate::double_dispatch(dim, code, ufiFunctor{}, ufi::AccessMode::WRITE, p,
                            ps);
    outputs[i] = reinterpret_cast<void*>(p);
  }

  const int64_t n = 100;

  fn(inputs.data(), outputs.data(), n);
}

void ufi_interface_register(legate::Library& library) {
  ufi::JuliaCustomTask::register_variants(library);
}

}  // namespace ufi

void wrap_ufi(jlcxx::Module& mod) {
  mod.method("ufi_interface_register", &ufi::ufi_interface_register);
  mod.method("register_julia_task", &ufi::register_julia_task);
  mod.method("create_library", &ufi::create_library);
  mod.set_const("JULIA_CUSTOM_TASK",
                legate::LocalTaskID{ufi::TaskIDs::JULIA_CUSTOM_TASK});
}
