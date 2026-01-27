#include "task.h"

#include <dlfcn.h>
#include <julia.h>
#include <uv.h>  // For uv_async_send

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>
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

#define UFI(MODE, ACCESSOR_CALL)                                              \
  template <                                                                  \
      typename T, int D,                                                      \
      typename std::enable_if<(D >= 1 && D <= REALM_MAX_DIM), int>::type = 0> \
  void ufi_##MODE(std::uintptr_t& p, const legate::PhysicalArray& rf) {       \
    auto shp = rf.shape<D>();                                                 \
    auto acc = rf.data().ACCESSOR_CALL<T, D>();                               \
    p = reinterpret_cast<std::uintptr_t>(                                     \
        static_cast<const void*>(acc.ptr(Realm::Point<D>(shp.lo))));          \
  }

UFI(read, read_accessor);
UFI(write, write_accessor);

struct ufiFunctor {
  template <legate::Type::Code CODE, int DIM>
  void operator()(AccessMode mode, std::uintptr_t& p,
                  const legate::PhysicalArray& arr) {
    using CppT = typename legate_util::code_to_cxx<CODE>::type;
    if (mode == AccessMode::READ)
      ufi_read<CppT, DIM>(p, arr);
    else
      ufi_write<CppT, DIM>(p, arr);
  }
};

inline legate::Library create_library(legate::Runtime* rt,
                                      std::string library_name) {
  // leverage default resource config and default mapper
  // TODO have the mapper configurable by users depending on their library
  // workload
  return rt->create_library(library_name, legate::ResourceConfig{});
}

// using julia_task_fn_t = void (*)(void** inputs, void** outputs, int64_t n);
using julia_task_fn_t = void (*)(void* task_ptr, void** inputs, void** outputs,
                                 int64_t n);

struct TaskEntry {
  julia_task_fn_t cpu_fn;
};

static std::unordered_map<uint32_t, TaskEntry> task_table;

// TaskRequest struct  - matches Julia's TaskRequest mutable struct
struct TaskRequestData {
  uint32_t task_id;  // Task ID instead of pointer
  void** inputs_ptr;
  void** outputs_ptr;
  int64_t n;
  size_t num_inputs;
  size_t num_outputs;
};

// Global state
static uv_async_t* g_async_handle = nullptr;
static TaskRequestData* g_request_ptr = nullptr;
static std::mutex g_completion_mutex;
static std::condition_variable g_completion_cv;
static std::atomic<bool> g_task_done{false};

// Completion callback that Julia will call
extern "C" void completion_callback_from_julia() {
  std::unique_lock<std::mutex> lock(g_completion_mutex);
  g_task_done.store(true);
  g_completion_cv.notify_one();
}

// Initialize async infrastructure - called from Julia
void initialize_async_system(void* async_handle_ptr, void* request_ptr) {
  g_async_handle = static_cast<uv_async_t*>(async_handle_ptr);
  g_request_ptr = static_cast<TaskRequestData*>(request_ptr);
  std::fprintf(stderr, "Async system initialized: handle=%p, request=%p\n",
               g_async_handle, g_request_ptr);
}

void register_julia_task(uint32_t task_id, void* fn) {
  auto cast = reinterpret_cast<julia_task_fn_t>(fn);  // just to verify the type
  auto [it, inserted] = task_table.emplace(task_id, TaskEntry{cast});
  assert(inserted && "task_id already registered");
}

/*static*/ void JuliaCustomTask::cpu_variant(legate::TaskContext context) {
  // Adopt the current thread into Julia
  // We need to do this because Legate worker threads are not created by Julia
  // and thus are unknown to the Julia runtime by default. Without adoption,
  // interactions with the Julia runtime (like allocations or GC) will segfault.
  thread_local static bool is_adopted = false;
  if (!is_adopted) {
    jl_adopt_thread();
    is_adopted = true;
  }

  jl_task_t* ct = jl_get_current_task();

  std::int32_t task_id = context.scalar(0).value<std::int32_t>();
  std::uintptr_t task_ptr = context.scalar(1).value<std::uintptr_t>();

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
    assert(p != 0);
    std::fprintf(stderr, "p = 0x%lx\n", (unsigned long)p);
    inputs.push_back(reinterpret_cast<void*>(p));
  }

  for (std::size_t i = 0; i < num_outputs; ++i) {
    auto ps = context.output(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    // store the raw pointer
    std::uintptr_t p;
    legate::double_dispatch(dim, code, ufiFunctor{}, ufi::AccessMode::WRITE, p,
                            ps);
    assert(p != 0);
    std::fprintf(stderr, "p = 0x%lx\n", (unsigned long)p);
    outputs.push_back(reinterpret_cast<void*>(p));
  }

  const int64_t n = 100;

  // Heap allocate arguments to ensure they are accessible to Julia
  // accessing C++ stack memory from the adopted thread can be problematic
  void** inputs_ptr = (void**)malloc(sizeof(void*) * num_inputs);
  void** outputs_ptr = (void**)malloc(sizeof(void*) * num_outputs);

  if (!inputs.empty()) {
    std::memcpy(inputs_ptr, inputs.data(), sizeof(void*) * num_inputs);
  }
  if (!outputs.empty()) {
    std::memcpy(outputs_ptr, outputs.data(), sizeof(void*) * num_outputs);
  }

  // Instead of calling Julia directly, we:
  //   1. Fill the shared TaskRequest structure
  //   2. Call uv_async_send to wake up Julia's async worker
  //   3. Wait for Julia to signal completion

  std::fprintf(stderr, "Preparing async request...\n");
  fflush(stderr);

  // Fill the shared request structure (Julia will read this)
  g_request_ptr->task_id = task_id;  // Pass task ID instead of pointer
  g_request_ptr->inputs_ptr = inputs_ptr;
  g_request_ptr->outputs_ptr = outputs_ptr;
  g_request_ptr->n = n;
  g_request_ptr->num_inputs = num_inputs;
  g_request_ptr->num_outputs = num_outputs;

  // Reset completion flag
  g_task_done.store(false);

  std::fprintf(stderr, "Signaling Julia via uv_async_send...\n");
  fflush(stderr);

  // Signal Julia's event loop - thread-safe
  int result = uv_async_send(g_async_handle);
  if (result != 0) {
    std::fprintf(stderr, "ERROR: uv_async_send failed with code %d\n", result);
  }

  std::fprintf(stderr, "Waiting for Julia to complete task...\n");
  fflush(stderr);

  // Wait for Julia to signal completion
  {
    std::unique_lock<std::mutex> lock(g_completion_mutex);
    g_completion_cv.wait(lock, [] { return g_task_done.load(); });
  }

  std::fprintf(stderr, "Julia task completed!\n");
  fflush(stderr);

  free(inputs_ptr);
  free(outputs_ptr);
}

void ufi_interface_register(legate::Library& library) {
  ufi::JuliaCustomTask::register_variants(library);
}

}  // namespace ufi

void wrap_ufi(jlcxx::Module& mod) {
  mod.method("ufi_interface_register", &ufi::ufi_interface_register);
  mod.method("register_julia_task", &ufi::register_julia_task);
  mod.method("create_library", &ufi::create_library);
  mod.method("initialize_async_system", &ufi::initialize_async_system);
  mod.set_const("JULIA_CUSTOM_TASK",
                legate::LocalTaskID{ufi::TaskIDs::JULIA_CUSTOM_TASK});
}
