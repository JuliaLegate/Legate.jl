#include "task.h"

#include <julia.h>
#include <uv.h>  // For uv_async_send

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
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
  int* ndim_ptr = nullptr;
  int64_t* dims_ptr = nullptr;

  ufiFunctor() = default;
  ufiFunctor(int* ndim, int64_t* dims) : ndim_ptr(ndim), dims_ptr(dims) {}

  template <legate::Type::Code CODE, int DIM>
  void operator()(ufi::AccessMode mode, std::uintptr_t& p,
                  const legate::PhysicalArray& rf) {
    if (ndim_ptr && *ndim_ptr == 0) {
      *ndim_ptr = DIM;
      auto shp = rf.shape<DIM>();
      for (int i = 0; i < DIM && i < 3; ++i) {
        dims_ptr[i] = shp.hi[i] - shp.lo[i] + 1;
      }
    }

    using CppT = typename legate_util::code_to_cxx<CODE>::type;
    if (mode == ufi::AccessMode::READ)
      ufi::ufi_read<CppT, DIM>(p, rf);
    else
      ufi::ufi_write<CppT, DIM>(p, rf);
  }
};

inline legate::Library create_library(legate::Runtime* rt,
                                      std::string library_name) {
  // leverage default resource config and default mapper
  // TODO have the mapper configurable by users depending on their library
  // workload
  return rt->create_library(library_name, legate::ResourceConfig{});
}

// TaskRequest struct  - matches Julia's TaskRequest mutable struct
struct TaskRequestData {
  uint32_t task_id;
  void** inputs_ptr;
  void** outputs_ptr;
  void** scalars_ptr;
  int* scalar_types;
  size_t num_inputs;
  size_t num_outputs;
  size_t num_scalars;
  int ndim;
  int64_t dims[3];  // Up to 3 dimensions
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

  // Scalars 0 and 1 are reserved (ID and Ptr). User scalars start at 2.
  const std::size_t total_scalars = context.num_scalars();
  const std::size_t num_scalars = (total_scalars > 2) ? total_scalars - 2 : 0;

  std::vector<void*> scalar_values;
  std::vector<int> scalar_types;

  int ndim = 0;
  int64_t dims[3] = {1, 1, 1};
  ufiFunctor functor{&ndim, dims};

  for (std::size_t i = 0; i < num_inputs; ++i) {
    auto ps = context.input(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    std::uintptr_t p;
    legate::double_dispatch(dim, code, functor, ufi::AccessMode::READ, p, ps);
    assert(p != 0);
    inputs.push_back(reinterpret_cast<void*>(p));
  }

  for (std::size_t i = 0; i < num_outputs; ++i) {
    auto ps = context.output(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    std::uintptr_t p;
    legate::double_dispatch(dim, code, functor, ufi::AccessMode::WRITE, p, ps);
    assert(p != 0);
    outputs.push_back(reinterpret_cast<void*>(p));
  }

  // Process User Scalars
  for (std::size_t i = 0; i < num_scalars; ++i) {
    // Offset by 2 because 0,1 are reserved (task_id and result_ptr)
    auto scal = context.scalar(i + 2);
    auto code = scal.type().code();
    auto size = scal.size();
    const void* p = scal.ptr();

    void* val_ptr = malloc(size);
    if (p != nullptr) {
      std::memcpy(val_ptr, p, size);
    } else {
      std::memset(val_ptr, 0, size);
    }

    scalar_values.push_back(val_ptr);
    scalar_types.push_back((int)code);
  }

  // Heap allocate arguments to ensure they are accessible to Julia
  // accessing C++ stack memory from the adopted thread can be problematic
  void** inputs_ptr = (void**)malloc(sizeof(void*) * num_inputs);
  void** outputs_ptr = (void**)malloc(sizeof(void*) * num_outputs);

  // Allocate scalar arrays
  void** scalars_ptr = nullptr;
  int* scalar_types_ptr = nullptr;
  if (num_scalars > 0) {
    scalars_ptr = (void**)malloc(sizeof(void*) * num_scalars);
    scalar_types_ptr = (int*)malloc(sizeof(int) * num_scalars);
  }

  if (!inputs.empty()) {
    std::memcpy(inputs_ptr, inputs.data(), sizeof(void*) * num_inputs);
  }
  if (!outputs.empty()) {
    std::memcpy(outputs_ptr, outputs.data(), sizeof(void*) * num_outputs);
  }
  if (num_scalars > 0) {
    std::memcpy(scalars_ptr, scalar_values.data(), sizeof(void*) * num_scalars);
    std::memcpy(scalar_types_ptr, scalar_types.data(),
                sizeof(int) * num_scalars);
  }

  // Instead of calling Julia directly, we:
  //   1. Fill the shared TaskRequest structure
  //   2. Call uv_async_send to wake up Julia's async worker
  //   3. Wait for Julia to signal completion

  std::fprintf(stderr, "Preparing async request...\n");
  fflush(stderr);

  // Fill the shared request structure (Julia will read this)
  g_request_ptr->task_id = task_id;
  g_request_ptr->inputs_ptr = inputs_ptr;
  g_request_ptr->outputs_ptr = outputs_ptr;
  g_request_ptr->scalars_ptr = scalars_ptr;
  g_request_ptr->scalar_types = scalar_types_ptr;
  g_request_ptr->num_inputs = num_inputs;
  g_request_ptr->num_outputs = num_outputs;
  g_request_ptr->num_scalars = num_scalars;
  g_request_ptr->ndim = ndim;
  for (int i = 0; i < 3; ++i) g_request_ptr->dims[i] = dims[i];

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

  if (scalars_ptr) {
    for (size_t i = 0; i < num_scalars; ++i) {
      free(scalars_ptr[i]);
    }
    free(scalars_ptr);
    free(scalar_types_ptr);
  }
}

void ufi_interface_register(legate::Library& library) {
  ufi::JuliaCustomTask::register_variants(library);
}

}  // namespace ufi

void wrap_ufi(jlcxx::Module& mod) {
  mod.method("ufi_interface_register", &ufi::ufi_interface_register);
  mod.method("create_library", &ufi::create_library);
  mod.method("initialize_async_system", &ufi::initialize_async_system);
  mod.set_const("JULIA_CUSTOM_TASK",
                legate::LocalTaskID{ufi::TaskIDs::JULIA_CUSTOM_TASK});
}
