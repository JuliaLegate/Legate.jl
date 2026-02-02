#include "task.h"

#include <uv.h>  // For uv_async_send

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "legate.h"
#include "types.h"

//#define DEBUG
#ifdef DEBUG
#define DEBUG_PRINT(...)                  \
  fprintf(stderr, "DEBUG: " __VA_ARGS__); \
  fflush(stderr);
#else
#define DEBUG_PRINT(...) ;
#endif

#define ERROR_PRINT(...)                  \
  fprintf(stderr, "ERROR: " __VA_ARGS__); \
  fflush(stderr);

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
  int is_gpu;  // Use int to match Julia Cint for alignment
  uint32_t task_id;
  void** inputs_ptr;
  void** outputs_ptr;
  void** scalars_ptr;
  int* inputs_types;
  int* outputs_types;
  int* scalar_types;
  size_t num_inputs;
  size_t num_outputs;
  size_t num_scalars;
  int ndim;
  int64_t dims[3];  // Up to 3 dimensions
};

// Global state
static TaskRequestData* g_request_ptr = nullptr;
static std::mutex g_issue_mutex;       // Serializes access to request buffer
static std::mutex g_completion_mutex;  // Protects completion CV
static std::condition_variable g_completion_cv;
static std::atomic<bool> g_task_done{false};
static std::atomic<bool> g_work_available{false};  // For polling

extern "C" int legate_poll_work() { return g_work_available.load() ? 1 : 0; }

extern "C" void completion_callback_from_julia() {
  std::unique_lock<std::mutex> lock(g_completion_mutex);
  g_task_done.store(true);
  g_work_available.store(false);  // Task completed
  g_completion_cv.notify_one();
}

// Initialize async infrastructure - called from Julia
void initialize_async_system(void* request_ptr) {
  g_request_ptr = static_cast<TaskRequestData*>(request_ptr);
  DEBUG_PRINT("Async system initialized: request=%p\n", g_request_ptr);
}

inline void JuliaTaskInterface(legate::TaskContext context, bool is_gpu) {
  std::int32_t task_id = context.scalar(0).value<std::int32_t>();

  const std::size_t num_inputs = context.num_inputs();
  const std::size_t num_outputs = context.num_outputs();

  std::vector<void*> inputs;
  std::vector<void*> outputs;

  std::vector<int> inputs_types;
  std::vector<int> outputs_types;

  // Scalar 0 is reserved for task ID. User scalars start at 1.
  const std::size_t total_scalars = context.num_scalars();
  const std::size_t num_scalars = (total_scalars > 1) ? total_scalars - 1 : 0;

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
    inputs_types.push_back((int)code);
  }

  for (std::size_t i = 0; i < num_outputs; ++i) {
    auto ps = context.output(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    std::uintptr_t p;
    legate::double_dispatch(dim, code, functor, ufi::AccessMode::WRITE, p, ps);
    assert(p != 0);
    outputs.push_back(reinterpret_cast<void*>(p));
    outputs_types.push_back((int)code);
  }

  // Process User Scalars
  for (std::size_t i = 0; i < num_scalars; ++i) {
    // Offset by 1 because scalar 0 is reserved for task_id
    auto scal = context.scalar(i + 1);
    auto code = scal.type().code();
    auto size = scal.size();
    const void* p = scal.ptr();

    auto* val_ptr = new char[size];
    if (p != nullptr) {
      std::memcpy(val_ptr, p, size);
    } else {
      std::memset(val_ptr, 0, size);
    }

    scalar_values.push_back(val_ptr);
    scalar_types.push_back((int)code);
  }

  // Instead of calling Julia directly, we:
  //   1. Fill the shared TaskRequest structure
  //   2. Call uv_async_send to wake up Julia's async worker
  //   3. Wait for Julia to signal completion

  DEBUG_PRINT("Preparing async request for task %d...\n", task_id);
  {
    // Hold issue lock for the ENTIRE duration of the transaction
    // to prevent other threads from overwriting the shared request buffer.
    std::lock_guard<std::mutex> issue_lock(g_issue_mutex);

    std::unique_lock<std::mutex> lock(g_completion_mutex);

    if (!g_request_ptr) {
      ERROR_PRINT("g_request_ptr is null in JuliaTaskInterface!\n");
      return;
    }

    // Fill the shared request structure (Julia will read this)
    g_request_ptr->is_gpu = is_gpu ? 1 : 0;
    g_request_ptr->task_id = task_id;
    // we don't have to worry about the lifetime of data as this function will
    // block until Julia is done with the task.
    g_request_ptr->inputs_ptr = inputs.data();
    g_request_ptr->outputs_ptr = outputs.data();
    g_request_ptr->scalars_ptr = scalar_values.data();
    g_request_ptr->inputs_types = inputs_types.data();
    g_request_ptr->outputs_types = outputs_types.data();
    g_request_ptr->scalar_types = scalar_types.data();
    g_request_ptr->num_inputs = num_inputs;
    g_request_ptr->num_outputs = num_outputs;
    g_request_ptr->num_scalars = num_scalars;
    g_request_ptr->ndim = ndim;
    for (int i = 0; i < 3; ++i) g_request_ptr->dims[i] = dims[i];

    // Reset completion flag
    g_task_done.store(false);
    g_work_available.store(true);  // Signal Julia to wake up

    DEBUG_PRINT("Signaling Julia for task %d...\n", task_id);
    DEBUG_PRINT("Waiting for Julia to complete task %d...\n", task_id);

    // Wait for Julia to signal completion
    g_completion_cv.wait(lock, [] { return g_task_done.load(); });
  }

  DEBUG_PRINT("Julia task %d completed!\n", task_id);

  // Free the memory we allocated for the scalar values
  for (void* ptr : scalar_values) {
    delete[] static_cast<char*>(ptr);
  }
}

/* Why not make it JuliaCustomTask::cpu_variant and JuliaCustomTask::gpu_variant
   you may ask? In Legate, a gpu_variant will provide the GPU context, and a
   cpu_variant will provide the CPU context. We need to ensure that Legate will
   place things on the CPU for our CPU tasking and GPU for our GPU tasking. The
   pointers returned by the cpu_variant and gpu_variant will be different. We
   need to pass the pointers to JuliaTaskInterface to send to Julia.
*/
/*static*/ void JuliaCustomTask::cpu_variant(legate::TaskContext context) {
  JuliaTaskInterface(context, false);
}
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
/*static*/ void JuliaCustomGPUTask::gpu_variant(legate::TaskContext context) {
  JuliaTaskInterface(context, true);
}
#endif

void ufi_interface_register(legate::Library& library) {
  ufi::JuliaCustomTask::register_variants(library);
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  ufi::JuliaCustomGPUTask::register_variants(library);
#endif
}

}  // namespace ufi

void wrap_ufi(jlcxx::Module& mod) {
  mod.method("_ufi_interface_register", &ufi::ufi_interface_register);
  mod.method("_create_library", &ufi::create_library);
  mod.method("_initialize_async_system", &ufi::initialize_async_system);
  mod.set_const("JULIA_CUSTOM_TASK",
                legate::LocalTaskID{ufi::TaskIDs::JULIA_CUSTOM_TASK});
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  mod.set_const("JULIA_CUSTOM_GPU_TASK",
                legate::LocalTaskID{ufi::TaskIDs::JULIA_CUSTOM_GPU_TASK});
#endif
}
