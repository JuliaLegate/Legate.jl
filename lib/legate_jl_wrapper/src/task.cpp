#include "task.h"

#include <legate.h>

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

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

#ifndef JULIA_LEGATE_UFI_EXPORT
#define JULIA_LEGATE_UFI_EXPORT extern "C"
#endif

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
  return rt->create_library(library_name, legate::ResourceConfig{});
}

constexpr int MAX_UFI_SLOTS = 32;
constexpr int MAX_UFI_ARGS = 16;
constexpr int MAX_SCALAR_SIZE = 64;

// TaskRequest — layout must match Julia's TaskRequest in ufi.jl
struct TaskRequestData {
  int is_gpu;        // Offset 0
  uint32_t task_id;  // Offset 4
  void** inputs_ptr; // Offset 8
  void** outputs_ptr; // Offset 16
  void** scalars_ptr; // Offset 24
  int ndim;          // Offset 32
  int64_t dims[REALM_MAX_DIM]; // Offset 40 (Padding ensures 8-byte alignment)
};

static_assert(sizeof(TaskRequestData) == 40 + REALM_MAX_DIM * sizeof(int64_t), "TaskRequestData size must be 64 bytes");

struct UFISlot {
  TaskRequestData request;
  std::mutex mutex;
  std::condition_variable cv;
  std::atomic<bool> task_done{false};
  std::atomic<bool> in_use{false};

  // Fixed-size buffers to avoid dynamic allocations during task dispatch
  void* inputs[MAX_UFI_ARGS];
  void* outputs[MAX_UFI_ARGS];
  void* scalar_ptrs[MAX_UFI_ARGS];
  char scalar_data[MAX_UFI_ARGS][MAX_SCALAR_SIZE];

  UFISlot() {
    task_done.store(false);
    in_use.store(false);
    request.inputs_ptr = inputs;
    request.outputs_ptr = outputs;
    request.scalars_ptr = scalar_ptrs;
  }

  void reset() {
    task_done.store(false);
    in_use.store(false);
    request.ndim = 0;
  }
};

static UFISlot g_ufi_slots[MAX_UFI_SLOTS];
static std::mutex g_slot_mutex;
static std::condition_variable g_slot_cv;
static std::mutex g_queue_mutex;
static std::vector<int> g_pending_queue;

static std::atomic<int> g_active_calls{0};
static std::atomic<int> g_started_count{0};

} // namespace ufi

namespace ufi {

struct ActiveCallGuard {
  ActiveCallGuard() { g_active_calls.fetch_add(1); }
  ~ActiveCallGuard() { g_active_calls.fetch_sub(1); }
};

JULIA_LEGATE_UFI_EXPORT void completion_callback_from_julia(int slot_id) {
  if (slot_id < 0 || slot_id >= MAX_UFI_SLOTS) return;
  DEBUG_PRINT("Completion callback for slot %d\n", slot_id);
  auto& slot = g_ufi_slots[slot_id];
  std::lock_guard<std::mutex> lock(slot.mutex);
  slot.task_done.store(true);
  slot.cv.notify_one();
}

JULIA_LEGATE_UFI_EXPORT int legate_get_max_slots() { return MAX_UFI_SLOTS; }

JULIA_LEGATE_UFI_EXPORT void* legate_get_slot_request_ptr(int slot_id) {
  if (slot_id < 0 || slot_id >= MAX_UFI_SLOTS) return nullptr;
  return static_cast<void*>(&g_ufi_slots[slot_id].request);
}

JULIA_LEGATE_UFI_EXPORT int legate_get_active_call_count() { return g_active_calls.load(); }
JULIA_LEGATE_UFI_EXPORT int legate_get_started_count() { return g_started_count.load(); }

JULIA_LEGATE_UFI_EXPORT int legate_pop_pending_slot_nonblocking() {
  std::lock_guard<std::mutex> lock(g_queue_mutex);
  if (g_pending_queue.empty()) return -1;
  int slot_id = g_pending_queue.front();
  g_pending_queue.erase(g_pending_queue.begin());
  return slot_id;
}

JULIA_LEGATE_UFI_EXPORT int legate_get_active_slot_count() {
  int count = 0;
  for (int i = 0; i < MAX_UFI_SLOTS; ++i) {
    if (g_ufi_slots[i].in_use.load()) count++;
  }
  return count;
}

void initialize_async_system() {
  for (int i = 0; i < MAX_UFI_SLOTS; ++i) g_ufi_slots[i].reset();
}

inline void JuliaTaskInterface(legate::TaskContext context, bool is_gpu) {
  ActiveCallGuard guard;
  if (context.num_scalars() == 0) {
    abort();
  }

  g_started_count.fetch_add(1);
  
  uint32_t task_id = context.scalar(0).value<uint32_t>();
  DEBUG_PRINT("JuliaTaskInterface(task=%u): started=%d, active=%d\n", task_id, g_started_count.load(), g_active_calls.load());

  DEBUG_PRINT("JuliaTaskInterface starting for task %u in slot search...\n", task_id);

  const size_t ni = std::min((size_t)context.num_inputs(), (size_t)MAX_UFI_ARGS);
  const size_t no = std::min((size_t)context.num_outputs(), (size_t)MAX_UFI_ARGS);
  const size_t nst = context.num_scalars();
  const size_t ns = std::min((nst > 1 ? nst - 1 : 0), (size_t)MAX_UFI_ARGS);

  if (context.num_inputs() > MAX_UFI_ARGS || context.num_outputs() > MAX_UFI_ARGS || (nst > 1 && nst - 1 > MAX_UFI_ARGS)) {
    ERROR_PRINT("Task %u: Argument count exceeds MAX_UFI_ARGS (%d). Inputs: %zu, Outputs: %zu, Scalars: %zu\n", 
                task_id, MAX_UFI_ARGS, context.num_inputs(), context.num_outputs(), nst - 1);
    abort();
  }

  DEBUG_PRINT("Task %u: inputs=%zu, outputs=%zu, scalars=%zu\n", task_id, ni, no, ns);

  // Claim a slot early as possible
  int slot_id = -1;
  {
    std::unique_lock<std::mutex> lock(g_slot_mutex);
    g_slot_cv.wait(lock, [&slot_id] {
      for (int i = 0; i < MAX_UFI_SLOTS; ++i) {
        bool expected = false;
        if (g_ufi_slots[i].in_use.compare_exchange_strong(expected, true)) {
          slot_id = i;
          return true;
        }
      }
      return false;
    });
  }
  DEBUG_PRINT("Task %u: Claimed slot %d\n", task_id, slot_id);

  auto& slot = g_ufi_slots[slot_id];
  // This prevents stale notifications from previous runs of the SAME slot.
  {
    std::lock_guard<std::mutex> lock(slot.mutex);
    slot.task_done.store(false);
  }

  slot.request.is_gpu = is_gpu ? 1 : 0;
  slot.request.task_id = task_id;
  slot.request.ndim = 0;

  ufiFunctor functor{&slot.request.ndim, slot.request.dims};

  for (size_t i = 0; i < ni; ++i) {
    auto ps = context.input(i);
    std::uintptr_t p = 0;
    legate::double_dispatch(ps.dim(), ps.type().code(), functor, ufi::AccessMode::READ, p, ps);
    slot.inputs[i] = reinterpret_cast<void*>(p);
  }

  for (size_t i = 0; i < no; ++i) {
    auto ps = context.output(i);
    std::uintptr_t p = 0;
    legate::double_dispatch(ps.dim(), ps.type().code(), functor, ufi::AccessMode::WRITE, p, ps);
    slot.outputs[i] = reinterpret_cast<void*>(p);
  }

  for (size_t i = 0; i < ns; ++i) {
    auto scal = context.scalar(i + 1);
    size_t sz = std::min(scal.size(), (size_t)MAX_SCALAR_SIZE);
    std::memcpy(slot.scalar_data[i], scal.ptr(), sz);
    slot.scalar_ptrs[i] = slot.scalar_data[i];
  }

  slot.task_done.store(false);
  {
    std::lock_guard<std::mutex> qlock(g_queue_mutex);
    g_pending_queue.push_back(slot_id);
  }

  // Wait for completion
  {
    std::unique_lock<std::mutex> lock(slot.mutex);
    DEBUG_PRINT("Task %u: Entering CV wait for slot %d...\n", task_id, slot_id);
    slot.cv.wait_for(lock, std::chrono::seconds(300), [&] { return slot.task_done.load(); });
    
    if (!slot.task_done.load()) {
      fprintf(stderr, "ERROR: Julia task %u TIMED OUT in slot %d after 300s\n", task_id, slot_id);
      abort(); 
    } else {
      DEBUG_PRINT("Task %u done in slot %d\n", task_id, slot_id);
    }
    
    slot.in_use.store(false);
    DEBUG_PRINT("Task %u: Released slot %d\n", task_id, slot_id);
  }

  {
    std::lock_guard<std::mutex> lock(g_slot_mutex);
    g_slot_cv.notify_all();
  }
}

void JuliaCustomTask::cpu_variant(legate::TaskContext context) { JuliaTaskInterface(context, false); }
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
void JuliaCustomGPUTask::gpu_variant(legate::TaskContext context) { JuliaTaskInterface(context, true); }
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
  mod.method("legate_get_max_slots", &ufi::legate_get_max_slots);
  mod.method("legate_get_slot_request_ptr", &ufi::legate_get_slot_request_ptr);
  mod.method("legate_pop_pending_slot_nonblocking", &ufi::legate_pop_pending_slot_nonblocking);
  mod.method("legate_get_active_call_count", &ufi::legate_get_active_call_count);
  mod.method("legate_get_active_slot_count", &ufi::legate_get_active_slot_count);
  mod.method("legate_get_started_count", &ufi::legate_get_started_count);
  mod.set_const("JULIA_CUSTOM_TASK", legate::LocalTaskID{ufi::TaskIDs::JULIA_CUSTOM_TASK});
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  mod.set_const("JULIA_CUSTOM_GPU_TASK", legate::LocalTaskID{ufi::TaskIDs::JULIA_CUSTOM_GPU_TASK});
#endif
}
