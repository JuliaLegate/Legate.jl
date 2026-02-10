#include "task.h"

#include <legate.h>
#include <uv.h>  // For uv_async_send

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <chrono>

#include "types.h"

// #define DEBUG

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
  // leverage default resource config and default mapper
  // TODO have the mapper configurable by users depending on their library
  // workload
  return rt->create_library(library_name, legate::ResourceConfig{});
}

constexpr int MAX_UFI_SLOTS = 8;
static std::atomic<uint64_t> g_task_sequence{0};

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

struct UFISlot {
  TaskRequestData request;
  std::mutex mutex;
  std::condition_variable cv;
  std::atomic<int32_t> work_available{0};
  std::atomic<bool> task_done{false};
  std::atomic<bool> in_use{false};

  UFISlot() {
    work_available.store(0);
    task_done.store(false);
    in_use.store(false);
  }

  void reset() {
    work_available.store(0);
    task_done.store(false);
    inputs.clear();
    outputs.clear();
    scalar_values.clear();
    inputs_types.clear();
    outputs_types.clear();
    scalar_types.clear();
    request.ndim = 0;
  }

  // Keep these alive during task execution
  std::vector<void*> inputs;
  std::vector<void*> outputs;
  std::vector<void*> scalar_values;
  std::vector<int> inputs_types;
  std::vector<int> outputs_types;
  std::vector<int> scalar_types;
};

// Global state
static UFISlot g_ufi_slots[MAX_UFI_SLOTS];
static std::mutex g_slot_mutex;          // Protects slot allocation
static std::mutex g_julia_callback_mutex; // Serializes entry into Julia for thread safety
static std::condition_variable g_slot_cv;
static uv_async_t* g_async_handle = nullptr;
static std::atomic<int> g_active_calls{0};
static std::atomic<int> g_work_sequence{1}; // Starts at 1, avoids 0

struct ActiveCallGuard {
  ActiveCallGuard() { g_active_calls.fetch_add(1); }
  ~ActiveCallGuard() { g_active_calls.fetch_sub(1); }
};

JULIA_LEGATE_UFI_EXPORT void legate_set_async_handle(void* handle) {
  g_async_handle = static_cast<uv_async_t*>(handle);
  DEBUG_PRINT("DEBUG: C++ Async handle set to: %p\n", (void*)g_async_handle);
}

JULIA_LEGATE_UFI_EXPORT int32_t* legate_get_slot_work_available_ptr(int slot_id) {
  if (slot_id < 0 || slot_id >= MAX_UFI_SLOTS) return nullptr;
  return reinterpret_cast<int32_t*>(&g_ufi_slots[slot_id].work_available);
}

JULIA_LEGATE_UFI_EXPORT void completion_callback_from_julia(int slot_id) {
  if (slot_id < 0 || slot_id >= MAX_UFI_SLOTS) return;
  DEBUG_PRINT("C++ completion callback received for slot %d\n", slot_id);
  auto& slot = g_ufi_slots[slot_id];
  std::unique_lock<std::mutex> lock(slot.mutex);
  slot.task_done.store(true);
  slot.cv.notify_one();
}

JULIA_LEGATE_UFI_EXPORT int legate_get_max_slots() { return MAX_UFI_SLOTS; }

JULIA_LEGATE_UFI_EXPORT void* legate_get_slot_request_ptr(int slot_id) {
  if (slot_id < 0 || slot_id >= MAX_UFI_SLOTS) return nullptr;
  return static_cast<void*>(&g_ufi_slots[slot_id].request);
}

JULIA_LEGATE_UFI_EXPORT int legate_get_pending_tasks_count() {
  return g_active_calls.load();
}

void initialize_async_system() {
  for (int i = 0; i < MAX_UFI_SLOTS; ++i) {
    g_ufi_slots[i].reset();
  }
}

inline void JuliaTaskInterface(legate::TaskContext context, bool is_gpu) {
  ActiveCallGuard guard;
  
  // Serialize entry to avoid crashing Julia internal mutexes
  std::unique_lock<std::mutex> callback_lock(g_julia_callback_mutex);

  if (context.num_scalars() == 0) {
    ERROR_PRINT("Task has no scalars! Variant aborted.\n");
    return;
  }
  
  std::int32_t task_id = context.scalar(0).value<std::int32_t>();

  DEBUG_PRINT("JuliaTaskInterface starting for task %d...\n", task_id);
  
  const std::size_t num_inputs = context.num_inputs();
  const std::size_t num_outputs = context.num_outputs();
  
  DEBUG_PRINT("Task %d: inputs=%zu, outputs=%zu\n", task_id, num_inputs, num_outputs);

  std::vector<void*> inputs;
  std::vector<void*> outputs;

  std::vector<int> inputs_types;
  std::vector<int> outputs_types;

  // Scalar 0 is reserved for task ID. User scalars start at 1.
  const std::size_t total_scalars = context.num_scalars();
  const std::size_t num_scalars = (total_scalars > 1) ? total_scalars - 1 : 0;
  
  DEBUG_PRINT("Task %d: total_scalars=%zu, user_scalars=%zu\n", task_id, total_scalars, num_scalars);

  std::vector<void*> scalar_values;
  std::vector<int> scalar_types;

  int ndim = 0;
  int64_t dims[3] = {1, 1, 1};
  ufiFunctor functor{&ndim, dims};

  DEBUG_PRINT("Task %d: Processing inputs...\n", task_id);
  for (std::size_t i = 0; i < num_inputs; ++i) {
    auto ps = context.input(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    std::uintptr_t p = 0;
    legate::double_dispatch(dim, code, functor, ufi::AccessMode::READ, p, ps);
    // Note: p may be null for empty domains, handled by Julia/operator()
    inputs.push_back(reinterpret_cast<void*>(p));
    inputs_types.push_back((int)code);
  }

  DEBUG_PRINT("Task %d: Processing outputs...\n", task_id);
  for (std::size_t i = 0; i < num_outputs; ++i) {
    auto ps = context.output(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    std::uintptr_t p = 0;
    legate::double_dispatch(dim, code, functor, ufi::AccessMode::WRITE, p, ps);
    // Note: p may be null for empty domains, handled by Julia/operator()
    outputs.push_back(reinterpret_cast<void*>(p));
    outputs_types.push_back((int)code);
  }

  DEBUG_PRINT("Task %d: Processing user scalars...\n", task_id);
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

  DEBUG_PRINT("Task %d: Finding an available slot...\n", task_id);

  // find an available slot using CV to avoid spin-waiting
  int slot_id = -1;
  {
    std::unique_lock<std::mutex> lock(g_slot_mutex);
    g_slot_cv.wait(lock, [&slot_id, task_id] {
      for (int i = 0; i < MAX_UFI_SLOTS; ++i) {
        if (!g_ufi_slots[i].in_use.load()) {
          g_ufi_slots[i].in_use.store(true);
          slot_id = i;
          return true;
        }
      }
      DEBUG_PRINT("Task %d: No slots available, waiting on CV...\n", task_id);
      return false;
    });
  }
  DEBUG_PRINT("Task %d: Claimed slot %d\n", task_id, slot_id);

  // Instead of calling Julia directly, we:
  //   1. Fill the shared TaskRequest structure
  //   2. Call uv_async_send to wake up Julia's async worker
  //   3. Wait for Julia to signal completion

  try {
    {
      auto& slot = g_ufi_slots[slot_id];
      std::unique_lock<std::mutex> lock(slot.mutex);

      // copy vectors to slot to keep them alive
      slot.inputs = std::move(inputs);
      slot.outputs = std::move(outputs);
      slot.scalar_values = std::move(scalar_values);
      slot.inputs_types = std::move(inputs_types);
      slot.outputs_types = std::move(outputs_types);
      slot.scalar_types = std::move(scalar_types);

      // 1. Fill the shared TaskRequest structure
      slot.request.is_gpu = is_gpu ? 1 : 0;
      slot.request.task_id = task_id;
      slot.request.inputs_ptr = num_inputs > 0 ? slot.inputs.data() : nullptr;
      slot.request.outputs_ptr = num_outputs > 0 ? slot.outputs.data() : nullptr;
      slot.request.scalars_ptr =
          num_scalars > 0 ? slot.scalar_values.data() : nullptr;
      slot.request.inputs_types =
          num_inputs > 0 ? slot.inputs_types.data() : nullptr;
      slot.request.outputs_types =
          num_outputs > 0 ? slot.outputs_types.data() : nullptr;
      slot.request.scalar_types =
          num_scalars > 0 ? slot.scalar_types.data() : nullptr;
      slot.request.num_inputs = num_inputs;
      slot.request.num_outputs = num_outputs;
      slot.request.num_scalars = num_scalars;
      slot.request.ndim = ndim;
      for (int i = 0; i < 3; ++i) slot.request.dims[i] = dims[i];

      // reset completion flag
      slot.task_done.store(false);
      
      // signal availability with unique sequence ID
      int seq = g_work_sequence.fetch_add(1);
      if (seq == 0) seq = g_work_sequence.fetch_add(1); // Skip 0
      slot.work_available.store(seq);
      
      DEBUG_PRINT(
          "[SEQ %d] Task %d ready in slot %d (work_seq=%d) addr=%p\n",
          (int)g_task_sequence++, task_id, slot_id, seq, (void*)&slot.work_available);
      
      // 2. Call uv_async_send to wake up Julia's async worker
      if (g_async_handle != nullptr) {
          uv_async_send(g_async_handle);
      }
    } 

    // release g_julia_callback_mutex BEFORE waiting for Julia
    callback_lock.unlock();

    // 3. Wait for Julia to signal completion
    {
      auto& slot = g_ufi_slots[slot_id];
      std::unique_lock<std::mutex> lock(slot.mutex);
      DEBUG_PRINT("Task %d: Entering CV wait for slot %d...\n", task_id, slot_id);
      auto timeout = std::chrono::seconds(30);
      bool success = slot.cv.wait_for(lock, timeout, [&] { return slot.task_done.load(); });

      if (!success) {
        fprintf(stderr, "ERROR: Julia task %d TIMED OUT in slot %d after 30s\n", task_id, slot_id);
      } else {
        DEBUG_PRINT("Task %d done in slot %d, proceeding to cleanup\n", task_id, slot_id);
      }

      // free the memory we allocated for the scalar values
      for (void* ptr : slot.scalar_values) {
        delete[] static_cast<char*>(ptr);
      }

      // reset slot data before releasing
      slot.request.inputs_ptr = nullptr;
      slot.request.outputs_ptr = nullptr;
      slot.request.scalars_ptr = nullptr;
      slot.inputs.clear();
      slot.outputs.clear();
      slot.scalar_values.clear();
      slot.inputs_types.clear();
      slot.outputs_types.clear();
      slot.scalar_types.clear();

      // release slot state
      DEBUG_PRINT("Releasing slot %d...\n", slot_id);
      slot.work_available.store(0);
      slot.in_use.store(false);
      lock.unlock();

      // notify threads waiting for an available slot (must use g_slot_mutex)
      {
        std::lock_guard<std::mutex> global_lock(g_slot_mutex);
        g_slot_cv.notify_all();
      }
    }
  } catch (const std::exception& e) {
    ERROR_PRINT("C++ exception [ERR_UFI_1] in JuliaTaskInterface for task %d: %s\n",
                task_id, e.what());
    abort();
  } catch (...) {
    ERROR_PRINT("Unknown C++ exception [ERR_UFI_2] in JuliaTaskInterface for task %d\n",
                task_id);
    abort();
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
  mod.method("legate_set_async_handle", &ufi::legate_set_async_handle);
  mod.set_const("JULIA_CUSTOM_TASK",
                legate::LocalTaskID{ufi::TaskIDs::JULIA_CUSTOM_TASK});
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  mod.set_const("JULIA_CUSTOM_GPU_TASK",
                legate::LocalTaskID{ufi::TaskIDs::JULIA_CUSTOM_GPU_TASK});
#endif
}
