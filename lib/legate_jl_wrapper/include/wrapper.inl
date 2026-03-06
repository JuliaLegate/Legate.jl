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

#include <deps/legion.h>
#include <legate.h>
#include <legate/mapping/machine.h>
#include <legate/runtime/runtime.h>
#include <legate/timing/timing.h>

using namespace legate;
/**
 * @brief These functions can be invoked from Julia.
 *
 * This namespace wraps Legate C++ functions exposed to Julia
 * via CxxWrap.
 */
namespace legate_wrapper {
namespace runtime {

/**
 * @ingroup legate_wrapper
 * @brief Start the Legate runtime.
 */
inline void start_legate() { legate::start(); }

/**
 * @ingroup legate_wrapper
 * @brief Finalize the Legate runtime.
 */
inline int32_t legate_finish() { return legate::finish(); }

/**
 * @ingroup legate_wrapper
 * @brief Return the current Legate runtime instance.
 */
inline Runtime* get_runtime() { return Runtime::get_runtime(); }

/**
 * @ingroup legate_wrapper
 * @brief Check whether the Legate runtime has started.
 */
inline bool has_started() { return legate::has_started(); }

/**
 * @ingroup legate_wrapper
 * @brief Check whether the Legate runtime has finished.
 */
inline bool has_finished() { return legate::has_finished(); }

/**
 * @ingroup legate_wrapper
 * @brief Issue an execution fence.
 *
 * @param block Whether to block until the fence is reached.
 */
inline void issue_execution_fence(bool block) {
  Runtime::get_runtime()->issue_execution_fence(block);
}

}  // namespace runtime

namespace tasking {
/**
 * @ingroup legate_wrapper
 * @brief Align two variables.
 *
 * Returns a new variable representing the alignment of `a` and `b`.
 */
inline Constraint align(const Variable& a, const Variable& b) {
  return legate::align(a, b);
}

/**
 * @ingroup legate_wrapper
 * @brief Create an auto task in the runtime.
 *
 * @param rt Pointer to the Runtime instance.
 * @param lib The Library to use for the task.
 * @param id LocalTaskID for the new task.
 * @return An AutoTask instance.
 */
inline AutoTask create_auto_task(Runtime* rt, Library lib, LocalTaskID id) {
  return rt->create_task(lib, id);
}

/**
 * @ingroup legate_wrapper
 * @brief Create an manual task in the runtime.
 *
 * @param rt Pointer to the Runtime instance.
 * @param lib The Library to use for the task.
 * @param id LocalTaskID for the new task.
 * @param domain The Domain for the manual task.
 * @return A ManualTask instance.
 */
inline ManualTask create_manual_task(Runtime* rt, Library lib, LocalTaskID id,
                                     const Domain& domain) {
  return rt->create_task(lib, id, domain);
}

/**
 * @ingroup legate_wrapper
 * @brief Create a Domain from a Shape.
 *
 * @param shape The shape.
 * @return A Domain instance.
 */
inline Domain domain_from_shape(const Shape& shape) {
  if (shape.volume() == 0) return Domain::NO_DOMAIN;
  switch (shape.ndim()) {
    case 1: {
      Legion::Point<1> lo = Legion::Point<1>::ZEROES();
      Legion::Point<1> hi;
      hi[0] = shape[0] - 1;
      return Domain(Legion::Rect<1>(lo, hi));
    }
    case 2: {
      Legion::Point<2> lo = Legion::Point<2>::ZEROES();
      Legion::Point<2> hi;
      hi[0] = shape[0] - 1;
      hi[1] = shape[1] - 1;
      return Domain(Legion::Rect<2>(lo, hi));
    }
    case 3: {
      Legion::Point<3> lo = Legion::Point<3>::ZEROES();
      Legion::Point<3> hi;
      hi[0] = shape[0] - 1;
      hi[1] = shape[1] - 1;
      hi[2] = shape[2] - 1;
      return Domain(Legion::Rect<3>(lo, hi));
    }
    default:
      return Domain::NO_DOMAIN;
  }
}

/**
 * @ingroup legate_wrapper
 * @brief Submit an auto task to the runtime.
 *
 * @param rt Pointer to the Runtime instance.
 * @param task The AutoTask to submit.
 */
inline auto submit_auto_task(Runtime* rt, AutoTask& task) {
  return rt->submit(std::move(task));
}

/**
 * @ingroup legate_wrapper
 * @brief Submit a manual task to the runtime.
 *
 * @param rt Pointer to the Runtime instance.
 * @param task The ManualTask to submit.
 */
inline auto submit_manual_task(Runtime* rt, ManualTask& task) {
  return rt->submit(std::move(task));
}

}  // namespace tasking

namespace data {
/**
 * @ingroup legate_wrapper
 * @brief Convert a string to a Scalar.
 *
 * @param str The string to convert.
 */
inline Scalar string_to_scalar(std::string str) { return Scalar(str); }

/**
 * @ingroup legate_wrapper
 * @brief Create a Scalar from a pointer and type.
 *
 * @param ptr Pointer to the scalar data.
 * @param ty The type of the scalar.
 */
inline Scalar make_scalar(void* ptr, const Type& ty) {
  return Scalar(ty, ptr, true);
}

/**
 * @ingroup legate_wrapper
 * @brief Create an unbound array.
 *
 * @param ty The type of the array elements.
 * @param dim The number of dimensions (default 1).
 * @param nullable Whether the array can contain nulls (default false).
 */
inline LogicalArray create_unbound_array(const Type& ty, std::uint32_t dim = 1,
                                         bool nullable = false) {
  return Runtime::get_runtime()->create_array(ty, dim, nullable);
}

/**
 * @ingroup legate_wrapper
 * @brief Create an array with a specified shape.
 *
 * @param shape The shape of the array.
 * @param ty The type of the array elements.
 * @param nullable Whether the array can contain nulls (default false).
 * @param optimize_scalar Whether to optimize scalar storage (default false).
 */
inline LogicalArray create_array(const Shape& shape, const Type& ty,
                                 bool nullable = false,
                                 bool optimize_scalar = false) {
  return Runtime::get_runtime()->create_array(shape, ty, nullable,
                                              optimize_scalar);
}

/**
 * @ingroup legate_wrapper
 * @brief Create an unbound store.
 *
 * @param ty The type of the store elements.
 * @param dim The dimensionality (default 1).
 */
inline LogicalStore create_unbound_store(const Type& ty,
                                         std::uint32_t dim = 1) {
  return Runtime::get_runtime()->create_store(ty, dim);
}

/**
 * @ingroup legate_wrapper
 * @brief Create a store with a specified shape.
 *
 * @param shape The shape of the store.
 * @param ty The type of the store elements.
 * @param optimize_scalar Whether to optimize scalar storage (default false).
 */
inline LogicalStore create_store(const Shape& shape, const Type& ty,
                                 bool optimize_scalar = false) {
  return Runtime::get_runtime()->create_store(shape, ty, optimize_scalar);
}

/**
 * @ingroup legate_wrapper
 * @brief Create a store from a scalar value.
 *
 * @param scalar The scalar to store.
 * @param shape The shape of the store (default 1-element).
 */
inline LogicalStore store_from_scalar(const Scalar& scalar,
                                      const Shape& shape = Shape{1}) {
  return Runtime::get_runtime()->create_store(scalar, shape);
}

/**
 * @ingroup legate_wrapper
 * @brief Attach an external store in system memory.
 *
 * @param ptr Pointer to the external memory.
 * @param shape The shape of the store.
 * @param ty The type of the store elements.
 */
inline LogicalStore attach_external_store_sysmem(void* ptr, const Shape& shape,
                                                 const Type& ty,
                                                 bool read_only) {
  legate::ExternalAllocation alloc = legate::ExternalAllocation::create_sysmem(
      ptr, shape.volume() * ty.size(), read_only);
  legate::mapping::DimOrdering ordering =
      legate::mapping::DimOrdering::c_order();

  auto store =
      legate::Runtime::get_runtime()->create_store(shape, ty, alloc, ordering);
  return store;
}

/**
 * @ingroup legate_wrapper
 * @brief Attach an external store in frame buffer memory.
 *
 * @param device_id The device ID.
 * @param ptr Pointer to the external memory.
 * @param shape The shape of the store.
 * @param ty The type of the store elements.
 * @param readonly Whether the store is read-only.
 */
inline LogicalStore attach_external_store_fbmem(int device_id, void* ptr,
                                                const Shape& shape,
                                                const Type& ty, bool readonly) {
  legate::ExternalAllocation alloc = legate::ExternalAllocation::create_fbmem(
      device_id, ptr, shape.volume() * ty.size(), readonly);
  legate::mapping::DimOrdering ordering =
      legate::mapping::DimOrdering::fortran_order();

  auto store =
      legate::Runtime::get_runtime()->create_store(shape, ty, alloc, ordering);
  return store;
}

struct GetPtrFunctor {
  template <legate::Type::Code CODE, int DIM>
  void* operator()(legate::PhysicalStore* store) {
#if !LEGATE_DEFINED(LEGATE_USE_CUDA)
    // Check if FLOAT16 is being used without CUDA support
    if (CODE == legate::Type::Code::FLOAT16) {
      throw std::runtime_error(
          "FLOAT16 type is not supported when building without CUDA");
    }
#endif
    using CppT = typename legate_util::code_to_cxx<CODE>::type;
    auto shp = store->shape<DIM>();
    return store->write_accessor<CppT, DIM>().ptr(Realm::Point<DIM>(shp.lo));
  }
};

/**
 * @ingroup legate_wrapper
 * @brief Get a pointer to the data in a PhysicalStore.
 *
 * @param store Pointer to the PhysicalStore.
 */
inline void* get_ptr(legate::PhysicalStore* store) {
  int dim = store->dim();
  legate::Type::Code code = store->type().code();
  return legate::double_dispatch(dim, code, GetPtrFunctor{}, store);
}

/**
 * @ingroup legate_wrapper
 * @brief Issue a copy between stores.
 *
 * @param target The target store for the copy.
 * @param source The source store for the copy.
 */
inline void issue_copy(LogicalStore& target, const LogicalStore& source) {
  Runtime::get_runtime()->issue_copy(target, source);
}

}  // namespace data

namespace time {

/**
 * @ingroup legate_wrapper
 * @brief Measure time in microseconds.
 */
inline uint64_t time_microseconds() {
  return legate::timing::measure_microseconds().value();
}

/**
 * @ingroup legate_wrapper
 * @brief Measure time in nanoseconds.
 */
inline uint64_t time_nanoseconds() {
  return legate::timing::measure_nanoseconds().value();
}
}  // namespace time
}  // namespace legate_wrapper
