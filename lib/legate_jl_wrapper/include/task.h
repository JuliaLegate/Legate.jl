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

#pragma once

#include <legate.h>

#include "jlcxx/jlcxx.hpp"

namespace ufi {
enum TaskIDs {
  // max local task ID for custom library
  // for some reason cupynumeric can have larger IDs? Not sure why.
  JULIA_CUSTOM_TASK = 1023,
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  JULIA_CUSTOM_GPU_TASK = 1022,
#endif
};

class JuliaCustomTask : public legate::LegateTask<JuliaCustomTask> {
 public:
  static inline const auto TASK_CONFIG =
      legate::TaskConfig{legate::LocalTaskID{ufi::JULIA_CUSTOM_TASK}};

  static void cpu_variant(legate::TaskContext context);
};

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
class JuliaCustomGPUTask : public legate::LegateTask<JuliaCustomGPUTask> {
 public:
  static inline const auto TASK_CONFIG =
      legate::TaskConfig{legate::LocalTaskID{ufi::JULIA_CUSTOM_GPU_TASK}};

  static void gpu_variant(legate::TaskContext context);
};
#endif

}  // namespace ufi

void wrap_ufi(jlcxx::Module& mod);