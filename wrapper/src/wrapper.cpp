#include "legate.h"

#include "types.h"

#include <vector>

using namespace legate;

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {

    wrap_type_enums(mod); // legate::Type

    // Need to wrap LogicalArray, LogicalStore

    mod.add_type<LocalTaskID>("LocalTaskID");
    mod.add_type<GlobalTaskID>("GlobalTaskID");

    mod.add_type<Shape>("Shape")
        .constructor<std::vector<std::uint64_t>>();
    
    mod.add_type<Scalar>("Scalar")
        .constructor<float>() 
        .constructor<double>()
        .constructor<int>(); // this is technically templated, but this is easier for now

    mod.add_type<Library>("LegateLibrary");

    // these have all the accessor methods
    mod.add_type<LogicalStore>("LogicalStore");
    mod.add_type<PhysicalStore>("PhysicalStore");

    mod.add_type<PhysicalArray>("PhysicalArray")
        .method("nullable", &PhysicalArray::nullable)
        .method("dim", &PhysicalArray::dim)
        .method("type", &PhysicalArray::type)
        .method("data", &PhysicalArray::data)

    mod.add_type<LogicalArray>("LogicalArray")
        .method("dim", &LogicalArray::dim)
        .method("type", &LogicalArray::type)
        .method("shape", &LogicalArray::shape)
        .method("unbound", &LogicalArray::unbound)
        .method("nullable", &LogicalArray::nullable)
        .method("data", &LogicalStore::data)
        .method("get_physical_array", &LogicalStore::get_physical_array)

    mod.add_type<AutoTask>("AutoTask");
    mod.add_type<ManualTask>("ManualTask");

    mod.add_type<Runtime>("LegateRuntime")
        .method("create_auto_task", static_cast<AutoTask (Runtime::*)(Library, LocalTaskID)>(&Runtime::create_task))
        .method("submit_auto_task", static_cast<void (Runtime::*)(AutoTask&&)>(&Runetime::submit))
        .method("submit_manual_task", static_cast<void (Runtime::*)>(ManualTask&&)(&Runtime::submit))
        .method("create_unbound_array", static_cast<LogicalArray (Runtime::*)(const Type*, std::uint32_t, bool)>(&Runtime::create_array),
                 jlcxx::kwarg("dim") = 1, jlcxx::kwrag("nullable") = false)
        .method("create_array", static_cast<LogicalArray (Runtime::*)(const Shape&, const Type&, bool, bool)>(&Runtime::create_array),
                 jlcxx::kwarg("nullable") = false, jlcxx::kwarg("optimize_scalar") = false)
        .method("create_unbound_store", static_cast<LogicalStore (Runtime::*)(const Type&, std::uint32_t)>(&Runtime::create_store), 
                 jlcxx::kwarg("dim") = 1)
        .method("create_store", static_cast<LogicalStore (Runtime::*)(const Shape&, const Type&, bool)>(&Runtime::create_store),
                 jlcxx::kwarg("optimize_scalar") = false)
        .method("store_from_scalar", static_cast<LogicalStore (Runtime::*)(const Scalar&, const Shape&)(&Runtime::create_store),
                 jlcxx::kwarg("shape") = Shape{1})
        .method("start", &Runtime::start)
        .method("has_started", &Runtime::start)
        .method("finish", &Runtime::finish)
        .method("has_finished", &Runtime::has_finished);

}