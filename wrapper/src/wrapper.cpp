#include "legate.h"
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"

#include <type_traits>
#include <vector>

#include "types.h"

using namespace legate;


JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {

    wrap_type_enums(mod); // legate::Type

    mod.map_type<LocalTaskID>("LocalTaskID");
    mod.map_type<GlobalTaskID>("GlobalTaskID");

    mod.add_type<Shape>("Shape")
        .constructor<std::vector<std::uint64_t>>();
    
    mod.add_type<Scalar>("Scalar")
        .constructor<float>() 
        .constructor<double>()
        .constructor<int>(); // this is technically templated, but this is easier for now
    jlcxx::stl::apply_stl<legate::Scalar>(mod); //enables std::vector<legate::Scalar> among other things

    mod.add_type<Library>("LegateLibrary");

    // these have all the accessor methods
    mod.add_type<LogicalStore>("LogicalStore");
    mod.add_type<PhysicalStore>("PhysicalStore");

    mod.add_type<PhysicalArray>("PhysicalArray")
        .method("nullable", &PhysicalArray::nullable)
        .method("dim", &PhysicalArray::dim)
        .method("type", &PhysicalArray::type)
        .method("data", &PhysicalArray::data);

    mod.add_type<LogicalArray>("LogicalArray")
        .method("dim", &LogicalArray::dim)
        .method("type", &LogicalArray::type)
        .method("shape", &LogicalArray::shape)
        .method("unbound", &LogicalArray::unbound)
        .method("nullable", &LogicalArray::nullable);

    mod.add_type<AutoTask>("AutoTask");
    mod.add_type<ManualTask>("ManualTask");

    mod.add_type<Runtime>("LegateRuntime")
        .method("create_auto_task", static_cast<AutoTask (Runtime::*)(Library, LocalTaskID)>(&Runtime::create_task))
        // issues with r-value references. Probably need to wrap a function and std::move the Task object
        // .method("submit_auto_task", static_cast<void (Runtime::*)(AutoTask&&)>(&Runtime::submit))
        // .method("submit_manual_task", static_cast<void (Runtime::*)>(ManualTask&&)(&Runtime::submit))
        .method("create_unbound_array", static_cast<LogicalArray (Runtime::*)(const Type&, std::uint32_t, bool)>(&Runtime::create_array),
                 jlcxx::kwarg("dim") = 1, jlcxx::kwarg("nullable") = false)
        .method("create_array", static_cast<LogicalArray (Runtime::*)(const Shape&, const Type&, bool, bool)>(&Runtime::create_array),
                 jlcxx::kwarg("nullable") = false, jlcxx::kwarg("optimize_scalar") = false)
        .method("create_unbound_store", static_cast<LogicalStore (Runtime::*)(const Type&, std::uint32_t)>(&Runtime::create_store), 
                 jlcxx::kwarg("dim") = 1)
        .method("create_store", static_cast<LogicalStore (Runtime::*)(const Shape&, const Type&, bool)>(&Runtime::create_store),
                 jlcxx::kwarg("optimize_scalar") = false)
        .method("store_from_scalar", static_cast<LogicalStore (Runtime::*)(const Scalar&, const Shape&)>(&Runtime::create_store),
                 jlcxx::kwarg("shape") = Shape{1});


    // intialization & cleanup
    mod.method("start", static_cast<void (*)()>(&legate::start));
    mod.method("has_started", &legate::has_started);
    mod.method("finish", &legate::finish);
    mod.method("has_finished", &legate::has_finished);


    // timing methods
    mod.add_type<timing::Time>("Time").method(
        "value", &timing::Time::value);
    mod.method("time_microseconds", &timing::measure_microseconds);
    mod.method("time_nanoseconds", &timing::measure_nanoseconds);

}