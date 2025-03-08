#include "legate.h"
#include "legate/timing/timing.h"
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"

#include <type_traits>
#include <vector>

#include "types.h"

using namespace legate;


struct WrapDefault {
    template <typename TypeWrapperT>
    void operator()(TypeWrapperT&& wrapped) {
      typedef typename TypeWrapperT::type WrappedT;
      wrapped.template constructor<typename WrappedT::value_type>();
    }
  };


JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {

    using jlcxx::Parametric;
    using jlcxx::TypeVar;

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

    mod.add_type<Parametric<TypeVar<1>>>("StdOptional")
      .apply<std::optional<legate::Type>, std::optional<int64_t>>(WrapDefault());

    mod.add_type<legate::Slice>("LegateSlice")
      .constructor<std::optional<int64_t>, std::optional<int64_t>>(jlcxx::kwarg("_start") = Slice::OPEN, jlcxx::kwarg("_stop") = Slice::OPEN);

    mod.add_type<Library>("LegateLibrary");

    mod.add_type<LogicalStore>("LogicalStore")
        .method("dim", &LogicalStore::dim)
        .method("type", &LogicalStore::type)
        .method("reinterpret_as", &LogicalStore::reinterpret_as)
        .method("promote", &LogicalStore::promote)
        .method("slice", &LogicalStore::slice)
        .method("get_physical_store", &LogicalStore::get_physical_store)
        .method("equal_storage", &LogicalStore::equal_storage);

    // This has all the accessor methods
    mod.add_type<PhysicalStore>("PhysicalStore")
        .method("dim", &PhysicalStore::dim)
        .method("type", &PhysicalStore::type)
        .method("is_readable", &PhysicalStore::is_readable)
        .method("is_writable", &PhysicalStore::is_writable)
        .method("is_reducible", &PhysicalStore::is_reducible)
        .method("valid", &PhysicalStore::valid);

    mod.add_type<PhysicalArray>("PhysicalArray")
        .method("nullable", &PhysicalArray::nullable)
        .method("dim", &PhysicalArray::dim)
        .method("type", &PhysicalArray::type)
        .method("data", &PhysicalArray::data);

    mod.add_type<LogicalArray>("LogicalArray")
        .method("dim", &LogicalArray::dim)
        .method("type", &LogicalArray::type)
        .method("unbound", &LogicalArray::unbound)
        .method("nullable", &LogicalArray::nullable);

    mod.add_type<Variable>("LegateVariable");

    mod.add_type<AutoTask>("AutoTask")
        .method("add_input", static_cast<Variable (AutoTask::*)(LogicalArray)>(&AutoTask::add_input))
        .method("add_output", static_cast<Variable (AutoTask::*)(LogicalArray)>(&AutoTask::add_output));
    mod.add_type<ManualTask>("ManualTask")
        .method("add_input", static_cast<void (ManualTask::*)(LogicalStore)>(&ManualTask::add_input))
        .method("add_output", static_cast<void (ManualTask::*)(LogicalStore)>(&ManualTask::add_output));

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
    // TODO catch the (Auto)ConfigurationError and make the Julia error nicer.
    mod.method("start", static_cast<void (*)()>(&legate::start), jlcxx::calling_policy::std_function);
    mod.method("has_started", &legate::has_started);
    mod.method("finish", &legate::finish);
    mod.method("has_finished", &legate::has_finished);


    // timing methods
    mod.add_type<timing::Time>("Time").method(
        "value", &timing::Time::value);
    mod.method("time_microseconds", &timing::measure_microseconds);
    mod.method("time_nanoseconds", &timing::measure_nanoseconds);

}