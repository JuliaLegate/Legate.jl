# Experimental feature guard. Mirrors cuNumeric.jl.

function Experimental(setting::Bool)
    setting && _check_ufi_thread_configuration()
    return task_local_storage(:Experimental, setting)
end

function assert_experimental()
    if get(task_local_storage(), :Experimental, false) !== true
        throw(
            ArgumentError(
                "Experimental features are disabled." *
                " Use `Legate.Experimental(true)` to enable them.",
            ),
        )
    end
end
