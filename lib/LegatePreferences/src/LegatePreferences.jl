#= Copyright 2026 Northwestern University, 
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
=#

module LegatePreferences
using Preferences

include("PreferenceBackend.jl")
using .PrefBackend

@make_preferences("legate_")

function maybe_warn_prerelease()
    load_preference(@__MODULE__, "legate_suppress_prerelease_warning", false) && return nothing

    @warn """
        Leagte.jl and cuNumeric.jl are under active development at the moment. This is a pre-release API and is subject to change. 
        Stability is not guaranteed until the first official release. We are actively working to improve the build experience to be more seamless
        and Julia-friendly. In parallel, we're developing a comprehensive testing framework to ensure reliability and robustness.
    """

    _set("suppress_prerelease_warning" => true;)
end

end # module LegatePreferences
