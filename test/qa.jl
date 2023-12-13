using DiffEqCallbacks, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(DiffEqCallbacks)
    Aqua.test_ambiguities(DiffEqCallbacks, recursive = false)
    Aqua.test_deps_compat(DiffEqCallbacks)
    Aqua.test_piracies(DiffEqCallbacks,
        treat_as_own = [])
    Aqua.test_project_extras(DiffEqCallbacks)
    Aqua.test_stale_deps(DiffEqCallbacks)
    Aqua.test_unbound_args(DiffEqCallbacks)
    Aqua.test_undefined_exports(DiffEqCallbacks)
end
