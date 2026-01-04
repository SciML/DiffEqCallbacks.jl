using DiffEqCallbacks, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(DiffEqCallbacks)
    Aqua.test_ambiguities(DiffEqCallbacks, recursive = false)
    Aqua.test_deps_compat(DiffEqCallbacks)
    Aqua.test_piracies(
        DiffEqCallbacks,
        treat_as_own = []
    )
    Aqua.test_project_extras(DiffEqCallbacks)
    Aqua.test_stale_deps(DiffEqCallbacks; ignore = [:JET])
    Aqua.test_unbound_args(DiffEqCallbacks)
    Aqua.test_undefined_exports(DiffEqCallbacks)
end

using JET
@testset "JET" begin
    # JET tests are not run on LTS due to version-specific behavior
    if VERSION >= v"1.11"
        # Basic package analysis - only check for critical issues in DiffEqCallbacks itself
        # Note: Many JET warnings come from dependencies, so we use target_modules filter
        @test_opt target_modules = (DiffEqCallbacks,) DiffEqCallbacks.SavedValues(Float64, Float64)
        @test_opt target_modules = (DiffEqCallbacks,) DiffEqCallbacks.IntegrandValues(Float64, Float64)
        @test_opt target_modules = (DiffEqCallbacks,) DiffEqCallbacks.IntegrandValuesSum(Float64)
    else
        @test_broken false  # JET tests skipped on LTS
    end
end
