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

        @testset "Data structure constructors" begin
            @test_opt target_modules = (DiffEqCallbacks,) DiffEqCallbacks.SavedValues(
                Float64, Float64)
            @test_opt target_modules = (DiffEqCallbacks,) DiffEqCallbacks.IntegrandValues(
                Float64, Float64)
            @test_opt target_modules = (DiffEqCallbacks,) DiffEqCallbacks.IntegrandValuesSum(
                Float64)
        end

        @testset "Callback constructors" begin
            # Test TerminateSteadyState constructor
            @test_opt target_modules = (DiffEqCallbacks,) TerminateSteadyState()
            @test_opt target_modules = (DiffEqCallbacks,) TerminateSteadyState(
                1e-8, 1e-6; min_t = 1.0)

            # Test SavingCallback constructor
            saved_values = SavedValues(Float64, Float64)
            save_func = (u, t, integrator) -> sum(u)
            @test_opt target_modules = (DiffEqCallbacks,) SavingCallback(
                save_func, saved_values)

            # Test PeriodicCallback constructor
            periodic_affect = (integrator) -> nothing
            @test_opt target_modules = (DiffEqCallbacks,) PeriodicCallback(
                periodic_affect, 0.1)

            # Test PresetTimeCallback constructor
            preset_affect = (integrator) -> nothing
            @test_opt target_modules = (DiffEqCallbacks,) PresetTimeCallback(
                [0.5, 1.0], preset_affect)

            # Test IterativeCallback constructor
            time_choice = (integrator) -> integrator.t + 0.1
            iter_affect = (integrator) -> nothing
            @test_opt target_modules = (DiffEqCallbacks,) IterativeCallback(
                time_choice, iter_affect)

            # Test AutoAbstol constructor
            @test_opt target_modules = (DiffEqCallbacks,) AutoAbstol()

            # Test StepsizeLimiter constructor
            dtFE = (u, p, t) -> 0.01
            @test_opt target_modules = (DiffEqCallbacks,) StepsizeLimiter(dtFE)

            # Test FunctionCallingCallback constructor
            func = (u, t, integrator) -> nothing
            @test_opt target_modules = (DiffEqCallbacks,) FunctionCallingCallback(func)

            # Test PositiveDomain constructor
            @test_opt target_modules = (DiffEqCallbacks,) PositiveDomain()

            # Test IntegratingCallback constructor
            integrand_values = IntegrandValues(Float64, Float64)
            integrand_func = (u, t, integrator) -> sum(u)
            @test_opt target_modules = (DiffEqCallbacks,) IntegratingCallback(
                integrand_func, integrand_values, 0.0)
        end
    else
        @test_broken false  # JET tests skipped on LTS
    end
end
