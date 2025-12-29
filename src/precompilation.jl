using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    @compile_workload begin
        # Precompile commonly used callback constructors with typical argument types

        # PeriodicCallback - very commonly used for periodic saves/actions
        periodic_cb = PeriodicCallback(u -> nothing, 1.0)
        periodic_cb_phase = PeriodicCallback(u -> nothing, 0.5; phase = 0.1)

        # SavingCallback - commonly used for custom saving
        saved_values_float = SavedValues(Float64, Float64)
        saving_cb_float = SavingCallback((u, t, integrator) -> t, saved_values_float)

        saved_values_vec = SavedValues(Float64, Vector{Float64})
        saving_cb_vec = SavingCallback((u, t, integrator) -> copy(u),
            saved_values_vec)

        # PresetTimeCallback - commonly used for events at specific times
        preset_cb = PresetTimeCallback([1.0, 2.0, 3.0], integrator -> nothing)
        preset_cb_single = PresetTimeCallback(1.0, integrator -> nothing)

        # TerminateSteadyState - commonly used for steady-state detection
        terminate_cb = TerminateSteadyState()
        terminate_cb_tols = TerminateSteadyState(1e-8, 1e-6)

        # AutoAbstol - commonly used for adaptive tolerance
        autoabstol_cb = AutoAbstol()
        autoabstol_cb_init = AutoAbstol(true; init_curmax = 1e-6)

        # IterativeCallback - used for custom iteration patterns
        iterative_cb = IterativeCallback(
            integrator -> integrator.t + 1.0,
            integrator -> nothing)
    end
end
