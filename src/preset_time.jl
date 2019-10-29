function PresetTimeCallback(tstops,user_affect!;
                            initialize = DiffEqBase.INITIALIZE_DEFAULT, kwargs...)
    condition = function (u, t, integrator)
      t in tstops
    end

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = function (integrator)
        user_affect!(integrator)
        nothing
    end

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_preset = function (c, u, t, integrator)
        initialize(c, u, t, integrator)
        add_tstop!.((integrator,), tstops)
        if t ∈ tstops
            user_affect!(integrator)
        end
    end
    DiscreteCallback(condition, affect!; initialize = initialize_preset, kwargs...)
end

export PresetTimeCallback
