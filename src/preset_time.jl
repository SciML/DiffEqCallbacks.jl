"""
```julia
PresetTimeCallback(tstops, user_affect!;
    initialize = DiffEqBase.INITIALIZE_DEFAULT,
    filter_tstops = true,
    kwargs...)
```

A callback that adds callback `affect!` calls at preset times. No playing around with
`tstops` or anything is required: this callback adds the triggers for you to make it
automatic.

## Arguments

  - `tstops`: the times for the `affect!` to trigger at.
  - `user_affect!`: an `affect!(integrator)` function to use at the time points.

## Keyword Arguments

  - `filter_tstops`: Whether to filter out tstops beyond the end of the integration timespan.
    Defaults to true. If false, then tstops can extend the interval of integration.
"""
function PresetTimeCallback(tstops, user_affect!;
        initialize = SciMLBase.INITIALIZE_DEFAULT,
        filter_tstops = true,
        sort_inplace = false, kwargs...)
    if tstops isa AbstractVector
        if sort_inplace
            sort!(tstops)
        else
            tstops = sort(tstops)
        end
        condition = let tstops=tstops
            function (u, t, integrator)
                if hasproperty(integrator, :dt)
                    insorted(t, tstops) && (integrator.t - integrator.dt) != integrator.t
                else
                    insorted(t, tstops)
                end
            end
        end
    elseif tstops isa Number
        condition = let tstops=tstops
            function (u, t, integrator)
                if hasproperty(integrator, :dt)
                    t == tstops && (integrator.t - integrator.dt) != integrator.t
                else
                    t == tstops
                end
            end
        end
    else
        throw(ArgumentError("tstops must either be a number or a Vector. Was $tstops"))
    end

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = function (integrator)
        user_affect!(integrator)
        nothing
    end

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_preset = function (c, u, t, integrator)
        initialize(c, u, t, integrator)

        if filter_tstops
            tdir = integrator.tdir
            tspan = integrator.sol.prob.tspan
            _tstops = tstops[@. tdir * tspan[1] < tdir * tstops < tdir * tspan[2]]
            add_tstop!.((integrator,), _tstops)
        else
            add_tstop!.((integrator,), tstops)
        end
        if t ∈ tstops
            user_affect!(integrator)
        end
    end
    DiscreteCallback(condition, affect!; initialize = initialize_preset, kwargs...)
end

export PresetTimeCallback
