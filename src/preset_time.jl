struct PresetTimeFunction{T, T2, T3}
    tstops::T
    filter_tstops::Bool
    initialize::T2
    user_affect!::T3
end
function (f::PresetTimeFunction)(u, t, integrator)
    return if hasproperty(integrator, :dt)
        is_preset_time(t, f.tstops) && (integrator.t - integrator.dt) != integrator.t
    else
        is_preset_time(t, f.tstops)
    end
end

function (f::PresetTimeFunction)(c, u, t, integrator)
    f.initialize(c, u, t, integrator)
    tstops = f.tstops

    if f.filter_tstops
        tdir = integrator.tdir
        tspan = integrator.sol.prob.tspan
        _tstops = tstops[@. tdir * tspan[1] < tdir * tstops < tdir * tspan[2]]
    else
        _tstops = tstops
    end
    for tstop in _tstops
        add_tstop!(integrator, tstop)
    end
    return if is_preset_time(t, tstops)
        f.user_affect!(integrator)
    end
end

function is_preset_time(t, tstops)
    insorted(t, tstops) && return true
    i = searchsortedfirst(tstops, t)
    return (i <= length(tstops) && is_close_preset_time(t, tstops[i])) ||
        (i > firstindex(tstops) && is_close_preset_time(t, tstops[i - 1]))
end

function is_close_preset_time(t, tstop)
    _t, _tstop = promote(t, tstop)
    if _t isa AbstractFloat && _tstop isa AbstractFloat && isfinite(_t) && isfinite(_tstop)
        scale = max(abs(_t), abs(_tstop))
        return abs(_t - _tstop) <= 100 * eps(scale)
    end

    return t == tstop
end

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
function PresetTimeCallback(
        tstops, user_affect!;
        initialize = SciMLBase.INITIALIZE_DEFAULT,
        filter_tstops = true,
        sort_inplace = false, kwargs...
    )
    if !(tstops isa AbstractVector) && !(tstops isa Number)
        throw(ArgumentError("tstops must either be a number or a Vector. Was $tstops"))
    end

    tstops = tstops isa Number ? [tstops] : (sort_inplace ? sort!(tstops) : sort(tstops))
    ptf = PresetTimeFunction(tstops, filter_tstops, initialize, user_affect!)
    return DiscreteCallback(ptf, user_affect!; initialize = ptf, kwargs...)
end

export PresetTimeCallback
