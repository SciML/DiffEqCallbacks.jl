struct PresetTimeFunction{T, T2, T3}
    tstops::T
    filter_tstops::Bool
    initialize::T2
    user_affect!::T3
end
function (f::PresetTimeFunction)(u, t, integrator)
    return if hasproperty(integrator, :dt)
        insorted(t, f.tstops) && (integrator.t - integrator.dt) != integrator.t
    else
        insorted(t, f.tstops)
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
    return if insorted(t, tstops)
        f.user_affect!(integrator)
    end
end

"""
    PresetTimeCallback(tstops, user_affect!; initialize = INITIALIZE_DEFAULT,
        filter_tstops = true, sort_inplace = false, kwargs...) -> DiscreteCallback

Construct a callback that schedules `user_affect!` at the supplied integration-time stops.

# Arguments

  - `tstops::Union{Number, AbstractVector}`: one time or a collection of callback times.
    Vector inputs are sorted before use.
  - `user_affect!`: a function `user_affect!(integrator)` applied at each scheduled stop.

# Keywords

  - `initialize = INITIALIZE_DEFAULT`: callback initialization function called as
    `initialize(callback, u, t, integrator)` before the stops are scheduled.
  - `filter_tstops::Bool = true`: schedule only stops strictly inside the integration
    interval. Set this to `false` to schedule all supplied stops, including values outside
    that interval.
  - `sort_inplace::Bool = false`: sort a vector `tstops` in place. By default, sort a
    copy and leave the supplied vector unchanged.
  - `kwargs...`: keyword arguments forwarded to `DiscreteCallback`.

# Returns

  - `DiscreteCallback`: a callback that schedules the requested stops and calls
    `user_affect!`.

# Throws

  - `ArgumentError`: if `tstops` is neither a number nor a vector.

# Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

hits = Float64[]
affect! = integrator -> push!(hits, integrator.t)
cb = PresetTimeCallback([0.25, 0.5, 0.75], affect!)

prob = ODEProblem((u, p, t) -> -u, 1.0, (0.0, 1.0))
sol = solve(prob, Tsit5(); callback = cb)
```
"""
function PresetTimeCallback(
        tstops, user_affect!;
        initialize = INITIALIZE_DEFAULT,
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
