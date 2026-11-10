struct PresetTimeFunction{T, T2, T3}
    tstops::T
    filter_tstops::Bool
    initialize::T2
    user_affect!::T3
    cache_key::Symbol
    sort_inplace::Bool
end
function PresetTimeFunction(tstops, filter_tstops, initialize, user_affect!)
    return PresetTimeFunction(
        tstops, filter_tstops, initialize, user_affect!,
        gensym(:PresetTimeCallback_cache), false
    )
end

function _normalize_preset_tstops(tstops, sort_inplace)
    if tstops isa Number
        return [tstops]
    elseif tstops isa AbstractVector
        return sort_inplace ? sort!(tstops) : sort(tstops)
    else
        throw(
            ArgumentError(
                "tstops must either be a number, a Vector, or a callable " *
                    "`tstops(integrator)` returning a number or a Vector. Was $tstops"
            )
        )
    end
end

# A callable `tstops` is evaluated once per solve, in the initialize below,
# and the result is kept in `task_local_storage` under a per-callback key so
# concurrent integrators sharing one callback on separate Tasks (for example
# threaded ensembles) do not overwrite each other's stops.
function _resolved_preset_tstops(f::PresetTimeFunction, integrator)
    f.tstops isa AbstractVector && return f.tstops
    resolved = get(task_local_storage(), f.cache_key, nothing)
    resolved === nothing || return resolved
    resolved = _normalize_preset_tstops(f.tstops(integrator), f.sort_inplace)
    task_local_storage()[f.cache_key] = resolved
    return resolved
end

function (f::PresetTimeFunction)(u, t, integrator)
    tstops = _resolved_preset_tstops(f, integrator)
    return if hasproperty(integrator, :dt)
        insorted(t, tstops) && (integrator.t - integrator.dt) != integrator.t
    else
        insorted(t, tstops)
    end
end

function (f::PresetTimeFunction)(c, u, t, integrator)
    f.initialize(c, u, t, integrator)
    tstops = if f.tstops isa AbstractVector
        f.tstops
    else
        resolved = _normalize_preset_tstops(f.tstops(integrator), f.sort_inplace)
        task_local_storage()[f.cache_key] = resolved
        resolved
    end

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

  - `tstops`: one time, a collection of callback times, or a callable
    `tstops(integrator)` returning a number or a collection of callback times. The
    callable is evaluated when the callback is initialized, so the stops can depend on
    the problem as it is being solved, for example `integrator -> integrator.p` when the
    parameters are the event times. The same callback can then be reused after
    `remake(prob; p = ...)` without rebuilding it. Vector inputs and vector results from
    the callable are sorted before use.
  - `user_affect!`: a function `user_affect!(integrator)` applied at each scheduled stop.

# Keywords

  - `initialize = INITIALIZE_DEFAULT`: callback initialization function called as
    `initialize(callback, u, t, integrator)` before the stops are scheduled.
  - `filter_tstops::Bool = true`: schedule only stops strictly inside the integration
    interval. Set this to `false` to schedule all supplied stops, including values outside
    that interval.
  - `sort_inplace::Bool = false`: sort a vector `tstops`, or a vector returned by a
    callable `tstops`, in place. By default, sort a copy and leave the supplied or
    returned vector unchanged.
  - `kwargs...`: keyword arguments forwarded to `DiscreteCallback`.

# Returns

  - `DiscreteCallback`: a callback that schedules the requested stops and calls
    `user_affect!`.

# Throws

  - `ArgumentError`: if `tstops` is neither a number, a vector, nor a callable, or if a
    callable `tstops` returns neither a number nor a vector.

# Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

hits = Float64[]
affect! = integrator -> push!(hits, integrator.t)
cb = PresetTimeCallback([0.25, 0.5, 0.75], affect!)

prob = ODEProblem((u, p, t) -> -u, 1.0, (0.0, 1.0))
sol = solve(prob, Tsit5(); callback = cb)
```

With parameter-dependent stops, pass a callable instead; here the parameters
themselves are the event times:

```julia
cb = PresetTimeCallback(integrator -> integrator.p, affect!)
prob = ODEProblem((u, p, t) -> -u, 1.0, (0.0, 1.0), [0.25, 0.5])
sol = solve(prob, Tsit5(); callback = cb)
sol = solve(remake(prob; p = [0.75]), Tsit5(); callback = cb)
```
"""
function PresetTimeCallback(
        tstops, user_affect!;
        initialize = INITIALIZE_DEFAULT,
        filter_tstops = true,
        sort_inplace = false, kwargs...
    )
    if tstops isa AbstractVector || tstops isa Number
        tstops = _normalize_preset_tstops(tstops, sort_inplace)
    elseif isempty(methods(tstops))
        throw(
            ArgumentError(
                "tstops must either be a number, a Vector, or a callable " *
                    "`tstops(integrator)`. Was $tstops"
            )
        )
    end

    ptf = PresetTimeFunction(
        tstops, filter_tstops, initialize, user_affect!,
        gensym(:PresetTimeCallback_cache), sort_inplace
    )
    return DiscreteCallback(ptf, user_affect!; initialize = ptf, kwargs...)
end

export PresetTimeCallback
