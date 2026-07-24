"""
    IterativeCallback(time_choice, user_affect!, tType = Float64;
        initial_affect = false, kwargs...)

A callback to be used to iteratively apply some affect. For example, if given the first
effect at `t₁`, you can define `t₂` to apply the next effect.

## Arguments

  - `time_choice(integrator)` determines the time of the next callback. If `nothing` is
    returned for the time choice, then the iterator ends.
  - `user_affect!` is the effect applied to the integrator at the stopping points.

## Keyword Arguments

  - `initial_affect` is whether to apply the affect at `t=0` which defaults to `false`
  - `initialize` is the callback initialization function.
  - `kwargs` are keyword arguments accepted by `DiscreteCallback`.

## Returns

A `DiscreteCallback` that repeatedly schedules the next time returned by
`time_choice`.

## Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

count = Ref(0)
hits = Float64[]
time_choice = integrator -> (count[] += 1; count[] <= 3 ? integrator.t + 0.1 : nothing)
affect! = integrator -> push!(hits, integrator.t)
cb = IterativeCallback(time_choice, affect!)

prob = ODEProblem((u, p, t) -> -u, 1.0, (0.0, 1.0))
sol = solve(prob, Tsit5(); callback = cb)
```
"""
function IterativeCallback(
        time_choice, user_affect!, tType = Float64;
        initial_affect = false,
        initialize = (cb, u, t, integrator) -> derivative_discontinuity!(
            integrator,
            initial_affect
        ),
        kwargs...
    )
    # Value of `t` at which `f` should be called next:
    tnext = Ref{Union{Nothing, eltype(tType)}}(typemax(tType))
    condition = function (u, t, integrator)
        return t == tnext[]
    end

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = function (integrator)
        user_affect!(integrator)

        # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
        tnew = time_choice(integrator)
        tnew === nothing && (tnext[] = tnew; return)
        tstops = get_tstops(integrator)
        #=
        Okay yeah, this is nasty
        the comparer is always less than for type stability, so in order
        for this to actually check the correct direction we multiply by
        tdir
        =#
        tdir_tnew = integrator.tdir * tnew
        tstops_array = get_tstops_array(integrator)
        for i in length(tstops):-1:1 # reverse iterate to encounter large elements earlier
            if tdir_tnew < tstops_array[i]
                tnext[] = tnew
                add_tstop!(integrator, tnew)
                break
            elseif tdir_tnew == tstops_array[i]
                # If it's already a tstop, no need to re-add! This is for the final point
                tnext[] = tnew
            end
        end
        return nothing
    end

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_iterative = function (c, u, t, integrator)
        initialize(c, u, t, integrator)
        return if initial_affect
            tnext[] = t
            affect!(integrator)
        else
            tnext[] = time_choice(integrator)
            if tnext[] != nothing
                add_tstop!(integrator, tnext[])
            end
        end
    end
    return DiscreteCallback(condition, affect!; initialize = initialize_iterative, kwargs...)
end

export IterativeCallback

struct PeriodicCallbackAffect{A, dT, K}
    affect!::A
    Δt::dT
    cache_key::K
end

# The cache is stored per-Task in `task_local_storage`, keyed by a
# `gensym`-unique symbol per callback instance. `initialize_periodic` is the
# only place that creates/overwrites the entry, so concurrent integrators
# running the same callback on separate Tasks (e.g. `EnsembleThreads`) each
# get their own independent `t0`/`index`. See #99.
const _PeriodicCache{T} = NamedTuple{(:t0, :index), Tuple{Base.RefValue{T}, Base.RefValue{Int}}}

@inline function _periodic_cache(key::Symbol, ::Type{T}) where {T}
    return task_local_storage(key)::_PeriodicCache{T}
end

function (S::PeriodicCallbackAffect)(integrator)
    add_next_tstop!(integrator, S)

    return S.affect!(integrator)
end

function add_next_tstop!(integrator, S::PeriodicCallbackAffect)
    cache = _periodic_cache(S.cache_key, typeof(S.Δt))
    # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever.
    # Convert into the integrator time type so Rational Δt produces Float64
    # tstops that match the condition comparison (DifferentialEquations.jl #878).
    tnew = convert(typeof(integrator.t), cache.t0[] + (cache.index[] + 1) * S.Δt)
    #=
    Okay yeah, this is nasty
    the comparer is always less than for type stability, so in order
    for this to actually check the correct direction we multiply by
    tdir
    =#
    tdir_tnew = integrator.tdir * tnew
    cache.index[] += 1
    return if tdir_tnew < get_tstops_max(integrator)
        add_tstop!(integrator, tnew)
    end
end

"""
    PeriodicCallback(f, Δt::Number; phase = 0, initial_affect = false,
        final_affect = false, kwargs...)

`PeriodicCallback` can be used when a function should be called periodically in terms of
integration time (as opposed to wall time), i.e. at `t = tspan[1]`, `t = tspan[1] + Δt`,
`t = tspan[1] + 2Δt`, and so on.

If a non-zero `phase` is provided, the invocations of the callback will be shifted by
`phase` time units, i.e., the calls will occur at
`t = tspan[1] + phase`, `t = tspan[1] + phase + Δt`,
`t = tspan[1] + phase + 2Δt`, and so on.

This callback can, for example, be used to model a
discrete-time controller for a continuous-time system, running at a fixed rate.

## Arguments

  - `f` the `affect!(integrator)` function to be called periodically
  - `Δt` is the period

  ## Keyword Arguments

  - `phase` is a phase offset
  - `initial_affect` is whether to apply the affect at the initial time, which defaults to `false`
  - `final_affect` is whether to apply the affect at the final time, which defaults to `false`
  - `kwargs` are keyword arguments accepted by the `DiscreteCallback` constructor.

## Returns

A `DiscreteCallback` that schedules an `affect!` call every `Δt` units of
integration time.

## Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

samples = Float64[]
affect! = integrator -> push!(samples, integrator.u)
cb = PeriodicCallback(affect!, 0.1; initial_affect = true)

prob = ODEProblem((u, p, t) -> -u, 1.0, (0.0, 1.0))
sol = solve(prob, Tsit5(); callback = cb)
```
"""
function PeriodicCallback(
        f, Δt::Number;
        phase = 0,
        initial_affect = false,
        final_affect = false,
        initialize = (cb, u, t, integrator) -> derivative_discontinuity!(
            integrator,
            initial_affect
        ),
        kwargs...
    )
    phase < 0 && throw(ArgumentError("phase offset must be non-negative"))
    # Unique per-callback key for the per-Task cache (see `_periodic_cache`).
    cache_key = gensym(:PeriodicCallback_cache)

    condition = function (u, t, integrator)
        cache = _periodic_cache(cache_key, typeof(Δt))
        fin = isfinished(integrator)
        # Convert the Rational (or other) period target into the integrator's
        # time type. Exact `t == t0 + index*Δt` fails when `t::Float64` and
        # `Δt::Rational` because e.g. `Float64(3//10) == 3//10` is false
        # (DifferentialEquations.jl #878).
        t_target = convert(typeof(t), cache.t0[] + cache.index[] * Δt)
        return (t == t_target && !fin) || (final_affect && fin)
    end

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = PeriodicCallbackAffect(f, Δt, cache_key)

    # Initialization: first call to `f` should be *before* any time steps have been taken:
    initialize_periodic = function (c, u, t, integrator)
        @assert integrator.tdir == sign(Δt)
        initialize(c, u, t, integrator)
        # Look up or create this Task's cache. The cache lives in
        # `task_local_storage` so each Task running this callback (e.g. ensemble
        # threads) has its own `t0`/`index` Refs. Allocation only happens the
        # first time a given Task initializes this callback; subsequent solves
        # on the same Task reuse the Refs and just rewrite their contents.
        storage = task_local_storage()
        existing = get(storage, cache_key, nothing)
        cache = if existing === nothing
            c = (t0 = Ref{typeof(Δt)}(), index = Ref(0))
            storage[cache_key] = c
            c
        else
            existing::_PeriodicCache{typeof(Δt)}
        end
        cache.t0[] = convert(typeof(Δt), t + phase)
        cache.index[] = iszero(phase) ? 0 : -1
        return if initial_affect
            affect!(integrator)
        else
            add_next_tstop!(integrator, affect!)
        end
    end

    return DiscreteCallback(condition, affect!; initialize = initialize_periodic, kwargs...)
end

@inline function isfinished(integrator)
    # Checking for floating point equality is OK here as `DifferentialEquations.jl`
    # sets the time exactly to the final time in the last iteration
    return integrator.t == last(integrator.sol.prob.tspan) ||
        (
        hasfield(typeof(integrator), :iter) &&
            (integrator.iter == integrator.opts.maxiters)
    )
end

export PeriodicCallback
