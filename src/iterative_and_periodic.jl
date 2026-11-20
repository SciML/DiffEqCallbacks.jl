"""
    IterativeCallback(time_choice, user_affect!, tType = Float64;
        initial_affect = false, initialize = ..., kwargs...) -> DiscreteCallback

Construct a callback that applies `user_affect!` at the sequence of integration times
returned by `time_choice`.

# Arguments

  - `time_choice`: a function `time_choice(integrator)` that returns the next callback
    time or `nothing` to stop scheduling further affects.
  - `user_affect!`: a function `user_affect!(integrator)` applied at each scheduled time.
  - `tType::Type = Float64`: type used to store the next callback time. Set this to the
    problem time type when it is not `Float64`.

# Keywords

  - `initial_affect::Bool = false`: apply `user_affect!` during callback initialization
    at the initial integration time before asking `time_choice` for the next time.
  - `initialize = ...`: callback initialization function called as
    `initialize(callback, u, t, integrator)` before the initial affect or first scheduled
    stop.
    By default, it marks a derivative discontinuity according to `initial_affect`.
  - `kwargs...`: keyword arguments forwarded to `DiscreteCallback`.

# Returns

  - `DiscreteCallback`: a callback that schedules each time returned by `time_choice`
    until it returns `nothing`.

# Examples

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

struct PeriodicCallbackAffect{A, dT, K, P}
    affect!::A
    Δt::dT
    cache_key::K
    exact::Bool
    # type of `t0` in the per-task cache
    t0_type::Type{P}
end

# The cache is stored per-Task in `task_local_storage`, keyed by a
# `gensym`-unique symbol per callback instance. `initialize_periodic` is the
# only place that creates/overwrites the entry, so concurrent integrators
# running the same callback on separate Tasks (e.g. `EnsembleThreads`) each
# get their own independent `t0`/`index`. See #99.
const _PeriodicCache{T, S} = NamedTuple{
    (:t0, :index, :tnext), Tuple{Base.RefValue{T}, Base.RefValue{Int}, Base.RefValue{S}},
}

@inline function _periodic_cache(key::Symbol, ::Type{T}, ::Type{S}) where {T, S}
    return task_local_storage(key)::_PeriodicCache{T, S}
end

# For a rational period and a phase that is an integer, a rational or a
# floating-point number, the tick with index `k` is `convert(S, t0 + k*Δt)` with
# `t0 = tstart + phase`, all in `Rational{BigInt}`: floating-point values are
# converted exactly, the sum cannot overflow or cancel, and each tick is rounded
# once. Ticks of different callbacks of one integrator that coincide
# mathematically are then equal as floating-point numbers, which avoids steps of a
# few ulps between them, and the result does not depend on the integer type of the
# rationals. In all other cases `t0 = tstart + phase` is stored in the type of the
# period and the tick is `convert(S, t0 + k*Δt)`. The time of the scheduled tick is
# kept in the cache, so the condition does not repeat the arithmetic.
const _ExactReal = Union{Integer, Rational, AbstractFloat}
_exact_ticks(Δt, phase) = Δt isa Rational && phase isa _ExactReal
_exact(x::_ExactReal) = Rational{BigInt}(x)

@inline function _periodic_tick(cache, k, Δt, exact, ::Type{S}) where {S}
    return if exact
        convert(S, cache.t0[] + k * _exact(Δt))
    else
        convert(S, cache.t0[] + k * Δt)
    end
end

function (S::PeriodicCallbackAffect)(integrator)
    add_next_tstop!(integrator, S)

    return S.affect!(integrator)
end

function add_next_tstop!(integrator, S::PeriodicCallbackAffect)
    cache = _periodic_cache(S.cache_key, S.t0_type, typeof(integrator.t))
    # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever.
    # Convert into the integrator time type so Rational Δt produces Float64
    # tstops that match the condition comparison (DifferentialEquations.jl #878).
    tnew = _periodic_tick(cache, cache.index[] + 1, S.Δt, S.exact, typeof(integrator.t))
    #=
    Okay yeah, this is nasty
    the comparer is always less than for type stability, so in order
    for this to actually check the correct direction we multiply by
    tdir
    =#
    tdir_tnew = integrator.tdir * tnew
    cache.index[] += 1
    cache.tnext[] = tnew
    return if tdir_tnew < get_tstops_max(integrator)
        add_tstop!(integrator, tnew)
    end
end

"""
    PeriodicCallback(f, Δt::Number; phase = 0, initial_affect = false,
        final_affect = false, initialize = ..., kwargs...) -> DiscreteCallback

Construct a callback that applies `f` at regular intervals of integration time. Scheduled
stops are separated by `Δt` and are offset from the initial time by `phase`. When
`initial_affect = true`, `f` is also applied during callback initialization.

# Arguments

  - `f`: a function `f(integrator)` applied at each periodic stop.
  - `Δt::Number`: signed integration-time period. Its sign must match the integration
    direction.

# Keywords

  - `phase = 0`: nonnegative offset of scheduled periodic stops from the initial integration
    time. A negative phase throws an `ArgumentError`.
  - `initial_affect::Bool = false`: apply `f` during callback initialization at the initial
    integration time.
  - `final_affect::Bool = false`: apply `f` when the integrator finishes, even when the
    final time is not a periodic stop.
  - `initialize = ...`: callback initialization function called as
    `initialize(callback, u, t, integrator)` before periodic stops are scheduled. By
    default, it marks a derivative discontinuity according to `initial_affect`.
  - `kwargs...`: keyword arguments forwarded to `DiscreteCallback`.

# Returns

  - `DiscreteCallback`: a callback that schedules `f` at periodic integration-time stops.

# Throws

  - `ArgumentError`: if `phase < 0`.
  - `AssertionError`: during callback initialization if the sign of `Δt` does not match
    the integration direction.

# Examples

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
    exact = _exact_ticks(Δt, phase)
    P = exact ? Rational{BigInt} : typeof(Δt)

    condition = function (u, t, integrator)
        cache = _periodic_cache(cache_key, P, typeof(t))
        fin = isfinished(integrator)
        # The scheduled tick in the integrator's time type. Exact
        # `t == t0 + index*Δt` fails when `t::Float64` and `Δt::Rational` because
        # e.g. `Float64(3//10) == 3//10` is false (DifferentialEquations.jl #878).
        t_target = cache.tnext[]
        return (t == t_target && !fin) || (final_affect && fin)
    end

    # Call f, update tnext, and make sure we stop at the new tnext
    affect! = PeriodicCallbackAffect(f, Δt, cache_key, exact, P)

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
        cache = if existing isa _PeriodicCache{P, typeof(t)}
            existing
        else
            c = (t0 = Ref{P}(), index = Ref(0), tnext = Ref{typeof(t)}())
            storage[cache_key] = c
            c
        end
        cache.t0[] = exact ? _exact(t) + _exact(phase) : convert(P, t + phase)
        cache.index[] = iszero(phase) ? 0 : -1
        cache.tnext[] = _periodic_tick(cache, cache.index[], Δt, exact, typeof(t))
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
