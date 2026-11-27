# Condition, affect and initialization of an `IterativeCallback`. They share `tnext`, the
# time at which `user_affect!` is applied next.
mutable struct IterativeCallbackFunctor{F, A, I, T}
    time_choice::F
    user_affect!::A
    initialize::I
    initial_affect::Bool
    tnext::Union{Nothing, T}
end

# A functor with no time scheduled yet
function fresh_iterative_functor(
        time_choice, user_affect!, initialize, initial_affect, ::Type{T}
    ) where {T}
    F, A, I = typeof(time_choice), typeof(user_affect!), typeof(initialize)
    return IterativeCallbackFunctor{F, A, I, T}(
        time_choice, user_affect!, initialize, initial_affect, typemax(T)
    )
end

(f::IterativeCallbackFunctor)(u, t, integrator) = t == f.tnext

# Call f, update tnext, and make sure we stop at the new tnext
function (f::IterativeCallbackFunctor)(integrator)
    f.user_affect!(integrator)

    # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever
    tnew = f.time_choice(integrator)
    tnew === nothing && (f.tnext = tnew; return)
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
            f.tnext = tnew
            add_tstop!(integrator, tnew)
            break
        elseif tdir_tnew == tstops_array[i]
            # If it's already a tstop, no need to re-add! This is for the final point
            f.tnext = tnew
        end
    end
    return nothing
end

# Initialization: first call to `f` should be *before* any time steps have been taken:
function (f::IterativeCallbackFunctor)(c, u, t, integrator)
    f.initialize(c, u, t, integrator)
    return if f.initial_affect
        f.tnext = t
        f(integrator)
    else
        f.tnext = f.time_choice(integrator)
        if f.tnext !== nothing
            add_tstop!(integrator, f.tnext)
        end
    end
end

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
    f = fresh_iterative_functor(
        time_choice, user_affect!, initialize, initial_affect, eltype(tType)
    )
    return DiscreteCallback(f, f; initialize = f, kwargs...)
end

# Each integrator schedules its own times
function SciMLBase.instantiate_callback(
        cb::DiscreteCallback{<:IterativeCallbackFunctor{F, A, I, T}}
    ) where {F, A, I, T}
    f = cb.condition
    g = fresh_iterative_functor(
        f.time_choice, f.user_affect!, f.initialize, f.initial_affect, T
    )
    return SciMLBase.remake(cb; condition = g, affect! = g, initialize = g)
end

export IterativeCallback

# Condition, affect and initialization of a `PeriodicCallback`. They share `t0`, the time of
# the first periodic stop, and `index`, the number of the next one.
mutable struct PeriodicCallbackAffect{A, dT, P, I}
    affect!::A
    Δt::dT
    phase::P
    initial_affect::Bool
    final_affect::Bool
    initialize::I
    t0::dT
    index::Int
end

function PeriodicCallbackAffect(
        affect!, Δt, phase, initial_affect, final_affect, initialize
    )
    return PeriodicCallbackAffect(
        affect!, Δt, phase, initial_affect, final_affect, initialize, zero(Δt), 0
    )
end

function (S::PeriodicCallbackAffect)(u, t, integrator)
    fin = isfinished(integrator)
    # Convert the Rational (or other) period target into the integrator's
    # time type. Exact `t == t0 + index*Δt` fails when `t::Float64` and
    # `Δt::Rational` because e.g. `Float64(3//10) == 3//10` is false
    # (DifferentialEquations.jl #878).
    t_target = convert(typeof(t), S.t0 + S.index * S.Δt)
    return (t == t_target && !fin) || (S.final_affect && fin)
end

# Call f, update tnext, and make sure we stop at the new tnext
function (S::PeriodicCallbackAffect)(integrator)
    add_next_tstop!(integrator, S)

    return S.affect!(integrator)
end

# Initialization: first call to `f` should be *before* any time steps have been taken:
function (S::PeriodicCallbackAffect)(c, u, t, integrator)
    @assert integrator.tdir == sign(S.Δt)
    S.initialize(c, u, t, integrator)
    S.t0 = convert(typeof(S.Δt), t + S.phase)
    S.index = iszero(S.phase) ? 0 : -1
    return if S.initial_affect
        S(integrator)
    else
        add_next_tstop!(integrator, S)
    end
end

function add_next_tstop!(integrator, S::PeriodicCallbackAffect)
    # Schedule next call to `f` using `add_tstops!`, but be careful not to keep integrating forever.
    # Convert into the integrator time type so Rational Δt produces Float64
    # tstops that match the condition comparison (DifferentialEquations.jl #878).
    tnew = convert(typeof(integrator.t), S.t0 + (S.index + 1) * S.Δt)
    #=
    Okay yeah, this is nasty
    the comparer is always less than for type stability, so in order
    for this to actually check the correct direction we multiply by
    tdir
    =#
    tdir_tnew = integrator.tdir * tnew
    S.index += 1
    return if tdir_tnew < get_tstops_max(integrator)
        add_tstop!(integrator, tnew)
    end
end

# Each integrator keeps its own schedule, also when integrators run on the same task
function SciMLBase.instantiate_callback(cb::DiscreteCallback{<:PeriodicCallbackAffect})
    S = cb.condition
    fresh = PeriodicCallbackAffect(
        S.affect!, S.Δt, S.phase, S.initial_affect, S.final_affect, S.initialize
    )
    return SciMLBase.remake(cb; condition = fresh, affect! = fresh, initialize = fresh)
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
    S = PeriodicCallbackAffect(f, Δt, phase, initial_affect, final_affect, initialize)
    return DiscreteCallback(S, S; initialize = S, kwargs...)
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
