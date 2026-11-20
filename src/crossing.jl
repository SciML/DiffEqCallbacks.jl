mutable struct CrossingState{X, A}
    const x::X
    const affect!::A
    const right::Bool
    const initial::Bool
    side::Int       # tracked sign of `x`
    arrival::Int    # side `x` reached zero from; 0 if it started at zero
    seen::Int       # sign of `x` at the latest evaluation off `side`
    pending::Bool   # the last tick was located before `x` took its new sign
    tclear::Any     # first time at which `x` is known to have that sign
    undo::Int       # `side` before the pending tick
    undo_arrival::Int
end

# A NaN `x` leaves the tracked sign unchanged
function signof(state::CrossingState, u, t, integrator)
    x = state.x(u, t, integrator)
    return x > 0 ? 1 : x < 0 ? -1 : isnan(x) ? state.side : 0
end

# Between a pending tick and `tclear`, `x` may still show its old sign; the conditions
# then report the new one, so the next step neither re-detects the tick nor misses roots.
crossing_suppressed(state::CrossingState, t, integrator) =
    state.pending && integrator.tdir * (t - state.tclear) <= 0

# Record that the sign of `x` became `next`, and tick if that kind of change is a tick.
function crossing_transition!(state::CrossingState, integrator, next)
    prev = state.side
    state.side = next
    tick = if iszero(prev)
        iszero(state.arrival) || (state.right && next == state.arrival)
    else
        iszero(next) && (state.arrival = prev)
        true
    end
    return if tick
        state.affect!(integrator, prev, next)
    else
        derivative_discontinuity!(integrator, false)
    end
end

# Live while `x` is nonzero. Never returns zero, so that a landing is bracketed and
# root-found like a crossing. A NaN `x` returns NaN, which is no sign change.
struct CrossingSideCondition{S}
    state::S
end
function (c::CrossingSideCondition)(u, t, integrator)
    state = c.state
    iszero(state.side) && return 1.0
    # Twice the sign, unlike the off-side value at an exact zero (see below)
    crossing_suppressed(state, t, integrator) && return 2 * float(state.side)
    x = state.x(u, t, integrator)
    (x * state.side > 0 || isnan(x)) && return float(x)
    # Off `side` already at the start of the step (a change made by a callback after
    # `CrossingSync`): locate it just after the start rather than bracket from the wrong side
    t == integrator.tprev && return float(state.side)
    state.seen = x > 0 ? 1 : x < 0 ? -1 : 0
    # Differs from the live value `x` by `oneunit(x)` at the same point, so that DiffEqBase
    # does not mistake the next step's start for the root it just handled
    return -(oneunit(x) + abs(x)) * state.side
end
function (c::CrossingSideCondition)(cb, u, t, integrator)
    state = c.state
    state.side = signof(state, u, t, integrator)
    state.arrival = 0
    state.seen = 0
    state.pending = false
    return if state.initial
        state.affect!(integrator, 0, state.side)
    else
        derivative_discontinuity!(integrator, false)
    end
end

# Live while `x` is exactly zero. Its sign change is `x` becoming nonzero.
struct CrossingZeroCondition{S}
    state::S
end
function (c::CrossingZeroCondition)(u, t, integrator)
    state = c.state
    !iszero(state.side) && return 1.0
    crossing_suppressed(state, t, integrator) && return -1.0
    x = state.x(u, t, integrator)
    (iszero(x) || t == integrator.tprev) && return -1.0
    isnan(x) && return float(x)
    state.seen = x > 0 ? 1 : -1
    return 1.0
end

crossing_rightfloat(t, integrator) = integrator.tdir > 0 ? nextfloat(t) : prevfloat(t)

struct CrossingAffect{S}
    state::S
end
function (a::CrossingAffect)(integrator)
    state = a.state
    actual = signof(state, integrator.u, integrator.t, integrator)
    if state.pending && integrator.tdir *
            (integrator.t - crossing_rightfloat(state.tclear, integrator)) <= 0
        # Right past a pending tick, `x` does not have the sign it announced, e.g. because
        # `affect!` reversed the motion: retract the tick, and report only a change from the
        # sign that `x` kept.
        state.pending = false
        state.side = state.undo
        state.arrival = state.undo_arrival
        return if actual == state.side
            derivative_discontinuity!(integrator, false)
        else
            crossing_transition!(state, integrator, actual)
        end
    end
    # A left root-find stops where `x` still has its old sign: take the new one from the
    # other end of the bracket.
    next = actual == state.side ? state.seen : actual
    tclear = crossing_rightfloat(integrator.t, integrator)
    if iszero(next) && !iszero(state.side)
        # `x` may be zero at a single float in passing: look one float beyond the zero,
        # unless that is past the end of the solve.
        t = iszero(actual) ? integrator.t : tclear
        t = crossing_rightfloat(t, integrator)
        # The tstops end at the final time, also after `reinit!(integrator; tf)`
        tstops = get_tstops(integrator)
        tdir_tend = isempty(tstops) ? integrator.tdir * integrator.t :
            get_tstops_max(integrator)
        if integrator.tdir * t <= tdir_tend &&
                signof(state, integrator(t), t, integrator) == -state.side
            next = -state.side
            tclear = t
        end
    end
    if actual != next
        state.pending = true
        state.tclear = tclear
        state.undo = state.side
        state.undo_arrival = state.arrival
    end
    return crossing_transition!(state, integrator, next)
end

# Picks up sign changes the continuous callbacks did not make, e.g. another callback
# setting `x` to zero.
struct CrossingSync{S}
    state::S
end
function (c::CrossingSync)(u, t, integrator)
    state = c.state
    state.pending && return !crossing_suppressed(state, t, integrator)
    return signof(state, u, t, integrator) != state.side
end
function (c::CrossingSync)(integrator)
    state = c.state
    state.pending = false
    next = signof(state, integrator.u, integrator.t, integrator)
    return if next == state.side
        derivative_discontinuity!(integrator, false)
    else
        crossing_transition!(state, integrator, next)
    end
end

"""
    CrossingCallback(x, affect!; right = true, initial = false, kwargs...) -> CallbackSet

Construct a callback that ticks when the sign of `x(u, t, integrator)` changes, including
when `x` reaches exactly zero and stays there; which departures from zero tick is listed
under "Kinds of tick". This is what a model needs to switch between sliding and static
friction, where the relative velocity sits at exactly zero for a while.

# Arguments

  - `x`: a function `x(u, t, integrator)` returning the scalar whose sign is tracked.
  - `affect!`: a function `affect!(integrator, prev, next)` called at each tick, where
    `prev` and `next` are the sign of `x` before and after the tick, each `-1`, `0` or `1`.

# Keywords

  - `right::Bool = true`: locate each tick just after the root (`RightRootFind`), where
    `x` already has its new sign. With `right = false` the tick is just before the root
    (`LeftRootFind`), where `x` still has its old sign; `next` is the sign it is about to
    take. If `affect!` keeps `x` from taking it, e.g. by reversing the motion, the tick is
    retracted without a call, so the next tick's `prev` is the sign `x` kept.
  - `initial::Bool = false`: also tick at the initialization, with `prev = 0`.
  - `kwargs...`: keyword arguments forwarded to the two `ContinuousCallback`s, for
    example `interp_points` or `save_positions`. `initialize`, `rootfind`, `affect_neg!`
    and `idxs` are set by the callback. An `abstol` must stay below `oneunit(x)`.

# Kinds of tick

| Kind | `x` | `(prev, next)` | Ticks |
|---|---|---|---|
| Crossing | passes through zero | `(∓1, ±1)` | always |
| Landing | reaches exactly zero and stays | `(±1, 0)` | always |
| Return | leaves zero back to the side it arrived from | `(0, ±1)` | with `right = true` |
| Continuation | leaves zero onward to the other side | | never |
| Start from zero | is exactly zero at the initialization, then becomes nonzero | `(0, ±1)` | always |
| Initialization | has its initial value | `(0, sign(x))` | with `initial = true` |

The integrator still stops where a continuation leaves zero, without calling `affect!`.
An `x` that is exactly zero at only one representable time on its way through is a
crossing, not a landing; with `right = true` that tick is at the zero itself. Reaching
zero at the end of the solve is a landing.

# Returns

  - `CallbackSet`: two `ContinuousCallback`s, one live while `x` is nonzero and one live
    while it is exactly zero, and a `DiscreteCallback` that keeps them in step with sign
    changes made by other callbacks. Such a change ticks at the time it is made if this
    callback comes after the one making it in the `CallbackSet`, and otherwise just after
    it, at the start of the next step.

The callback holds the sign of `x` between steps, so one instance must not be shared by
solves running at the same time.

# Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

ticks = Tuple{Float64, Int, Int}[]
cb = CrossingCallback((u, t, integrator) -> u[1],
    (integrator, prev, next) -> push!(ticks, (integrator.t, prev, next)))

prob = ODEProblem((u, p, t) -> [cos(t)], [0.0], (0.0, 7.0))
sol = solve(prob, Tsit5(); callback = cb)
```
"""
function CrossingCallback(x, affect!; right = true, initial = false, kwargs...)
    for k in (:initialize, :rootfind, :affect_neg!, :idxs)
        haskey(kwargs, k) &&
            throw(ArgumentError("`$k` is set by `CrossingCallback` and cannot be passed to it"))
    end
    state = CrossingState(x, affect!, right, initial, 0, 0, 0, false, nothing, 0, 0)
    rootfind = right ? SciMLBase.RightRootFind : SciMLBase.LeftRootFind
    side = CrossingSideCondition(state)
    tick! = CrossingAffect(state)
    return CallbackSet(
        ContinuousCallback(side, tick!; initialize = side, rootfind, kwargs...),
        ContinuousCallback(CrossingZeroCondition(state), tick!; rootfind, kwargs...),
        DiscreteCallback(CrossingSync(state), CrossingSync(state); save_positions = (false, false))
    )
end

export CrossingCallback
