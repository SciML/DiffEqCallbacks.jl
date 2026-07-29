mutable struct AutoAbstolAffect{T}
    curmax::T
end
# Now make `affect!` for this:
function (p::AutoAbstolAffect)(integrator)
    if p.curmax isa AbstractArray
        @. p.curmax = max(p.curmax, abs(integrator.u))
    else
        p.curmax = max(p.curmax, maximum(abs.(integrator.u)))
    end

    if integrator.opts.abstol isa AbstractArray
        integrator.opts.abstol .= p.curmax .* integrator.opts.reltol
    else
        integrator.opts.abstol = p.curmax .* integrator.opts.reltol
    end

    return derivative_discontinuity!(integrator, false)
end

function AutoAbstol_initialize(cb, u, t, integrator)
    if cb.affect!.curmax == zero(integrator.opts.abstol)
        cb.affect!.curmax = integrator.opts.abstol
    end
    return derivative_discontinuity!(integrator, false)
end

"""
    AutoAbstol(save = true; init_curmax = 0.0) -> DiscreteCallback

Construct a callback that updates `integrator.opts.abstol` after every accepted step to the
largest magnitude observed in the state so far, multiplied by `integrator.opts.reltol`.

# Arguments

  - `save::Bool = true`: save the solution immediately before the callback affect. Set this
    to `false` when another callback controls saving.

# Keywords

  - `init_curmax = 0.0`: initial maximum state magnitude. A zero value is replaced during
    initialization with the integrator's configured `abstol`; arrays update elementwise.

# Returns

  - `DiscreteCallback`: a callback that updates the absolute tolerance, then marks
    a derivative discontinuity after each affect.

# Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

f(u, p, t) = 0.5u
prob = ODEProblem(f, 1.0, (0.0, 2.0))
cb = AutoAbstol(; init_curmax = 1.0e-8)

sol = solve(prob, Tsit5(); callback = cb, reltol = 1.0e-6)
```
"""
function AutoAbstol(save = true; init_curmax = 0.0)
    affect! = AutoAbstolAffect(abs.(init_curmax))
    condition = true_condition
    save_positions = (save, false)
    return DiscreteCallback(
        condition, affect!;
        initialize = AutoAbstol_initialize,
        save_positions = save_positions
    )
end

export AutoAbstol
