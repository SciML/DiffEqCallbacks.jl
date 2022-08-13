# Default test function
# Terminate when all derivatives fall below a threshold or
#   when derivatives are smaller than a fraction of state
function allDerivPass(integrator, abstol, reltol, min_t)
    if DiffEqBase.isinplace(integrator.sol.prob)
        testval = first(get_tmp_cache(integrator))
        DiffEqBase.get_du!(testval, integrator)
        if typeof(integrator.sol.prob) <: DiffEqBase.DiscreteProblem
            @. testval = testval - integrator.u
        end
    else
        testval = get_du(integrator)
        if typeof(integrator.sol.prob) <: DiffEqBase.DiscreteProblem
            testval = testval - integrator.u
        end
    end

    if typeof(integrator.u) <: Array
        any(abs(d) > abstol && abs(d) > reltol * abs(u)
            for (d, abstol, reltol, u) in zip(testval, Iterators.cycle(abstol),
                                              Iterators.cycle(reltol), integrator.u)) &&
            (return false)
    else
        any((abs.(testval) .> abstol) .& (abs.(testval) .> reltol .* abs.(integrator.u))) &&
            (return false)
    end

    if min_t === nothing
        return true
    else
        return integrator.t >= min_t
    end
end

# Allow user-defined tolerances and test functions but use sensible defaults
# test function must take integrator, time, followed by absolute
#   and relative tolerance and return true/false
"""
```julia
TerminateSteadyState(abstol = 1e-8, reltol = 1e-6, test = allDerivPass; min_t = nothing)
```

`TerminateSteadyState` can be used to solve the problem for the steady-state
by running the solver until the derivatives of the problem converge to 0 or
`tspan[2]` is reached. This is an alternative approach to root finding (see
the [Steady State Solvers](@ref) section).

## Arguments

- `abstol` and `reltol` are the absolute and relative tolerance, respectively.
  These tolerances may be specified as scalars or as arrays of the same length
  as the states of the problem.
- `test` represents the function that evaluates the condition for termination. The default
  condition is that all derivatives should become smaller than `abstol` and the states times
 `reltol`. The user can pass any other function to implement a different termination condition.
  Such function should take four arguments: `integrator`, `abstol`, `reltol`, and `min_t`.

## Keyword Arguments

- `min_t` specifies an optional minimum `t` before the steady state calculations are allowed
  to terminate.
"""
function TerminateSteadyState(abstol = 1e-8, reltol = 1e-6, test = allDerivPass;
                              min_t = nothing)
    condition = (u, t, integrator) -> test(integrator, abstol, reltol, min_t)
    affect! = (integrator) -> terminate!(integrator)
    DiscreteCallback(condition, affect!; save_positions = (true, false))
end

export TerminateSteadyState
