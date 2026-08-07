# Default test function
# Terminate when all derivatives fall below a threshold or
#   when derivatives are smaller than a fraction of state
function allDerivPass(integrator, abstol, reltol, min_t)
    # Early exit
    if min_t !== nothing && integrator.t < min_t
        return false
    end

    testval = if integrator.sol.prob isa DiscreteProblem
        if SciMLBase.isinplace(integrator.sol.prob)
            testval = first(get_tmp_cache(integrator))
            @. testval = integrator.u - integrator.uprev
        else
            testval = integrator.u .- integrator.uprev
        end
    else
        if SciMLBase.isinplace(integrator.sol.prob)
            testval = first(get_tmp_cache(integrator))
            SciMLBase.get_du!(testval, integrator)
            if integrator.sol.prob isa SciMLBase.DiscreteProblem
                @. testval = testval - integrator.u
            end
            testval
        else
            testval = get_du(integrator)
            if integrator.sol.prob isa SciMLBase.DiscreteProblem
                testval = testval - integrator.u
            end
            testval
        end
    end

    if integrator.u isa Array
        return all(
            abs(d) <= max(abstol, reltol * abs(u))
                for (d, abstol, reltol, u) in zip(
                    testval, Iterators.cycle(abstol),
                    Iterators.cycle(reltol), integrator.u
                )
        )
    else
        return all(abs.(testval) .<= max.(abstol, reltol .* abs.(integrator.u)))
    end
end

struct WrappedTest{T, A, R, M}
    test::T
    abstol::A
    reltol::R
    min_t::M
end
(f::WrappedTest)(u, t, integrator) = f.test(integrator, f.abstol, f.reltol, f.min_t)

# Allow user-defined tolerances and test functions but use sensible defaults
# test function must take integrator, time, followed by absolute
#   and relative tolerance and return true/false
"""
    TerminateSteadyState(abstol = 1.0e-8, reltol = 1.0e-6, test = allDerivPass;
        min_t = nothing, wrap_test::Val = Val(true)) -> DiscreteCallback

`TerminateSteadyState` can be used to solve the problem for the steady-state
by running the solver until the derivatives of the problem converge to 0 or
`tspan[2]` is reached. This is an alternative approach to root finding; see the
[Steady State Solvers](https://docs.sciml.ai/DiffEqDocs/stable/solvers/steady_state_solve/)
documentation.

# Arguments

  - `abstol = 1.0e-8`: absolute termination tolerance. It may be a scalar or an array with
    the same length as the state.
  - `reltol = 1.0e-6`: relative termination tolerance. It may be a scalar or an array with
    the same length as the state.
  - `test = allDerivPass`: function that evaluates the termination condition. By default,
    every derivative must be smaller than `abstol` or the corresponding state magnitude
    times `reltol`. A custom wrapped test must accept `integrator`, `abstol`, `reltol`, and
    `min_t`.

# Keywords

  - `min_t = nothing`: optional minimum integration time before termination is allowed.
  - `wrap_test::Val = Val(true)`: with `Val(true)`, call `test` as
    `test(integrator, abstol, reltol, min_t)`. With `Val(false)`, use `test` directly as the
    callback condition `test(u, t, integrator)`.

# Returns

  - `DiscreteCallback`: a callback that terminates the integrator when `test` returns
    `true`.

# Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

f(u, p, t) = 1 - u
prob = ODEProblem(f, 0.0, (0.0, 100.0))
cb = TerminateSteadyState(1.0e-8, 1.0e-8)

sol = solve(prob, Tsit5(); callback = cb)
```
"""
function TerminateSteadyState(
        abstol = 1.0e-8, reltol = 1.0e-6, test::T = allDerivPass;
        min_t = nothing, wrap_test::Val{WT} = Val(true)
    ) where {T, WT}
    condition = if WT
        WrappedTest(test, abstol, reltol, min_t)
    else
        test
    end
    return DiscreteCallback(condition, terminate!; save_positions = (true, false))
end

export TerminateSteadyState
