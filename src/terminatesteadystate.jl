# Default test function
# Terminate when all derivatives fall below a threshold or
#   when derivatives are smaller than a fraction of state
function allDerivPass(integrator, abstol, reltol, min_t)
    # Early exit
    if min_t !== nothing && integrator.t < min_t
        return false
    end

    testval = if integrator.sol.prob isa DiscreteProblem
        if DiffEqBase.isinplace(integrator.sol.prob)
            testval = first(get_tmp_cache(integrator))
            @. testval = integrator.u - integrator.uprev
        else
            testval = integrator.u .- integrator.uprev
        end
    else
        if DiffEqBase.isinplace(integrator.sol.prob)
            testval = first(get_tmp_cache(integrator))
            DiffEqBase.get_du!(testval, integrator)
            if integrator.sol.prob isa DiffEqBase.DiscreteProblem
                @. testval = testval - integrator.u
            end
            testval
        else
            testval = get_du(integrator)
            if integrator.sol.prob isa DiffEqBase.DiscreteProblem
                testval = testval - integrator.u
            end
            testval
        end
    end

    if integrator.u isa Array
        return all(abs(d) <= max(abstol, reltol * abs(u))
        for (d, abstol, reltol, u) in zip(testval, Iterators.cycle(abstol),
            Iterators.cycle(reltol), integrator.u))
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
    TerminateSteadyState(abstol = 1e-8, reltol = 1e-6, test = allDerivPass; min_t = nothing,
        wrap_test::Val = Val(true))

`TerminateSteadyState` can be used to solve the problem for the steady-state
by running the solver until the derivatives of the problem converge to 0 or
`tspan[2]` is reached. This is an alternative approach to root finding (see
the [Steady State Solvers](https://docs.sciml.ai/DiffEqDocs/stable/solvers/steady_state_solve/) section).

## Arguments

  - `abstol` and `reltol` are the absolute and relative tolerance, respectively.
    These tolerances may be specified as scalars or as arrays of the same length
    as the states of the problem.
  - `test` represents the function that evaluates the condition for termination. The default
    condition is that all derivatives should become smaller than `abstol` or the states times
    `reltol`. The user can pass any other function to implement a different termination condition.
    Such function should take four arguments: `integrator`, `abstol`, `reltol`, and `min_t`.
  - `wrap_test` can be set to `Val(false)`, in which case `test` must have the definition
    `test(u, t, integrator)`. Otherwise, `test` must have the definition
    `test(integrator, abstol, reltol, min_t)`.

## Keyword Arguments

  - `min_t` specifies an optional minimum `t` before the steady state calculations are allowed
    to terminate.
"""
function TerminateSteadyState(abstol = 1e-8, reltol = 1e-6, test::T = allDerivPass;
        min_t = nothing, wrap_test::Val{WT} = Val(true)) where {T, WT}
    condition = if WT
        WrappedTest(test, abstol, reltol, min_t)
    else
        test
    end
    DiscreteCallback(condition, terminate!; save_positions = (true, false))
end

export TerminateSteadyState
