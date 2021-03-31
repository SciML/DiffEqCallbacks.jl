# Default test function
# Terminate when all derivatives fall below a threshold or
#   when derivatives are smaller than a fraction of state
function allDerivPass(integrator, abstol, reltol)
    if DiffEqBase.isinplace(integrator.sol.prob)
        testval = first(get_tmp_cache(integrator))
        DiffEqBase.get_du!(testval, integrator)
        if typeof(integrator.sol.prob) <: DiffEqBase.DiscreteProblem
            @. testval =  testval - integrator.u
        end
    else
        testval = get_du(integrator)
        if typeof(integrator.sol.prob) <: DiffEqBase.DiscreteProblem
            testval =  testval - integrator.u
        end
    end

    if typeof(integrator.u) <: Array
        any(abs(d) > abstol && abs(d) > reltol*abs(u) for (d,abstol, reltol, u) =
           zip(testval, Iterators.cycle(abstol), Iterators.cycle(reltol), integrator.u)) && (return false)
    else
        any((abs.(testval) .> abstol) .& (abs.(testval) .> reltol .* abs.(integrator.u))) && (return false)
    end
    return true
end

# Allow user-defined tolerances and test functions but use sensible defaults
# test function must take integrator, time, followed by absolute
#   and relative tolerance and return true/false
function TerminateSteadyState(abstol = 1e-8, reltol = 1e-6, test = allDerivPass)
    condition = (u, t, integrator) -> test(integrator, abstol, reltol)
    affect! = (integrator) -> terminate!(integrator)
    DiscreteCallback(condition, affect!; save_positions = (true, false))
end

export TerminateSteadyState
