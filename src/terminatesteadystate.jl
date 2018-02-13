
# Default test function
# Terminate when all derivatives fall below a threshold, with early exit
function allDerivPass(derivs, tol)
    for der in derivs
        der > tol && (return false)
    end
    return true
end

# Allow user-defined tolerances and test functions but use sensible defaults
# test function must take derivatives as first argument, tolerance as second,
#   return true/false
function TerminateSteadyState(tol = 1e-8, test = allDerivPass)
    condition = (u, t, integrator) -> test(integrator(t, Val{1}), tol)
    affect! = (integrator) -> terminate!(integrator)
    DiscreteCallback(condition, affect!)
end

export TerminateSteadyState
