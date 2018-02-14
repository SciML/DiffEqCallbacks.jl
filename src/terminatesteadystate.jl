
# Default test function
# Terminate when all derivatives fall below a threshold or
#   when derivatives are small than a fraction of state
function allDerivPass(integrator, t, abstol, reltol)
    du = integrator(t, Val{1})
    any(du .> abstol) &&  (return false)
    u = integrator(t)
    any(du .> reltol.*u) &&  (return false)
    return true
end

# Allow user-defined tolerances and test functions but use sensible defaults
# test function must take integrator, time, followed by absolute
#   and relative tolerance and return true/false
function TerminateSteadyState(abstol = 1e-8, reltol = 1e-6, test = allDerivPass)
    condition = (u, t, integrator) -> test(integrator, t, abstol, reltol)
    affect! = (integrator) -> terminate!(integrator)
    DiscreteCallback(condition, affect!)
end

export TerminateSteadyState
