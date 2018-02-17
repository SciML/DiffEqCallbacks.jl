
# Check if a cache for du can be used inside allDerivPass
Base.@pure usecache(z) = true
Base.@pure usecache(z::Number) = false
Base.@pure usecache(z::StaticArrays.SArray) = false

# Default test function
# Terminate when all derivatives fall below a threshold or
#   when derivatives are smaller than a fraction of state
function allDerivPass(integrator, abstol, reltol)
    if usecache(integrator.u)
        du = first(get_tmp_cache(integrator))
        DiffEqBase.get_du!(du, integrator)
    else
        du = get_du(integrator)
    end
    any(abs(d) > abstol && abs(d) > reltol*abs(u) for (d,abstol, reltol, u) =
           zip(du, cycle(abstol), cycle(reltol), integrator.u)) && (return false)
    return true
end

# Allow user-defined tolerances and test functions but use sensible defaults
# test function must take integrator, time, followed by absolute
#   and relative tolerance and return true/false
function TerminateSteadyState(abstol = 1e-8, reltol = 1e-6, test = allDerivPass)
    condition = (u, t, integrator) -> test(integrator, abstol, reltol)
    affect! = (integrator) -> terminate!(integrator)
    DiscreteCallback(condition, affect!)
end

export TerminateSteadyState
