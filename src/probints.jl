# Read the integrator's scalar error estimate. Older integrators stored it as the
# `EEst` field directly; current OrdinaryDiffEqCore moved it onto the controller
# cache (`integrator.controller_cache.EEst`, exposed there as `get_EEst`). Reading
# it this way keeps DiffEqCallbacks free of an OrdinaryDiffEqCore dependency while
# supporting both integrator layouts.
function _integrator_EEst(integrator)
    if hasproperty(integrator, :EEst)
        return getproperty(integrator, :EEst)
    else
        return integrator.controller_cache.EEst
    end
end

struct ProbIntsCache{T}
    σ::T
    order::Int
end
function (p::ProbIntsCache)(integrator)
    return integrator.u .= integrator.u .+ p.σ * sqrt(integrator.dt^(2 * p.order)) * randn(size(integrator.u))
end

"""
    ProbIntsUncertainty(σ, order, save = true) -> DiscreteCallback

The [ProbInts](https://arxiv.org/abs/1506.04592) method for uncertainty quantification
involves the transformation of an ODE into an associated SDE where the noise is related to
the timesteps and the order of the algorithm.

# Arguments

  - `σ`: noise scaling factor. It is recommended that `σ` is representative of the size
    of the errors in a single step of the equation. If such a value is unknown, it can be
    estimated automatically in adaptive time-stepping algorithms with
    [`AdaptiveProbIntsUncertainty`](@ref).
  - `order::Integer`: order of the ODE solver algorithm.
  - `save::Bool = true`: whether to save immediately before applying the random
    perturbation.

# Returns

  - `DiscreteCallback`: a callback that perturbs the array state after every accepted step.
    The state must support in-place broadcasting and have a defined `size`.

# References

Conrad P., Girolami M., Särkkä S., Stuart A., Zygalakis. K, Probability
Measures for Numerical Solutions of Differential Equations, arXiv:1506.04592

# Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

prob = ODEProblem((u, p, t) -> -u, [1.0], (0.0, 1.0))
cb = ProbIntsUncertainty(0.01, 5)

sol = solve(prob, Tsit5(); callback = cb)
```
"""
function ProbIntsUncertainty(σ, order, save = true)
    affect! = ProbIntsCache(σ, order)
    condition = (t, u, integrator) -> true
    save_positions = (save, false)
    return DiscreteCallback(condition, affect!, save_positions = save_positions)
end

struct AdaptiveProbIntsCache
    order::Int
end
function (p::AdaptiveProbIntsCache)(integrator)
    return integrator.u .= integrator.u .+ _integrator_EEst(integrator) * sqrt(integrator.dt^(2 * p.order)) * randn(size(integrator.u))
end

"""
    AdaptiveProbIntsUncertainty(order, save = true) -> DiscreteCallback

The [ProbInts](https://arxiv.org/abs/1506.04592) method for uncertainty quantification
involves the transformation of an ODE into an associated SDE where the noise is related to
the timesteps and the order of the algorithm.

`AdaptiveProbIntsUncertainty` is a more automated form of `ProbIntsUncertainty` which
uses the error estimate from within adaptive time stepping methods to estimate `σ` at
every step.

# Arguments

  - `order::Integer`: order of the ODE solver algorithm.
  - `save::Bool = true`: whether to save immediately before applying the random
    perturbation.

# Returns

  - `DiscreteCallback`: a callback that scales its array-state perturbation by the adaptive
    integrator's current local error estimate after every accepted step.

# Throws

  - An error during callback execution if the integrator does not expose a local error
    estimate through `EEst` or its controller cache.

# References

Conrad P., Girolami M., Särkkä S., Stuart A., Zygalakis. K, Probability
Measures for Numerical Solutions of Differential Equations, arXiv:1506.04592

# Examples

```julia
using DiffEqCallbacks, OrdinaryDiffEq

prob = ODEProblem((u, p, t) -> -u, [1.0], (0.0, 1.0))
cb = AdaptiveProbIntsUncertainty(5)

sol = solve(prob, Tsit5(); callback = cb)
```
"""
function AdaptiveProbIntsUncertainty(order, save = true)
    affect! = AdaptiveProbIntsCache(order)
    condition = (t, u, integrator) -> true
    save_positions = (save, false)
    return DiscreteCallback(condition, affect!, save_positions = save_positions)
end

export ProbIntsUncertainty, AdaptiveProbIntsUncertainty
