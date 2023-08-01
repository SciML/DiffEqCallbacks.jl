struct ProbIntsCache{T}
    σ::T
    order::Int
end
function (p::ProbIntsCache)(integrator)
    integrator.u .= integrator.u .+
                    p.σ * sqrt(integrator.dt^(2 * p.order)) * randn(size(integrator.u))
end

"""
```julia
ProbIntsUncertainty(σ, order, save = true)
```

The [ProbInts](https://arxiv.org/abs/1506.04592) method for uncertainty quantification
involves the transformation of an ODE into an associated SDE where the noise is related to
the timesteps and the order of the algorithm.

## Arguments

  - `σ` is the noise scaling factor. It is recommended that `σ` is representative of the size
    of the errors in a single step of the equation. If such a value is unknown, it can be
    estimated automatically in adaptive time-stepping algorithms via AdaptiveProbIntsUncertainty
  - `order` is the order of the ODE solver algorithm.
  - `save` is for choosing whether this callback should control the saving behavior. Generally
    this is true unless one is stacking callbacks in a `CallbackSet`.

## References

Conrad P., Girolami M., Särkkä S., Stuart A., Zygalakis. K, Probability
Measures for Numerical Solutions of Differential Equations, arXiv:1506.04592
"""
function ProbIntsUncertainty(σ, order, save = true)
    affect! = ProbIntsCache(σ, order)
    condtion = (t, u, integrator) -> true
    save_positions = (save, false)
    DiscreteCallback(condtion, affect!, save_positions = save_positions)
end

struct AdaptiveProbIntsCache
    order::Int
end
function (p::AdaptiveProbIntsCache)(integrator)
    integrator.u .= integrator.u .+
                    integrator.EEst * sqrt(integrator.dt^(2 * p.order)) *
                    randn(size(integrator.u))
end

"""
```julia
AdaptiveProbIntsUncertainty(order, save = true)
```

The [ProbInts](https://arxiv.org/abs/1506.04592) method for uncertainty quantification
involves the transformation of an ODE into an associated SDE where the noise is related to
the timesteps and the order of the algorithm.

`AdaptiveProbIntsUncertainty` is a more automated form of `ProbIntsUncertainty` which
uses the error estimate from within adaptive time stepping methods to estimate `σ` at
every step.

## Arguments

  - `order` is the order of the ODE solver algorithm.
  - `save` is for choosing whether this callback should control the saving behavior. Generally
    this is true unless one is stacking callbacks in a `CallbackSet`.

## References

Conrad P., Girolami M., Särkkä S., Stuart A., Zygalakis. K, Probability
Measures for Numerical Solutions of Differential Equations, arXiv:1506.04592
"""
function AdaptiveProbIntsUncertainty(order, save = true)
    affect! = AdaptiveProbIntsCache(order)
    condtion = (t, u, integrator) -> true
    save_positions = (save, false)
    DiscreteCallback(condtion, affect!, save_positions = save_positions)
end

export ProbIntsUncertainty, AdaptiveProbIntsUncertainty
