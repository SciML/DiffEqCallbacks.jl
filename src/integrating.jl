"""
    IntegrandValues{integrandType}

A struct used to save values of the integrand values in `integrand::Vector{integrandType}`.
"""
struct IntegrandValues{integrandType}
    integrand::Vector{integrandType}
end

"""
    IntegrandValues(ntegrandType::DataType)

Return `IntegrandValues{integrandType}` with empty storage vectors.
"""
function IntegrandValues(::Type{integrandType}) where {integrandType}
    IntegrandValues{integrandType}(Vector{integrandType}())
end

function Base.show(io::IO, integrand_values::IntegrandValues)
    integrandType = eltype(integrand_values.integrand)
    print(io, "IntegrandValues{integrandType=", integrandType, "}",
          "\nintegrand:\n", integrand_values.integrand)
end

mutable struct SavingIntegrandAffect{IntegrandFunc, integrandType, gaussPointsType, gaussWeightsType}
    integrand_func::IntegrandFunc
    integrand_values::IntegrandValues{integrandType}
    gauss_points::gaussPointsType
    gauss_weights::gaussWeightsType
end

function (affect!::SavingIntegrandAffect)(integrator)
    integral = zeros(eltype(eltype(affect!.integrand_values.integrand)),length(integrator.p))
    for i in 1:length(affect!.gauss_points)
        t_temp = ((integrator.t-integrator.tprev)/2)*affect!.gauss_points[i]+(integrator.t+integrator.tprev)/2
        integral .+= affect!.gauss_weights[i]*affect!.integrand_func(integrator(t_temp), t_temp, integrator)
    end
    integral *= -(integrator.t-integrator.tprev)/2
    push!(affect!.integrand_values.integrand, integral)
    u_modified!(integrator, false)
end


"""
```julia
IntegratingCallback(integrand_func, integrand_values::IntegrandValues, gauss_points = Vector{eltype(integrand_values.t)}())
```

The saving callback lets you define a function `integrand_func(u, t, integrator)` which
returns lambda*df/dp + dg/dp for calculating the adjoint integral.

## Arguments

- `integrand_func(u, t, integrator)` returns the quantity in the integral for computing dG/dp.
  Note that this should allocate the output (not as a view to `u`).
- `integrand_values::IntegrandValues` is the types that `integrand_func` will return, i.e.
  `integrand_func(t, u, integrator)::integrandType`. It's specified via
  `IntegrandValues(integrandType)`, i.e. give the type 
  that `integrand_func` will output (or higher compatible type).
- `gauss_points` are the Gauss-Legendre points, scaled for the correct limits of integration

The outputted values are saved into `integrand_values`. Time points are found via
`integrand_values.t` and the values are `integrand_values.integrand`.
"""
# need to make take in the Gaussian quadrature points (similar to saveat)
function IntegratingCallback(integrand_func, integrand_values::IntegrandValues, gauss_points, gauss_weights)
    gauss_points_vec = collect(gauss_points)
    gauss_weights_vec = collect(gauss_weights)
    affect! = SavingIntegrandAffect(integrand_func, integrand_values, gauss_points_vec, gauss_weights_vec)
    condition = (u, t, integrator) -> true
    DiscreteCallback(condition, affect!)
end

export IntegratingCallback, IntegrandValues
