"""
    IntegrandValues{tType<:Real, integrandType}

A struct used to save values of the time in `t::Vector{tType}` and
integrand values in `integrand::Vector{integrandType}`.
"""
struct IntegrandValues{tType, integrandType}
    t::Vector{tType}
    integrand::Vector{integrandType}
end

"""
    IntegrandValues(tType::DataType, integrandType::DataType)

Return `IntegrandValues{tType, integrandType}` with empty storage vectors.
"""
function IntegrandValues(::Type{tType}, ::Type{integrandType}) where {tType, integrandType}
    IntegrandValues{tType, integrandType}(Vector{tType}(), Vector{integrandType}())
end

function Base.show(io::IO, saved_values::IntegrandValues)
    tType = eltype(saved_values.t)
    savevalType = eltype(saved_values.integrand)
    print(io, "IntegrandValues{tType=", tType, ", integrandType=", savevalType, "}",
          "\nt:\n", saved_values.t, "\nintegrand:\n", saved_values.integrand)
end

mutable struct SavingIntegrandAffect{IntegrandFunc, tType, integrandType, gaussPointsType, gaussPointsCacheType}
    integrand_func::IntegrandFunc
    integrand_values::IntegrandValues{tType, integrandType}
    gauss_points::gaussPointsType
    gauss_points_cache::gaussPointsCacheType
    gaussiter::Int
end



function (affect!::SavingIntegrandAffect)(integrator, force_save = false)
    just_saved = false
    while !isempty(affect!.gauss_points) &&
        integrator.tdir * first(affect!.gauss_points) <= integrator.tdir * integrator.t # Perform saveat
        affect!.gaussiter += 1
        curt = pop!(affect!.gauss_points) # current time
        if curt != integrator.t # If <t, interpolate
            if integrator isa SciMLBase.AbstractODEIntegrator
                # Expand lazy dense for interpolation
                DiffEqBase.addsteps!(integrator)
            end
            if !DiffEqBase.isinplace(integrator.sol.prob)
                curu = integrator(curt)
            else
                curu = first(get_tmp_cache(integrator))
                integrator(curu, curt) # inplace since save_func allocates
            end
            copyat_or_push!(affect!.integrand_values.t, affect!.gaussiter, curt)
            copyat_or_push!(affect!.integrand_values.integrand, affect!.gaussiter,
                            affect!.integrand_func(curu, curt, integrator), Val{false})
        else # ==t, just save
            just_saved = true
            copyat_or_push!(affect!.integrand_values.t, affect!.gaussiter, integrator.t)
            copyat_or_push!(affect!.integrand_values.integrand, affect!.gaussiter,
                            affect!.integrand_func(integrator.u, integrator.t, integrator),
                            Val{false})
        end
    end
    u_modified!(integrator, false)
end

function integrand_saving_initialize(cb, u, t, integrator)
    if cb.affect!.gaussiter != 0
        if integrator.tdir > 0
            cb.affect!.gauss_points = BinaryMinHeap(cb.affect!.gauss_points_cache)
        else
            cb.affect!.gauss_points = BinaryMaxHeap(cb.affect!.gauss_points_cache)
        end
        cb.affect!.gaussiter = 0
    end
end

"""
```julia
IntegratingCallback(integrand_func, integrand_values::IntegrandValues, gauss_points = Vector{eltype(integrand_values.t)}();
               tdir=1)
```

The saving callback lets you define a function `integrand_func(u, t, integrator)` which
returns lambda*df/dp + dg/dp for calculating the adjoint integral.

## Arguments

- `integrand_func(u, t, integrator)` returns the quantity in the integral for computing dG/dp.
  Note that this should allocate the output (not as a view to `u`).
- `integrand_values::IntegrandValues` is the types that `integrand_func` will return, i.e.
  `integrand_func(t, u, integrator)::integrandType`. It's specified via
  `IntegrandValues(typeof(t),integrandType)`, i.e. give the type for time and the
  type that `integrand_func` will output (or higher compatible type).
- `gauss_points` are the Gauss-Legendre points, scaled for the correct limits of integration

## Keyword Arguments

- `tdir` should be `sign(tspan[end]-tspan[1])`. It defaults to `-1` since the adjoint solve is reverse.

The outputted values are saved into `integrand_values`. Time points are found via
`integrand_values.t` and the values are `integrand_values.integrand`.
"""
# need to make take in the Gaussian quadrature points (similar to saveat)
function IntegratingCallback(integrand_func, integrand_values::IntegrandValues, gauss_points = Vector{eltype(integrand_values.t)}();
                        tdir = -1)
    gauss_points_vec = collect(gauss_points)
    if tdir > 0
        gauss_points_internal = BinaryMinHeap(gauss_points_vec)
    else
        gauss_points_internal = BinaryMaxHeap(gauss_points_vec)
    end
    affect! = SavingIntegrandAffect(integrand_func, integrand_values, gauss_points_internal, gauss_points_vec, 0)
    condition = (u, t, integrator) -> true
    DiscreteCallback(condition, affect!;
                     initialize = integrand_saving_initialize,
                     save_positions = (false, false))
end

export IntegratingCallback, IntegrandValues
