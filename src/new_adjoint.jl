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

mutable struct SavingIntegrandAffect{IntegrandFunc, tType, integrandType, saveatType, saveatCacheType}
    integrand_func::IntegrandFunc
    integrand_values::IntegrandValues{tType, integrandType}
    saveat::saveatType
    saveat_cache::saveatCacheType
    save_everystep::Bool
    save_start::Bool
    save_end::Bool
    saveiter::Int
end

function (affect!::SavingIntegrandAffect)(integrator, force_save = false)
    just_saved = false
    if !just_saved &&
       affect!.save_everystep || force_save ||
       (affect!.save_end && integrator.t == integrator.sol.prob.tspan[end])
        affect!.saveiter += 1
        copyat_or_push!(affect!.integrand_values.t, affect!.saveiter, integrator.t)
        copyat_or_push!(affect!.integrand_values.integrand, affect!.saveiter,
                        affect!.integrand_func(integrator.u, integrator.t, integrator),
                        Val{false})
    end
    u_modified!(integrator, false)
end

function integrand_saving_initialize(cb, u, t, integrator)
    if cb.affect!.saveiter != 0
        if integrator.tdir > 0
            cb.affect!.saveat = BinaryMinHeap(cb.affect!.saveat_cache)
        else
            cb.affect!.saveat = BinaryMaxHeap(cb.affect!.saveat_cache)
        end
        cb.affect!.saveiter = 0
    end
    cb.affect!.save_start && cb.affect!(integrator)
end

"""
```julia
NewAdjointCallback(integrand_func, integrand_values::IntegrandValues;
               saveat=Vector{eltype(integrand_values.t)}(),
               save_everystep=isempty(saveat),
               save_start = true,
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

## Keyword Arguments

- `saveat` mimics `saveat` in `solve` from `solve`.
- `save_everystep` mimics `save_everystep` from `solve`.
- `save_start` mimics `save_start` from `solve`.
- `tdir` should be `sign(tspan[end]-tspan[1])`. It defaults to `-1` and should
  be adapted if `tspan[1] > tspan[end]`.

The outputted values are saved into `saved_values`. Time points are found via
`saved_values.t` and the values are `saved_values.saveval`.
"""
function NewAdjointCallback(integrand_func, integrand_values::IntegrandValues;
                        saveat = Vector{eltype(integrand_values.t)}(),
                        save_everystep = isempty(saveat),
                        save_start = save_everystep || isempty(saveat) || saveat isa Number,
                        save_end = save_everystep || isempty(saveat) || saveat isa Number,
                        tdir = -1)
    # saveat conversions, see OrdinaryDiffEq.jl -> integrators/type.jl
    saveat_vec = collect(saveat)
    if tdir > 0
        saveat_internal = BinaryMinHeap(saveat_vec)
    else
        saveat_internal = BinaryMaxHeap(saveat_vec)
    end
    affect! = SavingIntegrandAffect(integrand_func, integrand_values, saveat_internal, saveat_vec,
                           save_everystep, save_start, save_end, 0)
    condtion = (u, t, integrator) -> true
    DiscreteCallback(condtion, affect!;
                     initialize = integrand_saving_initialize,
                     save_positions = (false, false))
end

export NewAdjointCallback, IntegrandValues
