"""
    SavedValues{tType<:Real, savevalType}

A struct used to save values of the time in `t::Vector{tType}` and
additional values in `saveval::Vector{savevalType}`.
"""
struct SavedValues{tType, savevalType}
    t::Vector{tType}
    saveval::Vector{savevalType}
end

"""
    SavedValues(tType::DataType, savevalType::DataType)

Return `SavedValues{tType, savevalType}` with empty storage vectors.
"""
function SavedValues(::Type{tType}, ::Type{savevalType}) where {tType, savevalType}
    SavedValues{tType, savevalType}(Vector{tType}(), Vector{savevalType}())
end

function Base.show(io::IO, saved_values::SavedValues)
    tType = eltype(saved_values.t)
    savevalType = eltype(saved_values.saveval)
    print(io, "SavedValues{tType=", tType, ", savevalType=", savevalType, "}",
        "\nt:\n", saved_values.t, "\nsaveval:\n", saved_values.saveval)
end

@recipe function plot(saved_values::SavedValues)
    DiffEqArray(saved_values.t, saved_values.saveval)
end

mutable struct SavingAffect{SaveFunc, tType, savevalType, saveatType, saveatCacheType}
    save_func::SaveFunc
    saved_values::SavedValues{tType, savevalType}
    saveat::saveatType
    saveat_cache::saveatCacheType
    save_everystep::Bool
    save_start::Bool
    save_end::Bool
    saveiter::Int
end

function (affect!::SavingAffect)(integrator, force_save = false)
    just_saved = false
    # see OrdinaryDiffEq.jl -> integrator_utils.jl, function savevalues!
    while !isempty(affect!.saveat) &&
        integrator.tdir * first(affect!.saveat) <= integrator.tdir * integrator.t # Perform saveat
        affect!.saveiter += 1
        curt = pop!(affect!.saveat) # current time
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
            copyat_or_push!(affect!.saved_values.t, affect!.saveiter, curt)
            copyat_or_push!(affect!.saved_values.saveval, affect!.saveiter,
                affect!.save_func(curu, curt, integrator), Val{false})
        else # ==t, just save
            just_saved = true
            copyat_or_push!(affect!.saved_values.t, affect!.saveiter, integrator.t)
            copyat_or_push!(affect!.saved_values.saveval, affect!.saveiter,
                affect!.save_func(integrator.u, integrator.t, integrator),
                Val{false})
        end
    end
    if !just_saved &&
       affect!.save_everystep || force_save ||
       (affect!.save_end && integrator.t == integrator.sol.prob.tspan[end])
        affect!.saveiter += 1
        copyat_or_push!(affect!.saved_values.t, affect!.saveiter, integrator.t)
        copyat_or_push!(affect!.saved_values.saveval, affect!.saveiter,
            affect!.save_func(integrator.u, integrator.t, integrator),
            Val{false})
    end
    u_modified!(integrator, false)
end

function saving_initialize(cb, u, t, integrator)
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
SavingCallback(save_func, saved_values::SavedValues;
    saveat = Vector{eltype(saved_values.t)}(),
    save_everystep = isempty(saveat),
    save_start = true,
    tdir = 1)
```

The saving callback lets you define a function `save_func(u, t, integrator)` which
returns quantities of interest that shall be saved.

## Arguments

  - `save_func(u, t, integrator)` returns the quantities which shall be saved.
    Note that this should allocate the output (not as a view to `u`).
  - `saved_values::SavedValues` is the types that `save_func` will return, i.e.
    `save_func(t, u, integrator)::savevalType`. It's specified via
    `SavedValues(typeof(t),savevalType)`, i.e. give the type for time and the
    type that `save_func` will output (or higher compatible type).

## Keyword Arguments

  - `saveat` mimics `saveat` in `solve` from `solve`.
  - `save_everystep` mimics `save_everystep` from `solve`.
  - `save_start` mimics `save_start` from `solve`.
  - `tdir` should be `sign(tspan[end]-tspan[1])`. It defaults to `1` and should
    be adapted if `tspan[1] > tspan[end]`.

The outputted values are saved into `saved_values`. Time points are found via
`saved_values.t` and the values are `saved_values.saveval`.
"""
function SavingCallback(save_func, saved_values::SavedValues;
    saveat = Vector{eltype(saved_values.t)}(),
    save_everystep = isempty(saveat),
    save_start = save_everystep || isempty(saveat) || saveat isa Number,
    save_end = save_everystep || isempty(saveat) || saveat isa Number,
    tdir = 1)
    # saveat conversions, see OrdinaryDiffEq.jl -> integrators/type.jl
    saveat_vec = collect(saveat)
    if tdir > 0
        saveat_internal = BinaryMinHeap(saveat_vec)
    else
        saveat_internal = BinaryMaxHeap(saveat_vec)
    end
    affect! = SavingAffect(save_func, saved_values, saveat_internal, saveat_vec,
        save_everystep, save_start, save_end, 0)
    condtion = (u, t, integrator) -> true
    DiscreteCallback(condtion, affect!;
        initialize = saving_initialize,
        save_positions = (false, false))
end

export SavingCallback, SavedValues
