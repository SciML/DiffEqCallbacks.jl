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
    saveat_cache = cb.affect!.saveat_cache
    if cb.affect!.saveiter != 0 || saveat_cache isa Number
        tspan = integrator.sol.prob.tspan
        saveat_cache = cb.affect!.saveat_cache
        if saveat_cache isa Number
            step = saveat_cache
            t0, tf = tspan
            if !cb.affect!.save_start
                t0 += step
            end
            if !cb.affect!.save_end
                tf -= step
            end
            saveat_vec = range(t0, tf; step)
            # avoid saving end twice
            if tspan[end] == last(saveat_vec)
                cb.affect!.save_end = false
            end
        else
            saveat_vec = saveat_cache
        end
        if integrator.tdir > 0
            cb.affect!.saveat = BinaryMinHeap(saveat_vec)
        else
            cb.affect!.saveat = BinaryMaxHeap(saveat_vec)
        end
        cb.affect!.saveiter = 0
    end
    cb.affect!.save_start && cb.affect!(integrator)
    u_modified!(integrator, false)
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
  - `save_end` mimics `save_end` from `solve`.
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
    if saveat isa Number
        # expand to range using tspan in saving_initialize
        saveat_cache = saveat
        saveat_heap = fill(saveat, 0)
    else
        saveat_heap = saveat_cache = collect(saveat)
    end

    if tdir > 0
        saveat_internal = BinaryMinHeap(saveat_heap)
    else
        saveat_internal = BinaryMaxHeap(saveat_heap)
    end
    affect! = SavingAffect(save_func, saved_values, saveat_internal, saveat_cache,
        save_everystep, save_start, save_end, 0)
    condition = (u, t, integrator) -> true
    DiscreteCallback(condition, affect!;
        initialize = saving_initialize,
        save_positions = (false, false))
end

# Sometimes, `integ(t)` yields a scalar instead of a vector :(
as_array(t::Number) = [t]
as_array(t::AbstractArray) = t

function is_linear_enough((t₀, t₁), (u₀, u₁), integ, u_idxs, abstol, reltol)
    tspread = t₁ - t₀
    slopes = (u₁ .- u₀) ./ tspread
    # Our interpolation dimension is the dimension _after_ all the dimensions in `u₀`
    interp_dim = ndims(u₀) + 1

    # Take the three quartiles, find the maximum error:
    t_quartiles = permutedims(cat([
                t₀ + tspread * 0.25,
                t₀ + tspread * 0.50,
                t₀ + tspread * 0.75,
            ]; dims = interp_dim), reverse(1:interp_dim))

    y_linear = u₀ .+ (t_quartiles .- t₀) .* slopes
    y_interp = stack([as_array(integ(t))[u_idxs] for t in t_quartiles]; dims = interp_dim)

    y_err = maximum(abs.(y_linear - y_interp), dims = 1)
    worst_idx = argmax(y_err[:])
    t_max = t_quartiles[worst_idx]
    e_max = y_err[worst_idx]
    is_linear = isapprox(y_linear, y_interp; atol = abstol, rtol = reltol)

    #=
    @info("is_linear_enough",
        t₀,
        t₁,
        slopes = join(string.(slopes), ", "),
        y_err = join(string.(y_err), ", "),
        is_linear,
    )

    if t₁ >= 24.0 && !is_linear && Main.integ === nothing
        Main.t₀ = t₀
        Main.t₁ = t₁
        Main.u₀ = u₀
        Main.u₁ = u₁
        Main.integ = integ
        Main.y_linear = y_linear
        Main.y_interp = y_interp
        #error()
    end
    =#
    return t_max, e_max, is_linear
end

function linearizing_save_loop(save_ts::Vector, save_us::Vector, integ, u_idxs)
    abstol = integ.opts.abstol
    reltol = integ.opts.reltol

    periods_to_check = [
        (integ.tprev, as_array(integ(integ.tprev))[u_idxs]) => (integ.t, as_array(integ(integ.t))[u_idxs]),
    ]

    # Only check linearization after we've made at least a single step forward.
    if integ.tprev != integ.t
        # Keep splitting each period until all are linearized
        while !isempty(periods_to_check)
            (t₀, u₀), (t₁, u₁) = popfirst!(periods_to_check)

            t_max, e_max, is_linear = is_linear_enough((t₀, t₁),
                (u₀, u₁),
                integ,
                u_idxs,
                abstol,
                reltol)
            if isnan(e_max) || isnan(t_max)
                throw(ArgumentError("Unable to linearize; ran out of precision"))
            end
            if !is_linear
                # Sample at `t_max`, the point of maximal deviation, insert our two new periods into `periods_to_check`
                u_max = as_array(integ(t_max))[u_idxs]
                pushfirst!(periods_to_check, (t_max, u_max) => (t₁, u₁))
                pushfirst!(periods_to_check, (t₀, u₀) => (t_max, u_max))
            else
                # If we were linear enough, push the beginning of this period and continue!
                push!(save_ts, t₀)
                push!(save_us, u₀)
            end
        end
    else
        # Just push the first values:
        t₀, u₀ = first(periods_to_check)
        push!(saved_values.t, t₀)
        push!(saved_values.saveval, u₀)
    end
end

function linearizing_save_affect!(saved_values::SavedValues, integ)
    linearizing_save_loop(saved_values.t, saved_values.saveval, integ, Colon())
    u_modified!(integ, false)
end

function linearizing_save_affect!(saved_values::Vector{<:SavedValues}, integ)
    for u_idx in 1:length(saved_values)
        linearizing_save_loop(saved_values[u_idx].t, saved_values[u_idx].saveval, integ, u_idx)
    end
    u_modified!(integ, false)
end

function linearizing_save_finalize!(saved_values::SavedValues, integ)
    u_end = integ(integ.t)
    push!(saved_values.t, integ.t)
    push!(saved_values.saveval, u_end)
end
function linearizing_save_finalize!(saved_values::Vector{<:SavedValues}, integ)
    u_end = integ(integ.t)
    for u_idx in 1:length(saved_values)
        push!(saved_values[u_idx].t, integ.t)
        push!(saved_values[u_idx].saveval, u_end[u_idx])
    end
end

"""
    LinearizingSavingCallback(saved_values::SavedValues)

Provides a saving callback that inserts interpolation points into your signal such that
a naive linear interpolation of the resultant saved values will be within `abstol`/`reltol`
of the higher-order interpolation of your solution.  This essentially makes a time/space
tradeoff, where more points in time are saved, costing more memory, but interpolation is
incredibly cheap and downstream algorithm complexity is reduced by not needing to bother
with multiple interpolation types.

The algorithm internally checks 3 equidistant points between each time point to determine
goodness of fit versus the linearly interpolated function; this should be sufficient for
interpolations up to the 4th order, higher orders may need more points to ensure good
fit.  This has not been implemented yet.
"""
function LinearizingSavingCallback(saved_values::SavedValues{T, <:AbstractArray{T}}) where {T}
    return DiscreteCallback(
        # We will process every timestep
        (u, t, integ) -> begin
            return true
        end,
        # On each timepoint, we linearize and save the timesteps into `saved_values`
        integ -> linearizing_save_affect!(saved_values, integ);
        # We need to save the final step
        finalize = (c, u, t, integ) -> linearizing_save_finalize(saved_values, integ),
        # Don't add tstops to the left and right.
        save_positions = (false, false))
end

"""
    LinearizingSavingCallback(saved_values::Vector{SavedValues})

An alternative implementation linearizes each state variable independently; this allows
one state variable to be sampled more densely while other state variables retain their
sparse sampling.  In this case, `length(saved_values)` must equal the number of states,
e.g. `length(sol.u[1])`.
"""
function LinearizingSavingCallback(saved_values::Vector{SavedValues{T, T}}) where {T}
    return DiscreteCallback(
        # We will process every timestep
        (u, t, integ) -> begin
            return true
        end,
        # On each timepoint, we linearize and save the timesteps into `saved_values`
        integ -> linearizing_save_affect!(saved_values, integ);
        # We need to save the final step
        finalize = (c, u, t, integ) -> linearizing_save_finalize(saved_values, integ),
        # Don't add tstops to the left and right.
        save_positions = (false, false))
end

export SavingCallback, SavedValues, LinearizingSavingCallback
