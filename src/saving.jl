import SciMLBase: AbstractSciMLAlgorithm

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

function is_linear_enough!(caches, is_linear, t₀, t₁, u₀, u₁, integ, abstol, reltol)
    (; y_linear, y_interp, slopes) = caches
    tspread = t₁ - t₀
    num_us = length(u₀)
    @inbounds for u_idx in 1:num_us
        slopes[u_idx] = (u₁[u_idx] - u₀[u_idx]) / tspread
    end
    t_quartile(t_idx) = t₀ + tspread * t_idx / 4.0

    # Calculate interpolated and linear samplings in our three quartiles
    @inbounds for t_idx in 1:3
        t = t_quartile(t_idx)
        # Linear interpolation
        @inbounds for u_idx in 1:num_us
            y_linear[u_idx, t_idx] = u₀[u_idx] .+ (t - t₀) .* slopes[u_idx]
        end

        # Solver interpolation
        # We would like to use `integ(@view(y_interp[:, t_idx]))` here,
        # but in IDA the conversion of views to `NVector` loses the shared
        # memory that the view would have given us, so we instead use a
        # temporary array then copy it into `y_interp`, which loses very
        # little time and still prevents allocations from `integ(t)`.
        @with_cache caches.us u_interp begin
            integ(u_interp, t, Val{0}; idxs = nothing)
            for u_idx in 1:num_us
                y_interp[u_idx, t_idx] = u_interp[u_idx]
            end
        end
    end

    # Return `is_linear` for each state
    @inbounds for u_idx in 1:num_us
        is_linear[u_idx] = true
        for t_idx in 1:3
            is_linear[u_idx] &= isapprox(y_linear[u_idx, t_idx],
                y_interp[u_idx, t_idx];
                atol = abstol,
                rtol = reltol)
        end
    end
    # Find worst time index so that we split our period there
    t_max_idx = 1
    e_max = 0.0
    @inbounds for t_idx in 1:3
        for u_idx in 1:num_us
            e = abs(y_linear[u_idx, t_idx] - y_interp[u_idx, t_idx])
            if e > e_max
                t_max_idx = t_idx
                e_max = e
            end
        end
    end
    return t_quartile(t_max_idx)
end

function linearize_period(t₀, t₁, u₀, u₁, integ, ilsc, caches, u_mask,
        dtmin, interpolate_mask, abstol, reltol)
    # Sanity check that we don't accidentally infinitely recurse
    if t₁ - t₀ < dtmin
        @debug("Linearization failure",
            t₁, t₀, string(u₀), string(u₁), string(u_mask), dtmin)
        throw(ArgumentError("Linearization failed, fell below linearization subdivision threshold"))
    end

    @with_cache caches.u_masks is_linear begin
        tᵦ = is_linear_enough!(caches,
            is_linear,
            t₀, t₁,
            u₀, u₁,
            integ,
            abstol, reltol)

        # Rename `is_linear` to `is_nonlinear`, invert the meaning
        # and mask by `u_mask` (but re-use the memory)
        is_nonlinear = is_linear
        for u_idx in 1:length(is_linear)
            is_nonlinear[u_idx] = !is_linear[u_idx] & u_mask[u_idx] &
                                  interpolate_mask[u_idx]
        end

        if any(is_nonlinear)
            # If it's not linear, split this period into two and recurse, altering our `u_mask`:
            @with_cache caches.us uᵦ begin
                integ(uᵦ, tᵦ, Val{0}; idxs = nothing)
                linearize_period(
                    t₀, tᵦ,
                    u₀, uᵦ,
                    integ,
                    ilsc,
                    caches,
                    is_nonlinear,
                    dtmin,
                    interpolate_mask,
                    abstol,
                    reltol)

                # Recurse into the second half of the period as well, as we're not guaranteed that
                # the second half is linear yet.   Also, use the full `u_mask` as we need to store
                # everyone this time.
                linearize_period(
                    tᵦ, t₁,
                    uᵦ, u₁,
                    integ,
                    ilsc,
                    caches,
                    u_mask,
                    dtmin,
                    interpolate_mask,
                    abstol,
                    reltol)
            end
        else
            # If everyone is linear, store this period, according to our `u_mask`!
            store_u_block!(ilsc, Val(num_derivatives(ilsc)), integ, caches, t₁, u₁, u_mask)
        end
    end
end

function store_u_block!(
        ilsc, ::Val{num_derivatives}, integ, caches, t₁, u₁, u_mask) where {num_derivatives}
    @with_cache caches.us u begin
        for u_idx in 1:length(u)
            caches.u_block[1, u_idx] = u₁[u_idx]
        end
        for deriv_idx in 1:num_derivatives
            integ(u, t₁, Val{deriv_idx}; idxs = nothing)
            for u_idx in 1:length(u)
                caches.u_block[deriv_idx + 1, u_idx] = u[u_idx]
            end
        end
        store!(ilsc, t₁, caches.u_block, u_mask)
    end
end

struct LinearizingSavingCallbackCacheType{S, U}
    y_linear::Matrix{S}
    y_interp::Matrix{S}
    slopes::Vector{S}
    # U is not necessarily a `Vector{S}` because it can be an `NVector` thanks to Sundials.
    us::ThreadUnsafeCachePool{U}
    u_block::Matrix{S}
    u_masks::ThreadUnsafeCachePool{BitVector}

    function LinearizingSavingCallbackCacheType{S, U}(
            num_us::Int, num_derivatives::Int, U_alloc::Function) where {S, U}
        y_linear = Matrix{S}(undef, (num_us, 3))
        y_interp = Matrix{S}(undef, (num_us, 3))
        slopes = Vector{S}(undef, num_us)
        u_block = Matrix{S}(undef, (num_derivatives + 1, num_us))
        F_umasks = () -> BitVector(undef, num_us)
        u_masks = CachePool(BitVector, F_umasks; thread_safe = false)

        # Workaround for Sundials allocations; conversion from `Vector{S}`
        # to `NVector()` allocates, so we require the caller to pass in a
        # `U` and a `U_alloc` so that the default of `U = Vector{S}` and
        # U_alloc = () -> Vector{S}(undef, num_us)` can be overridden.
        # This is automatically done by `DiffEqCallbacksSundialsExt`, via
        # the `solver_state_type()` and `solver_state_alloc()` hooks below.
        us = CachePool(U, U_alloc; thread_safe = false)
        return new{S, U}(
            y_linear,
            y_interp,
            slopes,
            us,
            u_block,
            u_masks
        )
    end
end

# This exists purely so that different solver types can wrap/alter the
# type of the state vectors cached by the `LinearizingSavingCallbackCache`.
# U is typically something like `Vector{Float64}`.
solver_state_type(solver::AbstractSciMLAlgorithm, U::DataType) = U
function solver_state_alloc(solver::AbstractSciMLAlgorithm, U::DataType, num_us::Int)
    () -> U(undef, num_us)
end

"""
    LinearizingSavingCallbackCache(prob, solver; num_derivatives=0)

Top-level cache for the `LinearizingSavingCallback`.  Typically used
to vastly reduce the number of allocations when performing an ensemble
solve, where allocations from one solution can be used by the next.

Users must pass in `solver` to allow for solver-specific allocation
strategies.  As an example, `IDA` requires allocation of `NVector`
objects rather than `Vector{S}` objects, and to automatically
determine this, the `LinearizingSavingCallbackCache` takes in the
solver as well.  See the `DiffEqCallbacksSundialsExt` extension
for the details on how this type adjustment is made.

This top-level cache creates two thread-safe cache pools that are then
used by each solve to allocate thread-unsafe cache pools.  Those per-
solution cache pools are then re-used across solutions as the ensemble
finishes one trajectory and moves to another.

Example usage:

```julia
# Linearize the primal, and the first derivative
num_derivatives = 1

# Create a cache, to be used across all ensemble simulations
cache = LinearizingSavingCallbackCache(prob, solver; num_derivatives)

# Store the results in this array of independently linearized solutions
ilss = Vector{IndependentlyLinearizedSolution}(undef, num_trajectories)

# Create `prob_func` piece to remake `prob` to have the correct callback,
# hooking up the necessary caching pieces.
function linearizer_adding_remake(prob,i,_)
    ilss[i] = IndependentlyLinearizedSolution(prob, num_derivatives; cache_pool=cache.ils_cache)
    lsc = LinearizingSavingCallback(ilss[i]; cache_pool=cache.lsc_cache)
    return remake(prob; callback=lsc)
end

ensembleprob = EnsembleProblem(prob; prob_func=linearizer_adding_remake)
solve(ensembleprob, solver, EnsembleThreads(); ...)
```
"""
function LinearizingSavingCallbackCache(prob, solver; num_derivatives = 0, chunk_size = 512)
    T = eltype(prob.tspan)
    S = eltype(prob.u0)
    U = solver_state_type(solver, typeof(prob.u0))
    num_us = length(prob.u0)
    U_alloc = solver_state_alloc(solver, typeof(prob.u0), num_us)
    return LinearizingSavingCallbackCache(
        T, S, U, U_alloc, num_us; num_derivatives, chunk_size)
end

function make_lsc_cache(num_us, num_derivatives, S = Float64,
        U = Vector{S}, U_alloc = () -> U(undef, num_us))
    return CachePool(
        LinearizingSavingCallbackCacheType{S, U},
        () -> LinearizingSavingCallbackCacheType{S, U}(num_us, num_derivatives, U_alloc);
        thread_safe = true
    )
end

function LinearizingSavingCallbackCache(
        T, S, U, U_alloc, num_us; num_derivatives = 0, chunk_size = 512)
    return (;
        # This cache is used by the LinearizingSavingCallback, it creates `LinearizingSavingCallbackCacheType`
        # objects, which is quite a mouthful, but contains all the temporary values needed for a single
        # solve's linearization.  Notably, it contains within itself cachepools for `u` vectors and whatnot,
        # and it is _not_ thread-safe, because we assume that a single solve is single-threaded, so we use a
        # single thread-safe cache pool (the `lsc_cache`) to spawn off a collection of these smaller, thread-
        # unsafe (but faster to acquire/release) cache pools.
        lsc_cache = make_lsc_cache(num_us, num_derivatives, S, U, U_alloc),
        # This cache is used by the `IndependentlyLinearizedSolutionChunks` to do things like allocate `u`,
        # `t` and `time_mask` chunks.
        ils_cache = CachePool(
            IndependentlyLinearizedSolutionChunksCache{T, S},
            () -> IndependentlyLinearizedSolutionChunksCache{T, S}(
                num_us,
                num_derivatives,
                chunk_size
            ),
            thread_safe = true
        )
    )
end

function default_lsc_cache(T, S, num_us, num_derivatives)
    return CachePool(
        LinearizingSavingCallbackCacheType{S, Vector{S}},
        () -> LinearizingSavingCallbackCacheType{S, Vector{S}}(
            num_us, num_derivatives, () -> Vector{S}(undef, num_us));
        thread_safe = true
    )
end

"""
    LinearizingSavingCallback(ils::IndependentlyLinearizedSolution)
    LinearizingSavingCallback(ilss::Vector{IndependentlyLinearizedSolution})

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

This callback generator takes in an `IndependentlyLinearizedSolution` object to store
output into.  An `IndependentlyLinearizedSolution` object itself controls how many
derivatives (if any) to linearize along with the primal states themselves.

Example usage:

```julia
ils = IndependentlyLinearizedSolution(prob)
solve(prob, solver; callback=LinearizingSavingCallback(ils))
```

# Keyword Arguments
- `interpolate_mask::BitVector`: a set of `u` indices for which the integrator
  interpolant can be queried. Any false indices will be linearly-interpolated
  based on the `sol.t` points instead (no subdivision).  This is useful for when
  a solution needs to ignore certain indices due to badly-behaved interpolation.
"""
function LinearizingSavingCallback(ils::IndependentlyLinearizedSolution{T, S};
        interpolate_mask = BitVector(true for _ in 1:length(ils.ilsc.u_chunks)),
        abstol::Union{S, Nothing} = nothing,
        reltol::Union{S, Nothing} = nothing,
        cache_pool::CachePool{C} = make_lsc_cache(
            length(ils.ilsc.u_chunks), num_derivatives(ils.ilsc), S)
) where {T, S, C}
    ilsc = ils.ilsc
    full_mask = BitVector(true for _ in 1:length(ilsc.u_chunks))
    num_derivatives_val = Val(num_derivatives(ilsc))

    # `caches` is initialized in `initialize`, but we need to constrain
    # its type here so that the closures in `DiscreteCallback` are stable
    local caches::C
    #caches = acquire!(cache_pool)
    return DiscreteCallback(
        # We will process every timestep
        (u, t, integ) -> begin
            return true
        end,
        # On each timepoint, we linearize and save the timesteps into `ilsc`
        integ -> begin
            t₀ = integ.tprev
            t₁ = integ.t
            @with_cache caches.us u₀ begin
                @with_cache caches.us u₁ begin
                    # Get `u₀` and `u₁` from the integrator
                    integ(u₀, t₀, Val{0}; idxs = nothing)
                    integ(u₁, t₁, Val{0}; idxs = nothing)

                    # Store first timepoints.  Usually we'd do this in `initialize`
                    # but `integ(u, t, deriv)` doesn't work that early, and so we
                    # must wait until we've taken at least a single step.
                    if isempty(ilsc)
                        store_u_block!(
                            ilsc, num_derivatives_val, integ, caches, t₀, u₀, full_mask)
                    end

                    dtmin = eps(t₁ - t₀) * 1000.0
                    linearize_period(
                        t₀, t₁, u₀, u₁,
                        integ, ilsc, caches,
                        full_mask, dtmin, interpolate_mask,
                        # Loosen `abstol` and `reltol` according to the derivative level
                        something(abstol, integ.opts.abstol),
                        something(reltol, integ.opts.reltol)
                    )
                end
            end
            u_modified!(integ, false)
        end,
        # In our `initialize`, we create some caches so we allocate less.
        initialize = (c, u, t, integ) -> begin
            caches = acquire!(cache_pool)
            u_modified!(integ, false)
        end,
        # We need to finalize the ils and free our caches
        finalize = (c, u, t, integ) -> begin
            finish!(ils, check_error(integ))
            if cache_pool !== nothing
                release!(cache_pool, caches)
            end
        end,
        # Don't add tstops to the left and right.
        save_positions = (false, false))
end

export SavingCallback, SavedValues, LinearizingSavingCallback,
       LinearizingSavingCallbackCache
