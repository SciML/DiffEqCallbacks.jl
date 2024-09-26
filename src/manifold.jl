"""
    ManifoldProjection(
        manifold; nlsolve = missing, save = true, autonomous = nothing,
        manifold_jacobian = nothing, autodiff = nothing, kwargs...)

In many cases, you may want to declare a manifold on which a solution lives. Mathematically,
a manifold `M` is defined by a function `g` as the set of points where `g(u) = 0`. An
embedded manifold can be a lower dimensional object which constrains the solution. For
example, `g(u) = E(u) - C` where `E` is the energy of the system in state `u`, meaning that
the energy must be constant (energy preservation). Thus by defining the manifold the
solution should live on, you can retain desired properties of the solution.

`ManifoldProjection` projects the solution of the differential equation to the chosen
manifold `g`, conserving a property while conserving the order. It is a consequence of
convergence proofs both in the deterministic and stochastic cases that post-step projection
to manifolds keep the same convergence rate, thus any algorithm can be easily extended to
conserve properties. If the solution is supposed to live on a specific manifold or conserve
such property, this guarantees the conservation law without modifying the convergence
properties.

## Arguments

  - `manifold`: The residual function for the manifold. If the ODEProblem is an inplace
    problem, then `g` must be an inplace function of form `g(resid, u, p)` or
    `g(resid, u, p, t)` and similarly if the ODEProblem is an out-of-place problem then `g`
    must be a function of form `g(u, p)` or `g(u, p, t)`.

## Keyword Arguments

  - `nlsolve`: Defaults to a special single-factorize algorithm (denoted by `missing`) that
    would work in most cases (See [1] for details). Alternatively, a nonlinear solver as
    defined in the
    [NonlinearSolve.jl format](https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/)
    can be specified.
  - `save`: Whether to do the standard saving (applied after the callback)
  - `autonomous`: Whether `g` is an autonomous function of the form `g(resid, u, p)` or
    `g(u, p)`. Specify it as `Val(::Bool)` to disable runtime branching. If `nothing`,
    we attempt to infer it from the signature of `g`.
  - `residual_prototype`: The size of the manifold residual. If it is not specified,
    we assume it to be same as `u` in the inplace case. Else we run a single evaluation of
    `manifold` to determine the size.
  - `kwargs`: All other keyword arguments are passed to
    [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/) if
    `nlsolve` is not `missing`.
  - `autodiff`: The autodifferentiation algorithm to use to compute the Jacobian if
    `manifold_jacobian` is not specified. This must be specified if `manifold_jacobian` is
    not specified and `nlsolve` is `missing`. If `nlsolve` is not `missing`, then
    `autodiff` is ignored.
  - `manifold_jacobian`: The Jacobian of the manifold (wrt the state). This has the same
    signature as `manifold` and the first argument is the Jacobian if inplace.

### Saveat Warning

Note that the `ManifoldProjection` callback modifies the endpoints of the integration
intervals and thus breaks assumptions of internal interpolations. Because of this, the
values for given by saveat will not be order-matching. However, the interpolation error can
be proportional to the change by the projection, so if the projection is making small
changes then one is still safe. However, if there are large changes from each projection,
you should consider only saving at stopping/projection times. To do this, set `tstops` to
the same values as `saveat`. There is a performance hit by doing so because now the
integrator is forced to stop at every saving point, but this is guerenteed to match the
order of the integrator even with the ManifoldProjection.

## References

[1] Ernst Hairer, Christian Lubich, Gerhard Wanner. Geometric Numerical Integration:
Structure-Preserving Algorithms for Ordinary Differential Equations. Berlin ;
New York :Springer, 2002.
"""
@concrete mutable struct ManifoldProjection
    manifold
    manifold_jacobian
    autodiff
    nlcache::Any
    nlsolve
    kwargs
    autonomous
end

function ManifoldProjection(
        manifold; nlsolve = missing, save = true, autonomous = nothing,
        manifold_jacobian = nothing, autodiff = nothing, kwargs...)
    affect! = ManifoldProjection(
        manifold, autodiff, manifold_jacobian, nlsolve, kwargs, autonomous)
    return DiscreteCallback(
        Returns(true), affect!; initialize = initialize_manifold_projection,
        save_positions = (false, save))
end

function ManifoldProjection(
        manifold, autodiff, manifold_jacobian, nlsolve, kwargs, autonomous)
    if autonomous isa Val{true} || autonomous isa Val{false}
        wrapped_manifold = TypedNonAutonomousFunction{SciMLBase._unwrap_val(autonomous)}(
            manifold, nothing)
        wrapped_manifold_jacobian = if manifold_jacobian === nothing
            nothing
        else
            TypedNonAutonomousFunction{SciMLBase._unwrap_val(autonomous)}(
                manifold_jacobian, nothing)
        end
        autonomous = SciMLBase._unwrap_val(autonomous)
    else
        _autonomous = autonomous === nothing ? false : autonomous
        wrapped_manifold = UntypedNonAutonomousFunction(_autonomous, manifold, nothing)
        wrapped_manifold_jacobian = if manifold_jacobian === nothing
            nothing
        else
            UntypedNonAutonomousFunction(_autonomous, manifold_jacobian, nothing)
        end
    end
    return ManifoldProjection(wrapped_manifold, wrapped_manifold_jacobian,
        autodiff, nothing, nlsolve, kwargs, autonomous)
end

function (proj::ManifoldProjection)(integrator)
    # update current time if residual function is time-dependent
    proj.manifold.t = integrator.t
    proj.manifold_jacobian !== nothing && (proj.manifold_jacobian.t = integrator.t)

    SciMLBase.reinit!(proj.nlcache, integrator.u; integrator.p)

    if proj.nlsolve === missing
        _, u, retcode = SciMLBase.solve!(proj.nlcache)
    else
        sol = SciMLBase.solve!(proj.nlcache)
        (; u, retcode) = sol
    end

    if !SciMLBase.successful_retcode(retcode)
        SciMLBase.terminate!(integrator, retcode)
        return
    end

    SciMLBase.copyto!(integrator.u, u)
end

function initialize_manifold_projection(cb, u, t, integrator)
    return initialize_manifold_projection(cb.affect!, u, t, integrator)
end
function initialize_manifold_projection(affect!::ManifoldProjection, u, t, integrator)
    if affect!.autonomous === nothing
        autonomous = maximum(SciMLBase.numargs(affect!.manifold.f)) ==
                     2 + SciMLBase.isinplace(integrator.f)
        affect!.manifold.autonomous = autonomous
        affect!.manifold_jacobian !== nothing &&
            (affect!.manifold_jacobian.autonomous = autonomous)
    end

    if affect!.nlsolve === missing
        affect!.manifold.t = t
        affect!.manifold_jacobian !== nothing && (affect!.manifold_jacobian.t = t)
        cache = init_manifold_projection(
            Val(SciMLBase.isinplace(integrator.f)), affect!.manifold, affect!.autodiff,
            affect!.manifold_jacobian, u, integrator.p; affect!.kwargs...)
    else
        # nlfunc = NonlinearFunction{iip}(affect!.g; affect!.resid_prototype)
        # nlprob = NonlinearProblem(nlfunc, u, integrator.p)
        # affect!.nlcache = init(nlprob, affect!.nlsolve; affect!.kwargs...)
        error("Not Implemented")
    end
    affect!.nlcache = cache
    u_modified!(integrator, false)
end

export ManifoldProjection

# wrapper for non-autonomous functions
@concrete mutable struct TypedNonAutonomousFunction{autonomous}
    f
    t::Any
end

(f::TypedNonAutonomousFunction{false})(res, u, p) = f.f(res, u, p, f.t)
(f::TypedNonAutonomousFunction{true})(res, u, p) = f.f(res, u, p)

(f::TypedNonAutonomousFunction{false})(u, p) = f.f(u, p, f.t)
(f::TypedNonAutonomousFunction{true})(u, p) = f.f(u, p)

@concrete mutable struct UntypedNonAutonomousFunction
    autonomous::Bool
    f
    t::Any
end

function (f::UntypedNonAutonomousFunction)(res, u, p)
    return f.autonomous ? f.f(res, u, p) : f.f(res, u, p, f.t)
end
(f::UntypedNonAutonomousFunction)(u, p) = f.autonomous ? f.f(u, p) : f.f(u, p, f.t)

# This is the algorithm described in Hairer III.
@concrete mutable struct SingleFactorizeManifoldProjectionCache{iip}
    manifold
    p
    abstol
    maxiters::Int

    ũ
    JJᵀfact::Any  # LU might fail and we might end up doing QR
    u_cache
    λ_cache
    gu_cache

    first_call::Bool
    J
    JJᵀ
    manifold_jacobian
    autodiff
    di_extras
end

function SciMLBase.reinit!(
        cache::SingleFactorizeManifoldProjectionCache{iip}, u; p = cache.p) where {iip}
    if !cache.first_call || (cache.ũ !== u || cache.p !== p)
        compute_manifold_jacobian!(cache.J, cache.manifold_jacobian, cache.autodiff,
            Val(iip), cache.manifold, cache.gu_cache, u, p, cache.di_extras)
        mul!(cache.JJᵀ, cache.J, cache.J')
        cache.JJᵀfact = safe_factorize!(cache.JJᵀ)
    end
    cache.first_call = false
    cache.ũ = u
    cache.p = p
end

default_abstol(::Type{T}) where {T} = real(oneunit(T)) * (eps(real(one(T))))^(4 // 5)

function init_manifold_projection(IIP::Val{iip}, manifold, autodiff, manifold_jacobian, ũ,
        p; abstol = default_abstol(eltype(ũ)), maxiters = 1000,
        resid_prototype = nothing) where {iip}
    if iip
        if resid_prototype !== nothing
            gu = similar(resid_prototype)
            λ = similar(resid_prototype)
        else
            @warn "`resid_prototype` not provided for in-place problem. Assuming size of \
                   output is the same as input. This might be incorrect." maxlog=1
            gu = similar(ũ)
            λ = similar(ũ)
        end
    else
        gu = nothing
        λ = manifold(ũ, p)
    end

    J, di_extras = setup_manifold_jacobian(manifold_jacobian, autodiff, IIP, manifold,
        gu, ũ, p)
    JJᵀ = J * J'
    JJᵀfact = safe_factorize!(JJᵀ)

    return SingleFactorizeManifoldProjectionCache{iip}(
        manifold, p, abstol, maxiters, ũ, JJᵀfact, similar(ũ), λ, gu,
        true, J, JJᵀ, manifold_jacobian, autodiff, di_extras)
end

function SciMLBase.solve!(cache::SingleFactorizeManifoldProjectionCache{iip}) where {iip}
    fill!(cache.λ_cache, false)
    ũ = cache.ũ
    gu = cache.gu_cache

    internal_solve_failed = true

    if cache.gu_cache !== nothing
        cache.manifold(gu, ũ, cache.p)
    else
        gu = cache.manifold(ũ, cache.p)
    end

    for _ in 1:(cache.maxiters)
        if maximum(abs, gu) < cache.abstol
            internal_solve_failed = false
            break
        end

        δλ = cache.JJᵀfact \ gu
        @. cache.λ_cache -= δλ

        mul!(vec(cache.u_cache), cache.J', vec(cache.λ_cache))
        cache.u_cache += ũ
        if cache.gu_cache !== nothing
            cache.manifold(gu, cache.u_cache, cache.p)
        else
            gu = cache.manifold(cache.u_cache, cache.p)
        end
    end

    return (cache.λ_cache, cache.u_cache,
        ifelse(internal_solve_failed, ReturnCode.ConvergenceFailure, ReturnCode.Success))
end

function setup_manifold_jacobian(
        manifold_jacobian::M, autodiff, ::Val{iip}, manifold, gu, ũ, p) where {M, iip}
    if iip
        J = similar(ũ, promote_type(eltype(gu), eltype(ũ)), (length(gu), length(ũ)))
        manifold_jacobian(J, ũ, p)
    else
        J = manifold_jacobian(ũ, p)
    end
    return J, nothing
end

function setup_manifold_jacobian(
        ::Nothing, autodiff, ::Val{iip}, manifold, gu, ũ, p) where {iip}
    if iip
        di_extras = DI.prepare_jacobian(manifold, gu, autodiff, ũ, Constant(p))
        J = DI.jacobian(manifold, gu, di_extras, autodiff, ũ, Constant(p))
    else
        di_extras = DI.prepare_jacobian(manifold, autodiff, ũ, Constant(p))
        J = DI.jacobian(manifold, di_extras, autodiff, ũ, Constant(p))
    end
    return J, di_extras
end

function compute_manifold_jacobian!(J, manifold_jacobian, autodiff, ::Val{iip},
        manifold, gu, ũ, p, di_extras) where {iip}
    if iip
        manifold_jacobian(J, ũ, p)
    else
        J = manifold_jacobian(ũ, p)
    end
    return J
end

function compute_manifold_jacobian!(J, ::Nothing, autodiff, ::Val{iip}, manifold, gu,
        ũ, p, di_extras) where {iip}
    if iip
        DI.jacobian!(manifold, gu, J, di_extras, autodiff, ũ, Constant(p))
    else
        DI.jacobian!(manifold, J, di_extras, autodiff, ũ, Constant(p))
    end
    return J
end

function setup_manifold_jacobian(::Nothing, ::Nothing, args...)
    error("`autodiff` is set to `nothing` and analytic manifold jacobian is not provided.")
end

function safe_factorize!(A::AbstractMatrix)
    if issquare(A)
        fact = LinearAlgebra.cholesky(A; check = false)
        fact_sucessful(fact) && return fact
    elseif size(A, 1) > size(A, 2)
        fact = LinearAlgebra.qr(A)
        fact_sucessful(fact) && return fact
    end
    return LinearAlgebra.qr!(A, LinearAlgebra.ColumnNorm())
end

function fact_sucessful(F::LinearAlgebra.QRCompactWY)
    m, n = size(F)
    U = view(F.factors, 1:min(m, n), 1:n)
    return all(!iszero, Iterators.reverse(@view U[diagind(U)]))
end
function fact_sucessful(F::FT) where {FT}
    return hasmethod(LinearAlgebra.issuccess, (FT,)) ? LinearAlgebra.issuccess(F) : true
end
