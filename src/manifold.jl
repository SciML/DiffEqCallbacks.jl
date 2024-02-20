# wrapper for non-autonomous functions
mutable struct NonAutonomousFunction{iip, F, autonomous}
    f::F
    t::Any
end

(f::NonAutonomousFunction{true, F, true})(res, u, p) where {F} = f.f(res, u, p)
(f::NonAutonomousFunction{true, F, false})(res, u, p) where {F} = f.f(res, u, p, f.t)

(f::NonAutonomousFunction{false, F, true})(u, p) where {F} = f.f(u, p)
(f::NonAutonomousFunction{false, F, false})(u, p) where {F} = f.f(u, p, f.t)

SciMLBase.isinplace(::NonAutonomousFunction{iip}) where {iip} = iip

"""
    ManifoldProjection(g; nlsolve = missing, save = true, nlls = Val(true),
        isinplace = Val(true), autonomous = nothing, resid_prototype = nothing,
        kwargs...)

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

  - `g`: The residual function for the manifold.

      * This is an inplace function of form `g(resid, u, p)` or `g(resid, u, p, t)` which
        writes to the residual `resid` the difference from the manifold components. Here, it
        is assumed that `resid` is of the same shape as `u`.
      * If `isinplace = Val(false)`, then `g` should be a function of the form `g(u, p)` or
        `g(u, p, t)` which returns the residual.

## Keyword Arguments

  - `nlsolve`: A nonlinear solver as defined in the
    [NonlinearSolve.jl format](https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/)
  - `save`: Whether to do the standard saving (applied after the callback)
  - `nlls`: If the problem is a nonlinear least squares problem. `nlls = Val(false)`
    generates a `NonlinearProblem` which is typically faster than
    `NonlinearLeastSquaresProblem`, but is only applicable if the residual size is same as
    the state size.
  - `autonomous`: Whether `g` is an autonomous function of the form `g(resid, u, p)` or
    `g(u, p)`. Specify it as `Val(::Bool)` to ensure this function call is type stable.
  - `residual_prototype`: This needs to be specified if `nlls = Val(true)` and
    `inplace = Val(true)` are specified together, else it is taken to be same as `u`.
  - `kwargs`: All other keyword arguments are passed to
    [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/).

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

Ernst Hairer, Christian Lubich, Gerhard Wanner. Geometric Numerical Integration:
Structure-Preserving Algorithms for Ordinary Differential Equations. Berlin ;
New York :Springer, 2002.
"""
mutable struct ManifoldProjection{iip, nlls, autonomous, F, NL, R, K}
    g::F
    nlcache::Any
    nlsolve::NL
    resid_prototype::R
    kwargs::K

    function ManifoldProjection{iip, nlls, autonomous}(
            g, nlsolve, resid_prototype, kwargs) where {iip, nlls, autonomous}
        # replace residual function if it is time-dependent
        _g = NonAutonomousFunction{iip, typeof(g), autonomous}(g, 0)
        return new{iip, nlls, autonomous, typeof(_g), typeof(nlsolve),
            typeof(resid_prototype), typeof(kwargs)}(
            _g, nothing, nlsolve, resid_prototype, kwargs)
    end
end

# Now make `affect!` for this:
function (p::ManifoldProjection{iip, nlls, autonomous, NL})(integrator) where {iip, nlls,
        autonomous, NL}
    # update current time if residual function is time-dependent
    !autonomous && (p.g.t = integrator.t)

    # solve the nonlinear problem
    reinit!(p.nlcache, integrator.u; p = integrator.p)
    sol = solve!(p.nlcache)

    if !SciMLBase.successful_retcode(sol)
        SciMLBase.terminate!(integrator, sol.retcode)
        return
    end

    copyto!(integrator.u, sol.u)
end

function Manifold_initialize(cb, u, t, integrator)
    return Manifold_initialize(cb.affect!, u, t, integrator)
end
function Manifold_initialize(
        affect!::ManifoldProjection{iip, nlls}, u, t, integrator) where {iip, nlls}
    nlfunc = NonlinearFunction{iip}(affect!.g; affect!.resid_prototype)
    nlprob = if nlls
        NonlinearLeastSquaresProblem(nlfunc, u, integrator.p)
    else
        NonlinearProblem(nlfunc, u, integrator.p)
    end
    affect!.nlcache = init(nlprob, affect!.nlsolve; affect!.kwargs...)
    u_modified!(integrator, false)
end

# Since this is applied to every point, we can reasonably assume that the solution is close
# to the initial guess, so we would want to use NewtonRaphson / RobustMultiNewton instead of
# the default one.
function ManifoldProjection(g; nlsolve = missing, save = true, nlls = Val(true),
        isinplace = Val(true), autonomous = nothing, resid_prototype = nothing,
        kwargs...)
    # `nothing` is a valid solver, so this need to be `missing`
    _nlls = SciMLBase._unwrap_val(nlls)
    _nlsolve = nlsolve === missing ? (_nlls ? GaussNewton() : NewtonRaphson()) : nlsolve
    iip = SciMLBase._unwrap_val(isinplace)
    if autonomous === nothing
        if iip
            autonomous = maximum(SciMLBase.numargs(g)) == 3
        else
            autonomous = maximum(SciMLBase.numargs(g)) == 2
        end
    end
    affect! = ManifoldProjection{iip, _nlls, autonomous}(
        g, _nlsolve, resid_prototype, kwargs)
    condition = (u, t, integrator) -> true
    return DiscreteCallback(condition, affect!; initialize = Manifold_initialize,
        save_positions = (false, save))
end

export ManifoldProjection
