Base.@pure function determine_chunksize(u, alg::DiffEqBase.DEAlgorithm)
    determine_chunksize(u, get_chunksize(alg))
end
Base.@pure function determine_chunksize(u, CS)
    if CS != 0
        return CS
    else
        return ForwardDiff.pickchunksize(length(u))
    end
end

struct NLSOLVEJL_SETUP{CS, AD} end
Base.@pure function NLSOLVEJL_SETUP(; chunk_size = 0, autodiff = true)
    NLSOLVEJL_SETUP{chunk_size, autodiff}()
end
(::NLSOLVEJL_SETUP)(f, u0; kwargs...) = (res = NLsolve.nlsolve(f, u0; kwargs...); res.zero)
function (p::NLSOLVEJL_SETUP{CS, AD})(::Type{Val{:init}}, f, u0_prototype) where {CS, AD}
    AD ? autodiff = :forward : autodiff = :central
    OnceDifferentiable(f, u0_prototype, u0_prototype, autodiff,
        ForwardDiff.Chunk(determine_chunksize(u0_prototype, CS)))
end

# wrapper for non-autonomous functions
mutable struct NonAutonomousFunction{F, autonomous}
    f::F
    t::Any
    p::Any
end
(p::NonAutonomousFunction{F, true})(res, u) where {F} = p.f(res, u, p.p)
(p::NonAutonomousFunction{F, false})(res, u) where {F} = p.f(res, u, p.p, p.t)

"""
```julia
ManifoldProjection(g; nlsolve = NLSOLVEJL_SETUP(), save = true)
```

In many cases, you may want to declare a manifold on which a solution lives.
Mathematically, a manifold `M` is defined by a function `g` as the set of points
where `g(u)=0`. An embedded manifold can be a lower dimensional object which
constrains the solution. For example, `g(u)=E(u)-C` where `E` is the energy
of the system in state `u`, meaning that the energy must be constant (energy
preservation). Thus by defining the manifold the solution should live on, you
can retain desired properties of the solution.

`ManifoldProjection` projects the solution of the differential equation to the chosen
manifold `g`, conserving a property while conserving the order. It is a consequence of
convergence proofs both in the deterministic and stochastic cases that post-step projection
to manifolds keep the same convergence rate, thus any algorithm can be easily extended to
conserve properties. If the solution is supposed to live on a specific manifold or conserve
such property, this guarantees the conservation law without modifying the convergence
properties.

## Arguments

  - `g`: The residual function for the manifold. This is an inplace function of form
    `g(resid, u)` or `g(resid, u, p, t)` which writes to the residual `resid` the
    difference from the manifold components. Here, it is assumed that `resid` is of
    the same shape as `u`.

## Keyword Arguments

  - `nlsolve`: A nonlinear solver as defined [in the nlsolve format](https://docs.sciml.ai/DiffEqDocs/stable/features/linear_nonlinear/)
  - `save`: Whether to do the standard saving (applied after the callback)
  - `autonomous`: Whether `g` is an autonomous function of the form `g(resid, u)`.
  - `nlopts`: Optional arguments to nonlinear solver which can be any of the [NLsolve keywords](https://github.com/JuliaNLSolvers/NLsolve.jl#fine-tunings).

### Saveat Warning

Note that the `ManifoldProjection` callback modifies the endpoints of the integration intervals
and thus breaks assumptions of internal interpolations. Because of this, the values for given by
saveat will not be order-matching. However, the interpolation error can be proportional to the
change by the projection, so if the projection is making small changes then one is still safe.
However, if there are large changes from each projection, you should consider only saving at
stopping/projection times. To do this, set `tstops` to the same values as `saveat`. There is a
performance hit by doing so because now the integrator is forced to stop at every saving point,
but this is guerenteed to match the order of the integrator even with the ManifoldProjection.

## References

Ernst Hairer, Christian Lubich, Gerhard Wanner. Geometric Numerical Integration:
Structure-Preserving Algorithms for Ordinary Differential Equations. Berlin ;
New York :Springer, 2002.
"""
mutable struct ManifoldProjection{autonomous, F, NL, O}
    g::F
    nl_rhs::Any
    nlsolve::NL
    nlopts::O

    function ManifoldProjection{autonomous}(g, nlsolve, nlopts) where {autonomous}
        # replace residual function if it is time-dependent
        # since NLsolve only accepts functions with two arguments
        _g = NonAutonomousFunction{typeof(g), autonomous}(g, 0, 0)
        new{autonomous, typeof(_g), typeof(nlsolve), typeof(nlopts)}(_g, _g, nlsolve,
            nlopts)
    end
end

# Now make `affect!` for this:
function (p::ManifoldProjection{autonomous, NL})(integrator) where {autonomous, NL}
    # update current time if residual function is time-dependent
    if !autonomous
        p.g.t = integrator.t
    end
    p.g.p = integrator.p

    integrator.u .= p.nlsolve(p.nl_rhs, integrator.u; p.nlopts...)
end

function Manifold_initialize(cb, u::Number, t, integrator)
    cb.affect!.nl_rhs = cb.affect!.nlsolve(Val{:init}, cb.affect!.g, [u])
    u_modified!(integrator, false)
end

function Manifold_initialize(cb, u, t, integrator)
    cb.affect!.nl_rhs = cb.affect!.nlsolve(Val{:init}, cb.affect!.g, u)
    u_modified!(integrator, false)
end

function ManifoldProjection(g; nlsolve = NLSOLVEJL_SETUP(), save = true,
        autonomous = maximum(SciMLBase.numargs(g)) == 3,
        nlopts = Dict{Symbol, Any}())
    affect! = ManifoldProjection{autonomous}(g, nlsolve, nlopts)
    condition = (u, t, integrator) -> true
    save_positions = (false, save)
    DiscreteCallback(condition, affect!;
        initialize = Manifold_initialize,
        save_positions = save_positions)
end

export ManifoldProjection
