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

"""
Ernst Hairer, Christian Lubich, Gerhard Wanner. Geometric Numerical Integration:
Structure-Preserving Algorithms for Ordinary Differential Equations. Berlin ;
New York :Springer, 2002.
"""
function ManifoldProjection(g; nlsolve = NLSOLVEJL_SETUP(), save = true,
                            autonomous = maximum(SciMLBase.numargs(g)) == 3,
                            nlopts = Dict{Symbol, Any}())
    affect! = ManifoldProjection{autonomous}(g, nlsolve, nlopts)
    condtion = (u, t, integrator) -> true
    save_positions = (false, save)
    DiscreteCallback(condtion, affect!;
                     initialize = Manifold_initialize,
                     save_positions = save_positions)
end

export ManifoldProjection
