# wrapper for non-autonomous functions
mutable struct NonAutonomousFunction{F,autonomous}
  f::F
  t
  p
end
(p::NonAutonomousFunction{F,true})(res,u) where F = p.f(res, u)
(p::NonAutonomousFunction{F,false})(res,u) where F = p.f(res, u, p.p, p.t)

mutable struct ManifoldProjection{autonomous,F,NL,O}
  g::F
  nl_rhs
  nlsolve::NL
  nlopts::O

  function ManifoldProjection{autonomous}(g, nlsolve, nlopts) where {autonomous}
    # replace residual function if it is time-dependent
    # since NLsolve only accepts functions with two arguments
    _g = NonAutonomousFunction{typeof(g),autonomous}(g, 0,0)
    new{autonomous,typeof(_g),typeof(nlsolve),typeof(nlopts)}(_g, _g, nlsolve, nlopts)
  end
end

# Now make `affect!` for this:
function (p::ManifoldProjection{autonomous,NL})(integrator) where {autonomous,NL}
  # update current time if residual function is time-dependent
  if !autonomous
    p.g.t = integrator.t
  end
  p.g.p = integrator.p

  integrator.u .= p.nlsolve(p.nl_rhs, integrator.u; p.nlopts...)
end

function Manifold_initialize(cb,u::Number,t,integrator)
  cb.affect!.nl_rhs = cb.affect!.nlsolve(Val{:init}, cb.affect!.g, [u])
  u_modified!(integrator,false)
end

function Manifold_initialize(cb,u,t,integrator)
  cb.affect!.nl_rhs = cb.affect!.nlsolve(Val{:init}, cb.affect!.g, u)
  u_modified!(integrator,false)
end

function ManifoldProjection(g; nlsolve=NLSOLVEJL_SETUP(), save=true,
                            autonomous=numargs(g)==2, nlopts=Dict{Symbol,Any}())
  affect! = ManifoldProjection{autonomous}(g, nlsolve, nlopts)
  condtion = (u,t,integrator) -> true
  save_positions = (false,save)
  DiscreteCallback(condtion, affect!;
                   initialize = Manifold_initialize,
                   save_positions=save_positions)
end

export ManifoldProjection
