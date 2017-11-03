# wrapper for non-autonomous functions
mutable struct NonAutonomousFunction{F}
  f::F
  t
end
(p::NonAutonomousFunction)(u, res) = p.f(p.t, u, res)

mutable struct ManifoldProjection{autonomous,F,NL,O}
  g::F
  nl_rhs
  nlsolve::NL
  nlopts::O

  function ManifoldProjection{autonomous}(g, nlsolve, nlopts) where {autonomous}
    # replace residual function if it is time-dependent
    # since NLsolve only accepts functions with two arguments
    if !autonomous
      g = NonAutonomousFunction(g, 0)
    end

    new{autonomous,typeof(g),typeof(nlsolve),typeof(nlopts)}(g, g, nlsolve, nlopts)
  end
end

# Now make `affect!` for this:
function (p::ManifoldProjection{autonomous,NL})(integrator) where {autonomous,NL}
  # update current time if residual function is time-dependent
  if !autonomous
    p.g.t = integrator.t
  end

  integrator.u .= p.nlsolve(p.nl_rhs, integrator.u; p.nlopts...)
end

function Manifold_initialize(cb,t,u,integrator)
  cb.affect!.nl_rhs = cb.affect!.nlsolve(Val{:init}, cb.affect!.g, u)
  u_modified!(integrator,false)
end

function ManifoldProjection(g; nlsolve=NLSOLVEJL_SETUP(), save=true,
                            autonomous=numargs(g)==2, nlopts=Dict{Symbol,Any}())
  affect! = ManifoldProjection{autonomous}(g, nlsolve, nlopts)
  condtion = (t,u,integrator) -> true
  save_positions = (false,save)
  DiscreteCallback(condtion, affect!;
                   initialize = Manifold_initialize,
                   save_positions=save_positions)
end

export ManifoldProjection
