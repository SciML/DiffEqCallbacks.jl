###########

# NLsolve utils

Base.@pure function determine_chunksize(u,CS)
  if CS != 0
    return CS
  else
    return ForwardDiff.pickchunksize(length(u))
  end
end

function autodiff_setup(f!, initial_x,chunk_size::Type{Val{CS}}) where CS

    permf! = (fx, x) -> f!(reshape(x,size(initial_x)...), fx)

    fx2 = copy(initial_x)
    jac_cfg = ForwardDiff.JacobianConfig(f!, initial_x, ForwardDiff.Chunk{CS}())
    g! = (x, gx) -> ForwardDiff.jacobian!(gx, permf!, fx2, x, jac_cfg)

    fg! = (x, fx, gx) -> begin
        jac_res = DiffBase.DiffResult(fx, gx)
        ForwardDiff.jacobian!(jac_res, permf!, fx2, x, jac_cfg)
        DiffBase.value(jac_res)
    end

    return DifferentiableMultivariateFunction((x,resid)->f!(reshape(x,size(initial_x)...),
                                                            resid),
                                              g!, fg!)
end

function non_autodiff_setup(f!, initial_x)
  DifferentiableMultivariateFunction((x,resid)->f!(reshape(x,size(initial_x)...), resid))
end

struct NLSOLVEJL_SETUP{CS,AD} end
Base.@pure NLSOLVEJL_SETUP(;chunk_size=0,autodiff=true) = NLSOLVEJL_SETUP{chunk_size,autodiff}()
(p::NLSOLVEJL_SETUP)(f, u0; kwargs...) = (res=NLsolve.nlsolve(f, u0; kwargs...); res.zero)
function (p::NLSOLVEJL_SETUP{CS,AD})(::Type{Val{:init}},f,u0_prototype) where {CS,AD}
  if AD
    return non_autodiff_setup(f,u0_prototype)
  else
    return autodiff_setup(f,u0_prototype,Val{determine_chunksize(initial_x,CS)})
  end
end

get_chunksize(x) = 0
get_chunksize(x::NLSOLVEJL_SETUP{CS,AD}) where {CS,AD} = CS

#########################

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

  nlres = reshape(p.nlsolve(p.nl_rhs, vec(integrator.u); p.nlopts...),
                  size(integrator.u)...)::typeof(integrator.u)
  integrator.u .= nlres
end

function Manifold_initialize(cb,t,u,integrator)
  cb.affect!.nl_rhs = cb.affect!.nlsolve(Val{:init}, cb.affect!.g, u)
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
