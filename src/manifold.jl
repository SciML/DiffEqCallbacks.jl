###########

# NLsolve utils

Base.@pure function determine_chunksize(u,CS)
  if CS != 0
    return CS
  else
    return ForwardDiff.pickchunksize(length(u))
  end
end

function autodiff_setup(f!, initial_x, chunk_size::Type{Val{CS}}) where CS

    fvec! = NLsolve.reshape_f(f!, initial_x)
    permf! = (fx::AbstractVector, x::AbstractVector) -> fvec!(x, fx)

    fx2 = vec(copy(initial_x))
    jac_cfg = ForwardDiff.JacobianConfig(nothing, vec(initial_x), vec(initial_x),
                                         ForwardDiff.Chunk{CS}())
    function g!(x::AbstractVector, gx::AbstractMatrix)
        ForwardDiff.jacobian!(gx, permf!, fx2, x, jac_cfg)
    end

    fg! = (x::AbstractVector, fx::AbstractVector, gx::AbstractMatrix) -> begin
        jac_res = DiffBase.DiffResult(fx, gx)
        ForwardDiff.jacobian!(jac_res, permf!, fx2, x, jac_cfg)
        DiffBase.value(jac_res)
    end

    return DifferentiableMultivariateFunction(fvec!, g!, fg!)
end

non_autodiff_setup(f!, initial_x) = DifferentiableMultivariateFunction(f!, initial_x)

struct NLSOLVEJL_SETUP{CS,AD} end
Base.@pure NLSOLVEJL_SETUP(;chunk_size=0,autodiff=true) = NLSOLVEJL_SETUP{chunk_size,autodiff}()
(p::NLSOLVEJL_SETUP)(f, u0; kwargs...) = (res=NLsolve.nlsolve(f, u0; kwargs...); res.zero)
function (p::NLSOLVEJL_SETUP{CS,AD})(::Type{Val{:init}},f,u0_prototype) where {CS,AD}
  if AD
    return autodiff_setup(f, u0_prototype, Val{determine_chunksize(u0_prototype, CS)})
  else
    return non_autodiff_setup(f, u0_prototype)
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
