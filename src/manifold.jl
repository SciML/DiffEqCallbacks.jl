###########

# NLsolve utils

Base.@pure function determine_chunksize(u,CS)
  if CS != 0
    return CS
  else
    return ForwardDiff.pickchunksize(length(u))
  end
end

function autodiff_setup{CS}(f!, initial_x::Vector,chunk_size::Type{Val{CS}})

    permf! = (fx, x) -> f!(x, fx)

    fx2 = copy(initial_x)
    jac_cfg = ForwardDiff.JacobianConfig(f!, initial_x, ForwardDiff.Chunk{CS}())
    g! = (x, gx) -> ForwardDiff.jacobian!(gx, permf!, fx2, x, jac_cfg)

    fg! = (x, fx, gx) -> begin
        jac_res = DiffBase.DiffResult(fx, gx)
        ForwardDiff.jacobian!(jac_res, permf!, fx2, x, jac_cfg)
        DiffBase.value(jac_res)
    end

    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

function non_autodiff_setup(f!, initial_x::Vector)
  DifferentiableMultivariateFunction(f!)
end

immutable NLSOLVEJL_SETUP{CS,AD} end
Base.@pure NLSOLVEJL_SETUP(;chunk_size=0,autodiff=true) = NLSOLVEJL_SETUP{chunk_size,autodiff}()
(p::NLSOLVEJL_SETUP)(f,u0) = (res=NLsolve.nlsolve(f,u0); res.zero)
function (p::NLSOLVEJL_SETUP{CS,AD}){CS,AD}(::Type{Val{:init}},f,u0_prototype)
  if AD
    return non_autodiff_setup(f,u0_prototype)
  else
    return autodiff_setup(f,u0_prototype,Val{determine_chunksize(initial_x,CS)})
  end
end

get_chunksize(x) = 0
get_chunksize{CS,AD}(x::NLSOLVEJL_SETUP{CS,AD}) = CS

#########################

type ManifoldProjection{NL}
  nl_rhs
  nlsolve::NL
end
# Now make `affect!` for this:
function (p::ManifoldProjection)(integrator)
  nlres = p.nlsolve(p.nl_rhs,integrator.u)::typeof(integrator.u)
  integrator.u .= nlres
end

function Manifold_initialize(cb,t,u,integrator)
  cb.affect!.nl_rhs = cb.affect!.nlsolve(Val{:init},cb.affect!.nl_rhs,integrator.u)
end

function ManifoldProjection(g;nlsolve=NLSOLVEJL_SETUP(),save=true)
  affect! = ManifoldProjection(g,nlsolve)
  condtion = (t,u,integrator) -> true
  save_positions = (false,save)
  DiscreteCallback(condtion,affect!;
                   initialize = Manifold_initialize,
                   save_positions=save_positions)
end

export ManifoldProjection
