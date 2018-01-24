mutable struct AutoAbstolAffect{T}
  curmax::T
end
# Now make `affect!` for this:
function (p::AutoAbstolAffect)(integrator)
  if typeof(p.curmax) <: AbstractArray
    @. p.curmax = max(p.curmax,abs(integrator.u))
  else
    p.curmax = max(p.curmax,maximum(abs.(integrator.u)))
  end

  if typeof(integrator.opts.abstol) <: AbstractArray
    integrator.opts.abstol .= p.curmax .* integrator.opts.reltol
  else
    integrator.opts.abstol = p.curmax .* integrator.opts.reltol
  end

  u_modified!(integrator,false)
end

function AutoAbstol_initialize(cb,u,t,integrator)
  if cb.affect!.curmax == zero(integrator.opts.abstol)
    cb.affect!.curmax = integrator.opts.abstol
  end
  u_modified!(integrator,false)
end

function AutoAbstol(save=true;init_curmax=0.0)
  affect! = AutoAbstolAffect(abs.(init_curmax))
  condtion = (u,t,integrator) -> true
  save_positions = (save,false)
  DiscreteCallback(condtion,affect!;
                   initialize = AutoAbstol_initialize,
                   save_positions=save_positions)
end

export AutoAbstol
