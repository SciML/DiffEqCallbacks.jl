__precompile__()

module DiffEqCallbacks

  using DiffEqBase, NLsolve, ForwardDiff, RecursiveArrayTools, DataStructures
  import DiffResults

  import OrdinaryDiffEq: fix_dt_at_bounds!, modify_dt_for_tstops!, ode_addsteps!, ode_interpolant

  include("autoabstol.jl")
  include("manifold.jl")
  include("domain.jl")
  include("stepsizelimiters.jl")
  include("saving.jl")

end # module
