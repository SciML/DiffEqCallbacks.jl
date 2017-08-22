__precompile__()

module DiffEqCallbacks

  using DiffEqBase, NLsolve, ForwardDiff
  import DiffBase

  import OrdinaryDiffEq: fix_dt_at_bounds!, modify_dt_for_tstops!

  include("autoabstol.jl")
  include("manifold.jl")
  include("domain.jl")
  include("stepsizelimiters.jl")

end # module
