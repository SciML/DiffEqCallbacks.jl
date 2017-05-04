__precompile__()

module DiffEqCallbacks

  using DiffEqBase, NLsolve, ForwardDiff
  import DiffBase

  include("autoabstol.jl")
  include("manifold.jl")

end # module
