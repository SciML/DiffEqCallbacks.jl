__precompile__()

module DiffEqCallbacks

  using DiffEqBase, NLsolve, ForwardDiff

  include("autoabstol.jl")
  include("manifold.jl")

end # module
