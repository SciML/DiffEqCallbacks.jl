module DiffEqCallbacks

  using DiffEqBase, RecursiveArrayTools, DataStructures, RecipesBase, StaticArrays,
        NLsolve, ForwardDiff

  import Base.Iterators

  using Parameters: @unpack

  import SciMLBase

  import OrdinaryDiffEq
  using OrdinaryDiffEq: ODEIntegrator

  include("autoabstol.jl")
  include("manifold.jl")
  include("domain.jl")
  include("stepsizelimiters.jl")
  include("function_caller.jl")
  include("saving.jl")
  include("iterative_and_periodic.jl")
  include("terminatesteadystate.jl")
  include("preset_time.jl")

end # module
