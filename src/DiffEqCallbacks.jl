__precompile__()

module DiffEqCallbacks

  using DiffEqBase, RecursiveArrayTools, DataStructures, RecipesBase, StaticArrays,
        NLsolve

  import Base.Iterators

  import OrdinaryDiffEq: fix_dt_at_bounds!, modify_dt_for_tstops!,
                         ODEIntegrator

  include("autoabstol.jl")
  include("manifold.jl")
  include("domain.jl")
  include("stepsizelimiters.jl")
  include("function_caller.jl")
  include("saving.jl")
  include("iterative_and_periodic.jl")
  include("terminatesteadystate.jl")

end # module
