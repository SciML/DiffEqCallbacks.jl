module DiffEqCallbacks

  using DiffEqBase, RecursiveArrayTools, DataStructures, RecipesBase, StaticArrays,
        NLsolve, ForwardDiff

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
  include("preset_time.jl")

end # module
