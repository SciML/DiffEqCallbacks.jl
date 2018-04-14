__precompile__()

module DiffEqCallbacks

  using DiffEqBase, RecursiveArrayTools, DataStructures, RecipesBase, StaticArrays

  import Base.Iterators

  import OrdinaryDiffEq: fix_dt_at_bounds!, modify_dt_for_tstops!, ode_addsteps!,
                         ode_interpolant, NLSOLVEJL_SETUP, ODEIntegrator

  include("autoabstol.jl")
  include("manifold.jl")
  include("domain.jl")
  include("stepsizelimiters.jl")
  include("function_caller.jl")
  include("saving.jl")
  include("iterative_and_periodic.jl")
  include("terminatesteadystate.jl")

end # module
