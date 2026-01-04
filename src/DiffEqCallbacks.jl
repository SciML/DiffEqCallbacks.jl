module DiffEqCallbacks

using ConcreteStructs: @concrete
using DataStructures: DataStructures, BinaryMaxHeap, BinaryMinHeap
using DiffEqBase: DiffEqBase, get_tstops, get_tstops_array, get_tstops_max
using DifferentiationInterface: DifferentiationInterface, Constant
using LinearAlgebra: LinearAlgebra, adjoint, axpy!, copyto!, diagind, mul!, ldiv!
using Markdown: @doc_str
using PrecompileTools: PrecompileTools
using RecipesBase: @recipe
using RecursiveArrayTools: RecursiveArrayTools, DiffEqArray, copyat_or_push!
using SciMLBase: SciMLBase, CallbackSet, DiscreteCallback, NonlinearFunction,
    NonlinearLeastSquaresProblem, NonlinearProblem, RODEProblem,
    ReturnCode, SDEProblem, add_tstop!, check_error, get_du,
    get_proposed_dt, get_tmp_cache, init, reinit!,
    set_proposed_dt!, solve!, terminate!, u_modified!
using StaticArraysCore: StaticArraysCore

const DI = DifferentiationInterface
true_condition(u, t, integrator) = true

include("functor_helpers.jl")
include("autoabstol.jl")
include("manifold.jl")
include("domain.jl")
include("stepsizelimiters.jl")
include("function_caller.jl")
include("independentlylinearizedutils.jl")
include("saving.jl")
include("integrating.jl")
include("integrating_sum.jl")
include("iterative_and_periodic.jl")
include("terminatesteadystate.jl")
include("preset_time.jl")
include("probints.jl")
include("integrating_GK_affect.jl")
include("integrating_GK_sum.jl")
include("precompilation.jl")

end # module
