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
    set_proposed_dt!, solve!, terminate!
using StaticArraysCore: StaticArraysCore

# SciMLBase v3 renamed `u_modified!` → `derivative_discontinuity!` (with
# an `@deprecate` on the old name). Support both SciMLBase v2 and v3 by
# aliasing to whichever name the loaded SciMLBase provides.
@static if isdefined(SciMLBase, :derivative_discontinuity!)
    using SciMLBase: derivative_discontinuity!
else
    const derivative_discontinuity! = SciMLBase.u_modified!
end

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
