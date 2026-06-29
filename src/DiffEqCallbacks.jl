module DiffEqCallbacks

using ConcreteStructs: @concrete
using DataStructures: DataStructures, BinaryMaxHeap, BinaryMinHeap
using DiffEqBase: get_tstops, get_tstops_array, get_tstops_max
using DifferentiationInterface: DifferentiationInterface, Constant
using LinearAlgebra: LinearAlgebra, adjoint, axpy!, copyto!, diagind, mul!
using Markdown: @doc_str
using PrecompileTools: PrecompileTools
using RecipesBase: @recipe
using RecursiveArrayTools: RecursiveArrayTools, DiffEqArray, copyat_or_push!
using SciMLBase: SciMLBase, CallbackSet, DiscreteCallback, DiscreteProblem,
    NonlinearFunction, NonlinearProblem, RODEProblem,
    ReturnCode, SDEProblem, add_tstop!, check_error, get_du,
    get_proposed_dt, get_tmp_cache,
    set_proposed_dt!, terminate!
using StaticArraysCore: StaticArraysCore

# SciMLBase v3 renamed `u_modified!` → `derivative_discontinuity!` (with
# an `@deprecate` on the old name). Support both SciMLBase v2 and v3 by
# binding a const to whichever name the loaded SciMLBase provides. Both
# `derivative_discontinuity!` and `u_modified!` are public in their
# respective SciMLBase versions, so the qualified access is to public API; a
# `const` (rather than a `using` import on one branch) keeps the name a single
# local binding and avoids a stale-import false positive.
const derivative_discontinuity! = if isdefined(SciMLBase, :derivative_discontinuity!)
    SciMLBase.derivative_discontinuity!
else
    SciMLBase.u_modified!
end

const DI = DifferentiationInterface
true_condition(u, t, integrator) = true

# Local reproduction of SciMLBase's `_unwrap_val` (a leading-underscore internal
# that will not be made public): pull the value out of a `Val`, pass anything
# else through unchanged.
_unwrap_val(B) = B
_unwrap_val(::Val{B}) where {B} = B

# No-op default `initialize` for `DiscreteCallback`, matching SciMLBase's
# (non-public) `INITIALIZE_DEFAULT`: reset the derivative-discontinuity flag.
INITIALIZE_DEFAULT(cb, u, t, integrator) = derivative_discontinuity!(integrator, false)

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
