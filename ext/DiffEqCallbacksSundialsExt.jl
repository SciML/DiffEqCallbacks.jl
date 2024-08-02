module DiffEqCallbacksSundialsExt

using Sundials: NVector, IDA
import DiffEqCallbacks: solver_state_alloc, solver_state_type

# Allocator; `U` is typically something like `Vector{Float64}`
solver_state_alloc(solver::IDA, U::DataType, num_us::Int) = () -> NVector(U(undef, num_us))

# Type of `solver_state_alloc`, which is just `NVector`
solver_state_type(solver::IDA, U::DataType) = NVector

end # module
