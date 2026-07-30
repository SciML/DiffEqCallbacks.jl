# Internal Implementations

The bindings on this page are implementation details used by DiffEqCallbacks itself.
They are not supported extension points or stable user-facing APIs. Do not build
downstream packages on them; use the documented callback constructors instead.

```@docs
DiffEqCallbacks.CachePool
DiffEqCallbacks.IndependentlyLinearizedSolutionChunks
DiffEqCallbacks.affect!
DiffEqCallbacks.allocate_vjp
DiffEqCallbacks.allocate_zeros
DiffEqCallbacks.g_weights
DiffEqCallbacks.gauss_points
DiffEqCallbacks.gauss_weights
DiffEqCallbacks.gk_points
DiffEqCallbacks.gk_weights
DiffEqCallbacks.isaccepted
DiffEqCallbacks.modify_u!
DiffEqCallbacks.recursive_add!
DiffEqCallbacks.recursive_adjoint
DiffEqCallbacks.recursive_copy
DiffEqCallbacks.recursive_copyto!
DiffEqCallbacks.recursive_neg!
DiffEqCallbacks.recursive_sub!
DiffEqCallbacks.recursive_zero!
DiffEqCallbacks.sample!
DiffEqCallbacks.seek_forward
DiffEqCallbacks.setup
DiffEqCallbacks.store!
```
