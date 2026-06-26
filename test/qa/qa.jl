using SciMLTesting, DiffEqCallbacks, Test

run_qa(
    DiffEqCallbacks;
    explicit_imports = true,
    # JET is covered by the curated `@test_opt` constructor type-stability checks in
    # jet_tests.jl (which `run_qa`'s `JET.test_package` error analysis does not
    # subsume); keep `run_qa`'s JET off so loading JET there does not auto-enable it.
    jet = false,
    ei_kwargs = (;
        # `derivative_discontinuity!` is imported via `using SciMLBase:` inside a
        # `@static if isdefined(SciMLBase, :derivative_discontinuity!)` block (the
        # SciMLBase v2/v3 rename shim); ExplicitImports' static scan flags the import
        # as stale because the `else` branch also binds the name as a `const`, but it
        # is used bare on SciMLBase v3.
        no_stale_explicit_imports = (; ignore = (:derivative_discontinuity!,)),
        # Names that are still non-public in their owning module on the released
        # versions (verified on Julia 1.12 / SciMLBase 3.24.0, DiffEqBase 7.5.7,
        # LinearAlgebra+Base stdlib): go public as those packages release.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDEProblem, :AbstractODEIntegrator, :INITIALIZE_DEFAULT,
                :_unwrap_val, :alg_order, :isadaptive,   # SciMLBase internals
                :QRCompactWY,                            # LinearAlgebra internal
                :RefValue,                               # Base internal
            ),
        ),
        # DiffEqBase internals imported explicitly; go public as DiffEqBase releases.
        all_explicit_imports_are_public = (;
            ignore = (:get_tstops, :get_tstops_array, :get_tstops_max),
        ),
    ),
)
