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
        # Names still non-public in their owning module on the released versions
        # (verified on Julia 1.12 / SciMLBase 3.27.0, DiffEqBase 7.6.0, LinearAlgebra +
        # Base stdlib): they go public as those packages release.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractODEIntegrator, :INITIALIZE_DEFAULT, :_unwrap_val,  # SciMLBase internals
                :QRCompactWY,                                              # LinearAlgebra internal
                :RefValue,                                                 # Base internal
            ),
        ),
    ),
)
