using SciMLTesting, DiffEqCallbacks, Test

run_qa(
    DiffEqCallbacks;
    explicit_imports = true,
    # JET is covered by the curated `@test_opt` constructor type-stability checks in
    # jet_tests.jl (which `run_qa`'s `JET.test_package` error analysis does not
    # subsume); keep `run_qa`'s JET off so loading JET there does not auto-enable it.
    jet = false,
    ei_kwargs = (;
        # The only remaining non-public qualified accesses are to concrete stdlib
        # types used for method dispatch, for which there is no public spelling
        # (verified on Julia 1.12):
        #   `LinearAlgebra.QRCompactWY` — the concrete result type of `qr(A)` for a
        #     dense matrix; `fact_successful` dispatches on it to read `.factors`.
        #   `Base.RefValue`            — the concrete type behind `Ref`; used in a
        #     parametric `NamedTuple` type alias where the abstract `Ref` will not do.
        all_qualified_accesses_are_public = (;
            ignore = (
                :QRCompactWY,  # LinearAlgebra internal concrete factorization type
                :RefValue,     # Base internal concrete Ref type
            ),
        ),
    ),
)
