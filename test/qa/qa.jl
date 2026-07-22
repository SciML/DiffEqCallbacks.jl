using SciMLTesting, DiffEqCallbacks

run_qa(
    DiffEqCallbacks;
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
