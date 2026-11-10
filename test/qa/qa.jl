using SciMLTesting, DiffEqCallbacks, Test

# Package extensions only exist as modules once their trigger weakdep is loaded, and
# ExplicitImports skips extensions it cannot resolve via `Base.get_extension`. Loading
# Functors and ForwardDiff here is what puts `DiffEqCallbacksFunctorsExt` and
# `DiffEqCallbacksForwardDiffExt` in scope for the QA checks.
using ForwardDiff, Functors

# ExplicitImports silently skips an extension that fails to load, so assert the
# extension modules actually exist rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    @test Base.get_extension(DiffEqCallbacks, :DiffEqCallbacksFunctorsExt) !== nothing
    @test Base.get_extension(
        DiffEqCallbacks, :DiffEqCallbacksForwardDiffExt
    ) !== nothing
end

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
                # `ForwardDiff` exports neither spelling; both are only used in
                # `DiffEqCallbacksForwardDiffExt`:
                :Dual,  # the concrete dual-number type, dispatched on
                :value,  # the primal-value accessor
            ),
        ),
        # `DiffEqCallbacksFunctorsExt` exists to implement DiffEqCallbacks' own internal
        # recursive-container generics for `Functors`-traversable parameters, so it must
        # import them from the parent package; they are deliberately internal and have no
        # public spelling. ExplicitImports' `allow_internal_imports` does not cover this
        # because it exempts only imports sharing a `Base.moduleroot`, and an extension
        # module is its own root rather than the parent package's. The same holds for
        # the `AutoAbstol` primal-value hook imported by
        # `DiffEqCallbacksForwardDiffExt`.
        all_explicit_imports_are_public = (;
            ignore = (
                :_autoabstol_value,
                :allocate_vjp,
                :allocate_vjp_internal,
                :allocate_zeros,
                :internal_add!,
                :internal_adjoint,
                :internal_allocate_zeros,
                :internal_axpy!,
                :internal_copy,
                :internal_copyto!,
                :internal_neg!,
                :internal_scalar_mul!,
                :internal_sub!,
                :internal_zero!,
                :recursive_add!,
                :recursive_adjoint,
                :recursive_axpy!,
                :recursive_copy,
                :recursive_copyto!,
                :recursive_neg!,
                :recursive_scalar_mul!,
                :recursive_sub!,
                :recursive_zero!,
            ),
        ),
    ),
)
