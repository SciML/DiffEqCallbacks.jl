using SciMLTesting, DiffEqCallbacks, Test

# Package extensions only exist as modules once their trigger weakdep is loaded, and
# ExplicitImports skips extensions it cannot resolve via `Base.get_extension`. Loading
# Functors here is what puts `DiffEqCallbacksFunctorsExt` in scope for the QA checks.
using Functors

# ExplicitImports silently skips an extension that fails to load, so assert the
# extension modules actually exist rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    @test Base.get_extension(DiffEqCallbacks, :DiffEqCallbacksFunctorsExt) !== nothing
end

run_qa(
    DiffEqCallbacks;
    ei_kwargs = (;
        # `DiffEqCallbacksFunctorsExt` exists to implement DiffEqCallbacks' own internal
        # recursive-container generics for `Functors`-traversable parameters, so it must
        # import them from the parent package; they are deliberately internal and have no
        # public spelling. ExplicitImports' `allow_internal_imports` does not cover this
        # because it exempts only imports sharing a `Base.moduleroot`, and an extension
        # module is its own root rather than the parent package's.
        all_explicit_imports_are_public = (;
            ignore = (
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
