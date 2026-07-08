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

@testset "Public API documentation coverage" begin
    public_names = Set(names(DiffEqCallbacks; all = false))
    delete!(public_names, :DiffEqCallbacks)

    doc_pages = joinpath(pkgdir(DiffEqCallbacks), "docs", "src")
    rendered_names = Set{Symbol}()
    for page in filter(endswith(".md"), readdir(doc_pages; join = true))
        in_docs_block = false
        for line in eachline(page)
            stripped = strip(line)
            if stripped == "```@docs"
                in_docs_block = true
                continue
            elseif in_docs_block && startswith(stripped, "```")
                in_docs_block = false
                continue
            end
            if in_docs_block && !isempty(stripped) && !startswith(stripped, "#")
                name = last(split(stripped, '.'))
                push!(rendered_names, Symbol(name))
            end
        end
    end

    @testset "docstrings exist" begin
        missing_docstrings = sort!(collect(filter(name -> !Docs.hasdoc(DiffEqCallbacks, name), public_names)))
        @test isempty(missing_docstrings)
    end

    @testset "rendered docs entries exist" begin
        missing_rendered_entries = sort!(collect(setdiff(public_names, rendered_names)))
        stale_rendered_entries = sort!(collect(setdiff(rendered_names, public_names)))
        @test isempty(missing_rendered_entries)
        @test isempty(stale_rendered_entries)
    end
end
