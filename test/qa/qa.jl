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
        # Other packages' non-public names accessed in qualified form; go public as
        # those packages release. The `Success`/`ConvergenceFailure`/`InitialFailure`/
        # `successful_retcode`/`Iterators.reverse` group is `public` in its owning module
        # on Julia 1.11+ but flagged on the LTS (1.10, no `public` keyword), so it must
        # be ignored for the QA lane's lts run.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDEProblem, :AbstractODEIntegrator, :INITIALIZE_DEFAULT,
                :_unwrap_val, :alg_order, :isadaptive, :numargs,            # SciMLBase internals
                :successful_retcode,                                        # SciMLBase (public on 1.11+)
                :Success, :ConvergenceFailure, :InitialFailure,             # SciMLBase.ReturnCode (public on 1.11+)
                :QRCompactWY,                                               # LinearAlgebra internal
                :RefValue, :depwarn,                                        # Base internals
                :reverse,                                                   # Base.Iterators (public on 1.11+)
            ),
        ),
        # DiffEqBase internals imported explicitly; go public as DiffEqBase releases.
        all_explicit_imports_are_public = (;
            ignore = (:get_tstops, :get_tstops_array, :get_tstops_max),
        ),
    ),
)
