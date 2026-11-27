using Test, OrdinaryDiffEqTsit5, DiffEqCallbacks
using SciMLBase: SciMLBase, CallbackSet, instantiate_callback

# Callbacks with state give each integrator its own copy through `instantiate_callback`,
# which `init` and `solve` apply. The user's functions are shared, not copied.

f(u, p, t) = -u
calls = Tuple{Int, Float64}[]
record!(integrator) = (push!(calls, (integrator.p, integrator.t)); nothing)
times(id) = [t for (i, t) in calls if i == id]

@testset "Integrators alive at the same time: $name" for (name, cb) in (
        ("PeriodicCallback", PeriodicCallback(record!, 1.0)),
        ("IterativeCallback", IterativeCallback(integrator -> integrator.t + 1.0, record!)),
    )
    # Both integrators run on the same task, and the second starts while the first is alive
    empty!(calls)
    i1 = init(ODEProblem(f, 1.0, (0.0, 10.0), 1), Tsit5(); callback = cb)
    step!(i1, 2.5, true)
    i2 = init(ODEProblem(f, 1.0, (0.0, 10.0), 2), Tsit5(); callback = cb)
    step!(i1, 2.5, true)
    step!(i2, 2.5, true)
    @test times(1) == [1.0, 2.0, 3.0, 4.0, 5.0]
    @test times(2) == [1.0, 2.0]
end

@testset "FunctionCallingCallback keeps one position per integrator" begin
    seen = Tuple{Int, Float64}[]
    cb = FunctionCallingCallback(
        (u, t, integrator) -> push!(seen, (integrator.p, t));
        funcat = 0.0:0.5:2.0
    )
    integrators = [
        init(ODEProblem(f, 1.0, (0.0, 2.0), id), Tsit5(); callback = cb)
            for id in 1:2
    ]
    for _ in 1:3
        foreach(step!, integrators)
    end
    foreach(solve!, integrators)
    for id in 1:2
        @test [t for (i, t) in seen if i == id] == 0.0:0.5:2.0
    end
end

@testset "AutoAbstol tracks one maximum per integrator" begin
    g(u, p, t) = u
    cb = AutoAbstol()
    small, large = ODEProblem(g, 1.0, (0.0, 2.0)), ODEProblem(g, 1.0e3, (0.0, 2.0))
    ref = solve(small, Tsit5(); callback = cb)

    integrators = (init(small, Tsit5(); callback = cb), init(large, Tsit5(); callback = cb))
    for _ in 1:3
        foreach(step!, integrators)
    end
    foreach(solve!, integrators)
    @test integrators[1].sol.t == ref.t
    @test integrators[1].sol.u == ref.u
end

@testset "Fresh state and shared user functions: $name" for (name, cb) in (
        ("PeriodicCallback", PeriodicCallback(record!, 0.5; initial_affect = true)),
        ("IterativeCallback", IterativeCallback(integrator -> nothing, record!)),
    )
    c = instantiate_callback(cb)
    @test c.condition === c.affect! === c.initialize
    @test c.condition !== cb.condition
    @test c.save_positions == cb.save_positions
    # The user's function is stored as `user_affect!` or `affect!`
    function user_affect!(functor)
        return hasfield(typeof(functor), :user_affect!) ? functor.user_affect! :
            functor.affect!
    end
    @test user_affect!(c.condition) === user_affect!(cb.condition) === record!
end

@testset "Callbacks with state in their affect" begin
    dtFE(u, p, t) = 0.1
    func(u, t, integrator) = nothing
    for cb in (
            AutoAbstol(; init_curmax = [1.0, 2.0]), StepsizeLimiter(dtFE),
            FunctionCallingCallback(func; funcat = 0.5),
        )
        c = instantiate_callback(cb)
        @test c.affect! !== cb.affect!
        @test c.condition === cb.condition
        @test c.initialize === cb.initialize
    end

    cb = AutoAbstol(; init_curmax = [1.0, 2.0])
    c = instantiate_callback(cb)
    @test c.affect!.curmax == cb.affect!.curmax
    @test c.affect!.curmax !== cb.affect!.curmax

    cb = StepsizeLimiter(dtFE)
    @test instantiate_callback(cb).affect!.dtFE === dtFE

    cb = FunctionCallingCallback(func; funcat = 0.5)
    cb.affect!.funciter = 3
    c = instantiate_callback(cb)
    @test c.affect!.funciter == 1
    @test c.affect!.func === func
end

@testset "ManifoldProjection and domain callbacks" begin
    circle(resid, u, p, t) = (resid[1] = u[1]^2 + u[2]^2 - 1; nothing)
    cb = ManifoldProjection(circle; resid_prototype = [0.0])
    c = instantiate_callback(cb)
    @test c.affect! !== cb.affect!
    @test c.affect!.manifold !== cb.affect!.manifold
    @test c.affect!.manifold.f === circle
    @test c.affect!.nlcache === nothing

    # A CallbackSet of a manifold projection and a domain callback
    cbs = GeneralDomain(circle, [1.0, 0.0])
    cs = instantiate_callback(cbs)
    @test cs isa CallbackSet
    for (new, old) in zip(cs.discrete_callbacks, cbs.discrete_callbacks)
        @test new.affect! !== old.affect!
    end
    domain, old_domain = cs.discrete_callbacks[2].affect!, cbs.discrete_callbacks[2].affect!
    @test domain.u == old_domain.u
    @test domain.u !== old_domain.u
    @test domain.g !== old_domain.g
    @test domain.g.f === circle

    # PositiveDomain only has state when given `u`
    cb = PositiveDomain()
    @test instantiate_callback(cb) === cb
    cb = PositiveDomain([1.0, 2.0])
    @test instantiate_callback(cb).affect!.u !== cb.affect!.u
end
