"""
    gk_points::Vector{Vector{Float64}}

Precomputed Gaussian-Kronrod nodes up to degree 3*10-1 = 29.
Computed using QuadGK.jl with the command `[kronrod(i,-1,1)[1] for i in 1:10]`
"""
gk_points = [[-0.7745966692414833, 0.0, 0.7745966692414834],
    [-0.9258200997725514, -0.5773502691896257, 0.0, 0.5773502691896257, 0.9258200997725514],
    [-0.9604912687080203, -0.7745966692414834, -0.43424374934680254,
        0.0, 0.43424374934680254, 0.7745966692414834, 0.9604912687080203],
    [-0.9765602507375731, -0.8611363115940526, -0.64028621749631, -0.33998104358485626,
        0.0, 0.33998104358485626, 0.64028621749631, 0.8611363115940525, 0.976560250737573],
    [-0.9840853600948425, -0.906179845938664, -0.7541667265708493,
        -0.5384693101056831, -0.2796304131617833, 0.0, 0.2796304131617833,
        0.538469310105683, 0.7541667265708494, 0.9061798459386639, 0.9840853600948425],
    [-0.9887032026126789, -0.932469514203152, -0.8213733408650279,
        -0.6612093864662645, -0.4631182124753046, -0.238619186083197,
        0.0, 0.238619186083197, 0.4631182124753046, 0.6612093864662645,
        0.8213733408650279, 0.932469514203152, 0.9887032026126787],
    [-0.9914553711208126, -0.9491079123427585, -0.8648644233597691, -0.7415311855993945,
        -0.5860872354676911, -0.4058451513773972, -0.2077849550078985,
        0.0, 0.2077849550078985, 0.4058451513773971, 0.5860872354676911,
        0.7415311855993945, 0.8648644233597691, 0.9491079123427584, 0.9914553711208125],
    [-0.9933798758817161, -0.9602898564975362, -0.8941209068474564, -0.7966664774136267,
        -0.6723540709451586, -0.525532409916329, -0.360701097928132, -0.18343464249564978,
        0.0, 0.1834346424956499, 0.360701097928132, 0.525532409916329, 0.6723540709451585,
        0.7966664774136267, 0.8941209068474563, 0.9602898564975362, 0.9933798758817161],
    [-0.9946781606773403, -0.9681602395076261, -0.9149635072496779, -0.8360311073266358,
        -0.7344867651839337, -0.6133714327005905, -0.47546247911245987,
        -0.324253423403809, -0.1642235636149867, 0.0, 0.1642235636149867,
        0.324253423403809, 0.47546247911245976, 0.6133714327005904, 0.7344867651839337,
        0.8360311073266358, 0.914963507249678, 0.9681602395076261, 0.9946781606773403],
    [-0.9956571630258081, -0.9739065285171717, -0.9301574913557082,
        -0.8650633666889844, -0.7808177265864169, -0.6794095682990243,
        -0.5627571346686047, -0.4333953941292472, -0.29439286270146026,
        -0.14887433898163116, 0.0, 0.14887433898163116, 0.29439286270146026,
        0.4333953941292472, 0.5627571346686047, 0.6794095682990244, 0.7808177265864169,
        0.8650633666889844, 0.9301574913557082, 0.9739065285171717, 0.995657163025808]]

"""
    gk_weights::Vector{Vector{Float64}}

Precomputed Gaussian-Kronrod node weights up to degree 3*10-1 = 29.
Computed using QuadGK.jl with the command `[kronrod(i,-1,1)[2] for i in 1:10]`
"""
gk_weights = [[0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
    [0.19797979797979798, 0.4909090909090911, 0.6222222222222223,
        0.4909090909090911, 0.19797979797979798],
    [0.10465622602646725, 0.26848808986833345, 0.4013974147759622, 0.45091653865847414,
        0.4013974147759622, 0.26848808986833345, 0.10465622602646725],
    [0.06297737366547303, 0.1700536053357228, 0.26679834045228457,
        0.32694918960145164, 0.3464429818901364, 0.32694918960145164,
        0.26679834045228457, 0.1700536053357228, 0.06297737366547303],
    [0.04258203675108178, 0.11523331662247335, 0.18680079655649262, 0.24104033922864765,
        0.2728498019125588, 0.28298741785749126, 0.2728498019125588, 0.24104033922864765,
        0.18680079655649262, 0.11523331662247335, 0.04258203675108178],
    [0.03039615411981984, 0.08369444044690652, 0.13732060463444698,
        0.1810719943231376, 0.21320965227196234, 0.23377086411699424,
        0.2410725801734648, 0.23377086411699424, 0.21320965227196234, 0.1810719943231376,
        0.13732060463444698, 0.08369444044690652, 0.03039615411981984],
    [0.022935322010529256, 0.06309209262997842, 0.10479001032225017, 0.14065325971552592,
        0.16900472663926788, 0.19035057806478559, 0.20443294007529877, 0.20948214108472793,
        0.20443294007529877, 0.19035057806478559, 0.16900472663926788, 0.14065325971552592,
        0.10479001032225017, 0.06309209262997842, 0.022935322010529256],
    [0.017822383320710525, 0.049439395002139175, 0.08248229893135833,
        0.11164637082683965, 0.13626310925517227, 0.15665260616818855,
        0.17207060855521134, 0.18140002506803451, 0.1844464057446918, 0.18140002506803451,
        0.17207060855521134, 0.15665260616818855, 0.13626310925517227, 0.11164637082683965,
        0.08248229893135833, 0.049439395002139175, 0.017822383320710525],
    [0.014304775643838873, 0.03963189516026116, 0.06651815594027412, 0.09079068168872645,
        0.11178913468441828, 0.13000140685534115, 0.1452395883843662, 0.1564135277884838,
        0.16286282744011493, 0.16489601282834956, 0.16286282744011493, 0.1564135277884838,
        0.1452395883843662, 0.13000140685534115, 0.11178913468441828, 0.09079068168872645,
        0.06651815594027412, 0.03963189516026116, 0.014304775643838873],
    [0.011694638867371846, 0.03255816230796465, 0.05475589657435192,
        0.07503967481091997, 0.09312545458369768, 0.10938715880229767,
        0.12349197626206591, 0.1347092173114734, 0.14277593857706, 0.1477391049013384,
        0.0, 0.1477391049013384, 0.14277593857706, 0.1347092173114734,
        0.12349197626206591, 0.10938715880229767, 0.09312545458369768, 0.07503967481091997,
        0.05475589657435192, 0.03255816230796465, 0.011694638867371846]]
"""
    g_weights::Vector{Vector{Float64}}

Precomputed respective Gaussian node weights up to degree 2*10-1 = 19.
Computed using QuadGK.jl with the command `[kronrod(i,-1,1)[3] for i in 1:10]`
"""
g_weights = [[2.0],
    [1.0000000000000002, 1.0000000000000002],
    [0.5555555555555556, 0.8888888888888885, 0.5555555555555556],
    [0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454],
    [0.2369268850561892, 0.4786286704993663, 0.5688888888888887,
        0.4786286704993663, 0.2369268850561892],
    [0.17132449237917044, 0.36076157304813855, 0.4679139345726909,
        0.4679139345726909, 0.36076157304813855, 0.17132449237917044],
    [0.12948496616886981, 0.2797053914892767, 0.38183005050511887, 0.41795918367346907,
        0.38183005050511887, 0.2797053914892767, 0.12948496616886981],
    [0.10122853629037629, 0.22238103445337457, 0.3137066458778873, 0.3626837833783619,
        0.3626837833783619, 0.3137066458778873, 0.22238103445337457, 0.10122853629037629],
    [0.08127438836157448, 0.1806481606948575, 0.26061069640293544,
        0.31234707704000275, 0.3302393550012595, 0.31234707704000275,
        0.26061069640293544, 0.1806481606948575, 0.08127438836157448],
    [0.06667134430868811, 0.14945134915058084, 0.2190863625159821, 0.26926671930999635,
        0.29552422471475276, 0.29552422471475276, 0.26926671930999635,
        0.2190863625159821, 0.14945134915058084, 0.06667134430868811]]

mutable struct SavingIntegrandGKAffect{
    IntegrandFunc,
    tType,
    IntegrandType,
    IntegrandCacheType
}
    integrand_func::IntegrandFunc
    integrand_values::IntegrandValues{tType, IntegrandType}
    integrand_cache::IntegrandCacheType
    accumulation_cache::IntegrandCacheType
    gk_step_cache::IntegrandCacheType
    gk_err_cache::IntegrandCacheType
    tol::Float64
end

function integrate_gk!(affect!::SavingIntegrandGKAffect, integrator,
        bound_l, bound_r; order = 7, tol = 1e-7)
    affect!.gk_step_cache = recursive_zero!(affect!.gk_step_cache)
    affect!.gk_err_cache = recursive_zero!(affect!.gk_err_cache)
    for i in 1:(2 * order + 1)
        t_temp = (gk_points[order][i]+1)*((bound_r-bound_l)/2) + bound_l
        if DiffEqBase.isinplace(integrator.sol.prob)
            curu = first(get_tmp_cache(integrator))
            integrator(curu, t_temp)
            if affect!.integrand_cache == nothing
                recursive_axpy!(gk_weights[order][i],
                    affect!.integrand_func(curu, t_temp, integrator), affect!.gk_step_cache)
                if i%2==0
                    recursive_axpy!(g_weights[order][div(i, 2)],
                        affect!.integrand_func(curu, t_temp, integrator), affect!.gk_err_cache)
                end
            else
                affect!.integrand_func(affect!.integrand_cache, curu, t_temp, integrator)
                recursive_axpy!(gk_weights[order][i],
                    affect!.integrand_cache, affect!.gk_step_cache)
                if i%2==0
                    recursive_axpy!(g_weights[order][div(i, 2)],
                        affect!.integrand_cache, affect!.gk_err_cache)
                end
            end
        else
            recursive_axpy!(gk_weights[order][i],
                affect!.integrand_func(integrator(t_temp), t_temp, integrator), affect!.gk_step_cache)
            if i%2==0
                recursive_axpy!(g_weights[order][div(i, 2)],
                    affect!.integrand_func(integrator(t_temp), t_temp, integrator), affect!.gk_err_cache)
            end
        end
    end
    if sum(abs.((affect!.gk_step_cache .- affect!.gk_err_cache) .* (bound_r-bound_l) ./
                2))<tol
        recursive_axpy!(
            1, affect!.gk_step_cache .* (bound_r-bound_l) ./ 2, affect!.accumulation_cache)
    else
        integrate_gk!(
            affect!, integrator, bound_l, (bound_l+bound_r)/2, order = order, tol = tol/2)
        integrate_gk!(
            affect!, integrator, (bound_l+bound_r)/2, bound_r, order = order, tol = tol/2)
    end
end

function (affect!::SavingIntegrandGKAffect)(integrator)
    n = 0
    if integrator.sol.prob isa Union{SDEProblem, RODEProblem}
        throw("Gauss-Kronrod algorithm is not necessarily convergent for this problem type")
    else
        n = div(SciMLBase.alg_order(integrator.alg) + 1, 2)
    end
    accumulation_cache = recursive_zero!(affect!.accumulation_cache)
    integrate_gk!(
        affect!, integrator, integrator.tprev, integrator.t, order = n, tol = affect!.tol)
    push!(affect!.integrand_values.ts, integrator.t)
    push!(affect!.integrand_values.integrand, recursive_copy(affect!.accumulation_cache))
    u_modified!(integrator, false)
end

"""
```julia
IntegratingGKCallback(integrand_func,
    integrand_values::IntegrandValues, integrand_prototype)
```

Let one define a function `integrand_func(u, t, integrator)::typeof(integrand_prototype)` which
returns Integral(integrand_func(u(t),t)dt) over the problem tspan.

## Arguments

  - `integrand_func(out, u, t, integrator)` for in-place problems and `out = integrand_func(u, t, integrator)` for
    out-of-place problems. Returns the quantity in the integral for computing dG/dp.
    Note that for out-of-place problems, this should allocate the output (not as a view to `u`).
  - `integrand_values::IntegrandValues` is the types that `integrand_func` will return, i.e.
    `integrand_func(t, u, integrator)::integrandType`. It's specified via
    `IntegrandValues(integrandType)`, i.e. give the type
    that `integrand_func` will output (or higher compatible type).
  - `integrand_prototype` is a prototype of the output from the integrand.

The outputted values are saved into `integrand_values`. The values are found
via `integrand_values.integrand`.

!!! note

    Method has automatic error control (h-adaptive quadrature).

    This method is currently limited to ODE solvers of order 10 or lower. Open an issue if other
    solvers are required.

    If `integrand_func` is in-place, you must use `cache` to store the output of `integrand_func`.
"""
function IntegratingGKCallback(
        integrand_func, integrand_values::IntegrandValues, integrand_prototype, tol = 1e-7)
    affect! = SavingIntegrandGKAffect(
        integrand_func, integrand_values, integrand_prototype,
        allocate_zeros(integrand_prototype), allocate_zeros(integrand_prototype), allocate_zeros(integrand_prototype), tol)
    condition = true_condition
    DiscreteCallback(condition, affect!, save_positions = (false, false))
end

export IntegratingGKCallback
