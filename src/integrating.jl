"""
    gauss_points::Vector{Vector{Float64}}

Precomputed Gaussian nodes up to degree 2*10-1 = 19.
Computed using FastGaussQuadrature.jl with the command `[gausslegendre(i)[1] for i in 1:10]`
"""
gauss_points = [[0.0],
    [-0.5773502691896258, 0.5773502691896258],
    [-0.7745966692414834, 0.0, 0.7745966692414834],
    [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
    [-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664],
    [
        -0.932469514203152,
        -0.6612093864662645,
        -0.2386191860831969,
        0.2386191860831969,
        0.6612093864662645,
        0.932469514203152,
    ],
    [
        -0.9491079123427586,
        -0.7415311855993945,
        -0.4058451513773972,
        0.0,
        0.4058451513773972,
        0.7415311855993945,
        0.9491079123427586,
    ],
    [
        -0.9602898564975363,
        -0.7966664774136267,
        -0.525532409916329,
        -0.1834346424956498,
        0.1834346424956498,
        0.525532409916329,
        0.7966664774136267,
        0.9602898564975363,
    ],
    [
        -0.9681602395076261,
        -0.8360311073266358,
        -0.6133714327005904,
        -0.3242534234038089,
        0.0,
        0.3242534234038089,
        0.6133714327005904,
        0.8360311073266358,
        0.9681602395076261,
    ],
    [
        -0.9739065285171717,
        -0.8650633666889845,
        -0.6794095682990244,
        -0.4333953941292472,
        -0.14887433898163122,
        0.14887433898163122,
        0.4333953941292472,
        0.6794095682990244,
        0.8650633666889845,
        0.9739065285171717,
    ]]
"""
    gauss_weights::Vector{Vector{Float64}}

Precomputed Gaussian node weights up to degree 2*10-1 = 19.
Computed using FastGaussQuadrature.jl with the command `[gausslegendre(i)[2] for i in 1:10]`
"""
gauss_weights = [[2.0],
    [1.0, 1.0],
    [0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
    [0.34785484513745385, 0.6521451548625462, 0.6521451548625462, 0.34785484513745385],
    [
        0.23692688505618908,
        0.47862867049936647,
        0.5688888888888889,
        0.47862867049936647,
        0.23692688505618908,
    ],
    [
        0.17132449237917025,
        0.3607615730481385,
        0.46791393457269126,
        0.46791393457269126,
        0.3607615730481385,
        0.17132449237917025,
    ],
    [
        0.1294849661688702,
        0.2797053914892766,
        0.3818300505051189,
        0.4179591836734694,
        0.3818300505051189,
        0.2797053914892766,
        0.1294849661688702,
    ],
    [
        0.10122853629037676,
        0.22238103445337445,
        0.31370664587788744,
        0.36268378337836193,
        0.36268378337836193,
        0.31370664587788744,
        0.22238103445337445,
        0.10122853629037676,
    ],
    [
        0.08127438836157437,
        0.18064816069485742,
        0.2606106964029354,
        0.31234707704000275,
        0.3302393550012598,
        0.31234707704000275,
        0.2606106964029354,
        0.18064816069485742,
        0.08127438836157437,
    ],
    [
        0.06667134430868821,
        0.14945134915058056,
        0.21908636251598207,
        0.2692667193099965,
        0.2955242247147529,
        0.2955242247147529,
        0.2692667193099965,
        0.21908636251598207,
        0.14945134915058056,
        0.06667134430868821,
    ]]

"""
    IntegrandValues{integrandType}

A struct used to save values of the integrand values in `integrand::Vector{integrandType}`.
"""
struct IntegrandValues{integrandType}
    integrand::Vector{integrandType}
end

"""
    IntegrandValues(integrandType::DataType)

Return `IntegrandValues{integrandType}` with empty storage vectors.
"""
function IntegrandValues(::Type{integrandType}) where {integrandType}
    IntegrandValues{integrandType}(Vector{integrandType}())
end

function Base.show(io::IO, integrand_values::IntegrandValues)
    integrandType = eltype(integrand_values.integrand)
    print(io, "IntegrandValues{integrandType=", integrandType, "}",
        "\nintegrand:\n", integrand_values.integrand)
end

mutable struct SavingIntegrandAffect{IntegrandFunc, integrandType, integrandCacheType}
    integrand_func::IntegrandFunc
    integrand_values::IntegrandValues{integrandType}
    integrand_cache::integrandCacheType
end

function (affect!::SavingIntegrandAffect)(integrator)
    n = div(SciMLBase.alg_order(integrator.alg) + 1, 2)
    integral = zeros(eltype(eltype(affect!.integrand_values.integrand)),
        length(integrator.p))
    for i in 1:n
        t_temp = ((integrator.t - integrator.tprev) / 2) * gauss_points[n][i] + 
                 (integrator.t + integrator.tprev) / 2
        if DiffEqBase.isinplace(integrator.sol.prob)
            curu = first(get_tmp_cache(integrator))
            integrator(curu, t_temp)
            affect!.integrand_func(affect!.integrand_cache, curu, t_temp, integrator)
            integral .+= gauss_weights[n][i] * affect!.integrand_cache
        else
            integral .+= gauss_weights[n][i] * 
                         affect!.integrand_func(integrator(t_temp), t_temp, integrator)
        end
    end
    integral *= -(integrator.t - integrator.tprev) / 2
    push!(affect!.integrand_values.integrand, integral)
    u_modified!(integrator, false)
end

"""
```julia
IntegratingCallback(integrand_func,
    integrand_values::IntegrandValues,
    gauss_points = Vector{eltype(integrand_values.t)}())
```

Lets one define a function `integrand_func(u, t, integrator)` which
returns Integral(integrand_func(u(t),t)dt over the problem tspan.

## Arguments

  - `integrand_func(out, u, t, integrator)` for in-place problems and `out = integrand_func(u, t, integrator)` for
    out-of-place problems. Returns the quantity in the integral for computing dG/dp.
    Note that for out-of-place problems, this should allocate the output (not as a view to `u`).
  - `integrand_values::IntegrandValues` is the types that `integrand_func` will return, i.e.
    `integrand_func(t, u, integrator)::integrandType`. It's specified via
    `IntegrandValues(integrandType)`, i.e. give the type
    that `integrand_func` will output (or higher compatible type).

The outputted values are saved into `integrand_values`. Time points are found via
`integrand_values.t` and the values are `integrand_values.integrand`.

!!! note

    This method is currently limited to ODE solvers of order 10 or lower. Open an issue if other
    solvers are required.
"""
function IntegratingCallback(integrand_func, integrand_values::IntegrandValues, cache)
    affect! = SavingIntegrandAffect(integrand_func, integrand_values, cache)
    condition = (u, t, integrator) -> true
    DiscreteCallback(condition, affect!, save_positions=(false,false))
end

export IntegratingCallback, IntegrandValues
