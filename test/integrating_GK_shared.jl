function compute_dGdp(integrand)
    temp = zeros(length(integrand.integrand), length(integrand.integrand[1]))
    for i in 1:length(integrand.integrand)
        for j in 1:length(integrand.integrand[1])
            temp[i, j] = integrand.integrand[i][j]
        end
    end
    return sum(temp, dims = 1)[:]
end

function compute_dGdp_nt(integrand)
    temp = zeros(length(integrand.integrand), 4)
    for i in 1:length(integrand.integrand)
        temp[i, 1:2] .= integrand.integrand[i].x.αβ
        temp[i, 3:4] .= integrand.integrand[i].δγ
    end
    return sum(temp, dims = 1)[:]
end

#### TESTING ON LINEAR SYSTEM WITH ANALYTICAL SOLUTION ####

function simple_linear_system(u, p, t)
    a, b = p
    return [-a * u[2], b * u[1]]
end

function adjoint_linear(u, p, t, sol)
    a, b = p
    return -[0 b; -a 0] * u - 2.0 * (sol(t) .- 1.0)
end

function adjoint_linear_inplace(du, u, p, t, sol)
    a, b = p
    return du .= -[0 b; -a 0] * u - 2.0 * (sol(t) .- 1.0)
end

function analytical_derivative(p, t)
    a, b = p
    d1 = (
        b * (
            (cos(2t * sqrt(a * b)) - 4cos(t * sqrt(a * b))) / (b * sqrt(a / b)) +
                (b * t * cos(2t * sqrt(a * b))) / sqrt(a * b) +
                (-4b * t * cos(t * sqrt(a * b))) / sqrt(a * b) +
                2(
                (2b * t * sin(t * sqrt(a * b))) / sqrt(a * b) +
                    (-b * t * sin(2t * sqrt(a * b))) / sqrt(a * b)
            ) * sqrt(a / b)
        ) +
            (b * t * (a + 3b)) / sqrt(a * b) +
            (-a * b * t * cos(2t * sqrt(a * b))) / sqrt(a * b) + 3 / sqrt(a / b) +
            2t * sqrt(a * b) - sin(2t * sqrt(a * b))
    ) / (4.0(a^0.5) * (b^1.5)) +
        (
        (3a * (b / (a^2))) / sqrt(b / a) +
            (a * (b / (a^2)) * cos(2t * sqrt(a * b))) / sqrt(b / a) +
            (b * t * (b + 3a)) / sqrt(a * b) +
            (b * t * (a - b) * cos(2t * sqrt(a * b))) / sqrt(a * b) +
            (-4a * (b / (a^2)) * cos(t * sqrt(a * b))) / sqrt(b / a) +
            (-4a * b * t * cos(t * sqrt(a * b))) / sqrt(a * b) +
            (2a * b * t * sqrt(b / a) * sin(2t * sqrt(a * b))) / sqrt(a * b) +
            (-4a * b * t * sqrt(b / a) * sin(t * sqrt(a * b))) / sqrt(a * b) +
            6t * sqrt(a * b) + 8sqrt(b / a) * cos(t * sqrt(a * b)) + sin(2t * sqrt(a * b)) -
            6sqrt(b / a) - 8sin(t * sqrt(a * b)) - 2sqrt(b / a) * cos(2t * sqrt(a * b))
    ) /
        (4.0(a^1.5) * (b^0.5)) +
        (
        -2.0(b^1.5) *
            (
            (
                b * (
                    2(cos(2t * sqrt(a * b)) - 4cos(t * sqrt(a * b))) * sqrt(a / b) +
                        sin(2t * sqrt(a * b)) - 8sin(t * sqrt(a * b))
                ) + 6b * sqrt(a / b) +
                    2t * (a + 3b) * sqrt(a * b) - a * sin(2t * sqrt(a * b))
            ) / (16.0a * (b^3.0))
        )
    ) /
        (a^0.5) -
        6.0(a^0.5) * (b^0.5) *
        (
        (
            (a - b) * sin(2t * sqrt(a * b)) + 8a * sqrt(b / a) * cos(t * sqrt(a * b)) +
                2t * (b + 3a) * sqrt(a * b) - 6a * sqrt(b / a) - 8a * sin(t * sqrt(a * b)) -
                2a * sqrt(b / a) * cos(2t * sqrt(a * b))
        ) / (16.0b * (a^3.0))
    )
    d2 = (
        b * (
            (a * t * cos(2t * sqrt(a * b))) / sqrt(a * b) +
                (-(a / (b^2)) * (cos(2t * sqrt(a * b)) - 4cos(t * sqrt(a * b)))) / sqrt(a / b) +
                (-4a * t * cos(t * sqrt(a * b))) / sqrt(a * b) +
                2(
                (2a * t * sin(t * sqrt(a * b))) / sqrt(a * b) +
                    (-a * t * sin(2t * sqrt(a * b))) / sqrt(a * b)
            ) * sqrt(a / b)
        ) +
            (-3b * (a / (b^2))) / sqrt(a / b) + (a * t * (a + 3b)) / sqrt(a * b) +
            (-t * (a^2) * cos(2t * sqrt(a * b))) / sqrt(a * b) + 6sqrt(a / b) +
            2(cos(2t * sqrt(a * b)) - 4cos(t * sqrt(a * b))) * sqrt(a / b) +
            6t * sqrt(a * b) + sin(2t * sqrt(a * b)) - 8sin(t * sqrt(a * b))
    ) /
        (4.0(a^0.5) * (b^1.5)) +
        (
        (4cos(t * sqrt(a * b))) / sqrt(b / a) + (-cos(2t * sqrt(a * b))) / sqrt(b / a) +
            (a * t * (b + 3a)) / sqrt(a * b) +
            (-4t * (a^2) * cos(t * sqrt(a * b))) / sqrt(a * b) +
            (a * t * (a - b) * cos(2t * sqrt(a * b))) / sqrt(a * b) +
            (-4t * (a^2) * sqrt(b / a) * sin(t * sqrt(a * b))) / sqrt(a * b) +
            (2t * (a^2) * sqrt(b / a) * sin(2t * sqrt(a * b))) / sqrt(a * b) +
            -3 / sqrt(b / a) + 2t * sqrt(a * b) - sin(2t * sqrt(a * b))
    ) /
        (4.0(a^1.5) * (b^0.5)) +
        (
        -2.0(a^1.5) *
            (
            (
                (a - b) * sin(2t * sqrt(a * b)) + 8a * sqrt(b / a) * cos(t * sqrt(a * b)) +
                    2t * (b + 3a) * sqrt(a * b) - 6a * sqrt(b / a) - 8a * sin(t * sqrt(a * b)) -
                    2a * sqrt(b / a) * cos(2t * sqrt(a * b))
            ) / (16.0b * (a^3.0))
        )
    ) / (b^0.5) -
        6.0(a^0.5) * (b^0.5) *
        (
        (
            b * (
                2(cos(2t * sqrt(a * b)) - 4cos(t * sqrt(a * b))) * sqrt(a / b) +
                    sin(2t * sqrt(a * b)) - 8sin(t * sqrt(a * b))
            ) + 6b * sqrt(a / b) +
                2t * (a + 3b) * sqrt(a * b) - a * sin(2t * sqrt(a * b))
        ) / (16.0a * (b^3.0))
    )
    return [d1, d2]
end

function callback_saving_linear(u, t, integrator, sol)
    return -1 .* [-sol(t)[2] 0; 0 sol(t)[1]]' * u
end
function callback_saving_linear_inplace(du, u, t, integrator, sol)
    return du .= -1 .* [-sol(t)[2] 0; 0 sol(t)[1]]' * u
end
