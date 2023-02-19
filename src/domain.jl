# type definitions

abstract type AbstractDomainAffect{T, S, uType} end

struct PositiveDomainAffect{T, S, uType} <: AbstractDomainAffect{T, S, uType}
    abstol::T
    scalefactor::S
    u::uType
end

struct GeneralDomainAffect{autonomous, F, T, S, uType} <: AbstractDomainAffect{T, S, uType}
    g::F
    abstol::T
    scalefactor::S
    u::uType
    resid::uType

    function GeneralDomainAffect{autonomous}(g::F, abstol::T, scalefactor::S, u::uType,
                                             resid::uType) where {autonomous, F, T, S, uType
                                                                  }
        new{autonomous, F, T, S, uType}(g, abstol, scalefactor, u, resid)
    end
end

# definitions of callback functions

# Workaround since it is not possible to add methods to an abstract type:
# https://github.com/JuliaLang/julia/issues/14919
(f::PositiveDomainAffect)(integrator) = affect!(integrator, f)
(f::GeneralDomainAffect)(integrator) = affect!(integrator, f)

# general method defintions for domain callbacks

"""
    affect!(integrator, f::AbstractDomainAffect)

Apply domain callback `f` to `integrator`.
"""
function affect!(integrator, f::AbstractDomainAffect{T, S, uType}) where {T, S, uType}
    if !SciMLBase.isadaptive(integrator)
        throw(ArgumentError("domain callback can only be applied to adaptive algorithms"))
    end

    # define array of next time step, absolute tolerance, and scale factor
    if uType <: Nothing
        if typeof(integrator.u) <: Union{Number, StaticArraysCore.SArray}
            u = integrator.u
        else
            u = similar(integrator.u)
        end
    else
        u = f.u
    end
    abstol = T <: Nothing ? integrator.opts.abstol : f.abstol
    scalefactor = S <: Nothing ? 1 // 2 : f.scalefactor

    # setup callback and save addtional arguments for checking next time step
    args = setup(f, integrator)

    # obtain proposed next time step
    dt = get_proposed_dt(integrator)

    # ensure that t + dt <= first(tstops)
    tdir = integrator.tdir
    if SciMLBase.has_tstop(integrator)
        tdir_t = tdir * integrator.t
        tdir_tstop = SciMLBase.first_tstop(integrator)
        dt = tdir * min(abs(dt), abs(tdir_tstop - tdir_t)) # step! to the end
    end
    t = integrator.t + dt

    dt_modified = false
    p = integrator.p
    while tdir * dt > 0
        # calculate estimated value of next step and its residuals
        if typeof(u) <: Union{Number, StaticArraysCore.SArray}
            u = integrator(t)
        else
            integrator(u, t)
        end

        # check whether time step is accepted
        isaccepted(u, p, t, abstol, f, args...) && break

        # reduce time step
        dtcache = dt
        dt *= scalefactor
        dt_modified = true
        t = integrator.t + dt

        # abort iteration when time step cannot be reduced any further
        # TODO: ideally, we would go back and shorten the previous time step instead
        # of displaying this warning
        if dtcache == dt
            if integrator.opts.verbose
                @warn("Could not restrict values to domain. Iteration was canceled since ",
                      "proposed time step dt = ", dt, " could not be reduced.")
            end
            break
        end
    end

    # update current and next time step
    if dt_modified # add safety factor since guess is based on extrapolation
        set_proposed_dt!(integrator, 9 // 10 * dt)
    end

    # modify u
    u_modified!(integrator, modify_u!(integrator, f))

    return
end

"""
    modify_u!(integrator, f::AbstractDomainAffect)

Modify current state vector `u` of `integrator` if required, and return whether it actually
was modified.
"""
modify_u!(integrator, ::AbstractDomainAffect) = false

"""
    setup(f::AbstractDomainAffect, integrator)

Setup callback `f` and return an arbitrary tuple whose elements are used as additional
arguments in checking whether time step is accepted.
"""
setup(::AbstractDomainAffect, integrator) = ()

"""
    isaccepted(u, abstol, f::AbstractDomainAffect, args...)

Return whether `u` is an acceptable state vector at the next time point given absolute
tolerance `abstol`, callback `f`, and other optional arguments.
"""
isaccepted(u, p, t, tolerance, ::AbstractDomainAffect, args...) = true

# specific method definitions for positive domain callback

function modify_u!(integrator, f::PositiveDomainAffect)
    # set all negative values to zero
    _set_neg_zero!(integrator, integrator.u) # Returns true if modified
end

function _set_neg_zero!(integrator, u::AbstractArray)
    modified = false
    @inbounds for i in eachindex(integrator.u)
        if integrator.u[i] < 0
            integrator.u[i] = 0
            modified = true
        end
    end
    modified
end

function _set_neg_zero!(integrator, u::Number)
    modified = false
    if integrator.u < 0
        integrator.u = 0
        modified = true
    end
    modified
end

function _set_neg_zero!(integrator, u::StaticArraysCore.SArray)
    modified = false
    @inbounds for i in eachindex(integrator.u)
        if u[i] < 0
            u = setindex(u, zero(first(u)), i)
            modified = true
        end
    end
    modified && (integrator.u = u)
    modified
end

# state vector is accepted if its entries are greater than -abstol
isaccepted(u, p, t, abstol::Number, ::PositiveDomainAffect) = all(ui -> ui > -abstol, u)
function isaccepted(u, p, t, abstol, ::PositiveDomainAffect)
    length(u) == length(abstol) ||
        throw(DimensionMismatch("numbers of states and tolerances do not match"))
    all(ui > -tol for (ui, tol) in zip(u, abstol))
end

# specific method definitions for general domain callback

# create array of residuals
function setup(f::GeneralDomainAffect, integrator)
    typeof(f.resid) <: Nothing ? (similar(integrator.u),) : (f.resid,)
end

function isaccepted(u, p, t, abstol, f::GeneralDomainAffect{autonomous, F, T, S, uType},
                    resid) where {autonomous, F, T, S, uType}
    # calculate residuals
    if autonomous
        f.g(resid, u, p)
    else
        f.g(resid, u, p, t)
    end

    # accept time step if residuals are smaller than the tolerance
    if typeof(abstol) <: Number
        all(x -> x < abstol, resid)
    else
        # element-wise comparison
        length(resid) == length(abstol) ||
            throw(DimensionMismatch("numbers of residuals and tolerances do not match"))
        all(x < y for (x, y) in zip(resid, abstol))
    end
end

# callback definitions

"""
```julia
GeneralDomain(g, u = nothing; nlsolve = NLSOLVEJL_SETUP(), save = true,
              abstol = nothing, scalefactor = nothing,
              autonomous = maximum(SciMLBase.numargs(g)) == 3,
              nlopts = Dict(:ftol => 10 * eps()))
```

A `GeneralDomain` callback in DiffEqCallbacks.jl generalizes the concept of
a `PositiveDomain` callback to arbitrary domains. Domains are specified by
in-place functions `g(resid, u, p)` or `g(resid, u, p, t)` that calculate residuals of a
state vector `u` at time `t` relative to that domain, with `p` the parameters of the
corresponding integrator. As for `PositiveDomain`, steps are accepted if residuals
of the extrapolated values at the next time step are below
a certain tolerance. Moreover, this callback is automatically coupled with a
`ManifoldProjection` that keeps all calculated state vectors close to the desired
domain, but in contrast to a `PositiveDomain` callback the nonlinear solver in a
`ManifoldProjection` cannot guarantee that all state vectors of the solution are
actually inside the domain. Thus, a `PositiveDomain` callback should generally be
preferred.

## Arguments

  - `g`: the implicit definition of the domain as a function `g(resid, u, p)` or
    `g(resid, u, p, t)` which is zero when the value is in the domain.
  - `u`: A prototype of the state vector of the integrator. A copy of it is saved and
    extrapolated values are written to it. If it is not specified,
    every application of the callback allocates a new copy of the state vector.

## Keyword Arguments

  - `nlsolve`: A nonlinear solver as defined [in the nlsolve format](@ref linear_nonlinear)
    which is passed to a `ManifoldProjection`.
  - `save`: Whether to do the standard saving (applied after the callback).
  - `abstol`: Tolerance up to, which residuals are accepted. Element-wise tolerances
    are allowed. If it is not specified, every application of the callback uses the
    current absolute tolerances of the integrator.
  - `scalefactor`: Factor by which an unaccepted time step is reduced. If it is not
    specified, time steps are halved.
  - `autonomous`: Whether `g` is an autonomous function of the form `g(resid, u, p)`.
  - `nlopts`: Optional arguments to nonlinear solver of a `ManifoldProjection` which
    can be any of the [NLsolve keywords](https://github.com/JuliaNLSolvers/NLsolve.jl#fine-tunings).
    The default value of `ftol = 10*eps()` ensures that convergence is only declared
    if the infinite norm of residuals is very small and hence the state vector is very
    close to the domain.

## References

Shampine, Lawrence F., Skip Thompson, Jacek Kierzenka and G. D. Byrne.
Non-negative solutions of ODEs. Applied Mathematics and Computation 170
(2005): 556-569.
"""
function GeneralDomain(g, u = nothing; nlsolve = NLSOLVEJL_SETUP(), save = true,
                       abstol = nothing, scalefactor = nothing,
                       autonomous = maximum(SciMLBase.numargs(g)) == 3,
                       nlopts = Dict(:ftol => 10 * eps()))
    if typeof(u) <: Nothing
        affect! = GeneralDomainAffect{autonomous}(g, abstol, scalefactor, nothing, nothing)
    else
        affect! = GeneralDomainAffect{autonomous}(g, abstol, scalefactor, deepcopy(u),
                                                  deepcopy(u))
    end
    condition = (u, t, integrator) -> true
    CallbackSet(ManifoldProjection(g; nlsolve = nlsolve, save = false,
                                   autonomous = autonomous, nlopts = nlopts),
                DiscreteCallback(condition, affect!; save_positions = (false, save)))
end

@doc doc"""
```julia
PositiveDomain(u = nothing; save = true, abstol = nothing, scalefactor = nothing)
```

Especially in biology and other natural sciences, a desired property of
dynamical systems is the positive invariance of the positive cone, i.e.
non-negativity of variables at time ``t_0`` ensures their non-negativity at times
``t \geq t_0`` for which the solution is defined. However, even if a system
satisfies this property mathematically it can be difficult for ODE solvers to
ensure it numerically, as these [MATLAB examples](https://www.mathworks.com/help/matlab/math/nonnegative-ode-solution.html)
show.

To deal with this problem, one can specify `isoutofdomain=(u,p,t) -> any(x
-> x < 0, u)` as additional [solver option](@ref solver_options),
which will reject any step that leads to non-negative values and reduce the next
time step. However, since this approach only rejects steps and hence
calculations might be repeated multiple times until a step is accepted, it can
be computationally expensive.

Another approach is taken by a `PositiveDomain` callback in
DiffEqCallbacks.jl, which is inspired by
[Shampine's et al. paper about non-negative ODE solutions](https://www.sciencedirect.com/science/article/pii/S0096300304009683).
It reduces the next step by a certain scale factor until the extrapolated value
at the next time point is non-negative with a certain tolerance. Extrapolations
are cheap to compute but might be inaccurate, so if a time step is changed it
is additionally reduced by a safety factor of 0.9. Since extrapolated values are
only non-negative up to a certain tolerance and in addition actual calculations
might lead to negative values, also any negative values at the current time point
are set to 0. Hence, by this callback non-negative values at any time point are
ensured in a computationally cheap way, but the quality of the solution
depends on how accurately extrapolations approximate next time steps.

Please note, that the system should be defined also outside the positive domain,
since even with these approaches, negative variables might occur during the
calculations. Moreover, one should follow Shampine's et al. advice and set the
derivative ``x'_i`` of a negative component ``x_i`` to ``\max \{0, f_i(x, t)\}``,
where ``t`` denotes the current time point with state vector ``x`` and ``f_i``
is the ``i``-th component of function ``f`` in an ODE system ``x' = f(x, t)``.

## Arguments

- `u`: A prototype of the state vector of the integrator. A copy of it is saved and
  extrapolated values are written to it. If it is not specified,
  every application of the callback allocates a new copy of the state vector.

## Keyword Arguments

- `save`: Whether to do the standard saving (applied after the callback).
- `abstol`: Tolerance up to, which negative extrapolated values are accepted.
  Element-wise tolerances are allowed. If it is not specified, every application
  of the callback uses the current absolute tolerances of the integrator.
- `scalefactor`: Factor by which an unaccepted time step is reduced. If it is not
  specified, time steps are halved.

## References

Shampine, Lawrence F., Skip Thompson, Jacek Kierzenka and G. D. Byrne.
Non-negative solutions of ODEs. Applied Mathematics and Computation 170
(2005): 556-569.
"""
function PositiveDomain(u = nothing; save = true, abstol = nothing, scalefactor = nothing)
    if typeof(u) <: Nothing
        affect! = PositiveDomainAffect(abstol, scalefactor, nothing)
    else
        affect! = PositiveDomainAffect(abstol, scalefactor, deepcopy(u))
    end
    condition = (u, t, integrator) -> true
    DiscreteCallback(condition, affect!; save_positions = (false, save))
end

export GeneralDomain, PositiveDomain
