# Keep ODE solution in a domain specified by a function. Inspired by:
# Shampine, L.F., S. Thompson, J.A. Kierzenka, and G.D. Byrne, "Non-negative solutions
# of ODEs," Applied Mathematics and Computation Vol. 170, 2005, pp. 556-569.

# type definitions

abstract type AbstractDomainAffect{T,S,uType} end

struct PositiveDomainAffect{T,S,uType} <: AbstractDomainAffect{T,S,uType}
    abstol::T
    scalefactor::S
    u::uType
end

struct GeneralDomainAffect{F,T,S,uType} <: AbstractDomainAffect{T,S,uType}
    g::F
    abstol::T
    scalefactor::S
    u::uType
    resid::uType
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
function affect!(integrator, f::AbstractDomainAffect{T,S,uType}) where {T,S,uType}
    # modify u
    u_modified!(integrator, modify_u!(integrator, f))

    # define array of next time step, absolute tolerance, and scale factor
    u = typeof(f.u) <: Void ? similar(integrator.u) : f.u
    abstol = typeof(f.abstol) <: Void ? integrator.opts.abstol : f.abstol
    scalefactor = typeof(f.scalefactor) <: Void ? integrator.opts.qmin : f.scalefactor

    # setup callback and save addtional arguments for checking next time step
    args = setup(f, integrator)

    # cache current time step
    dt = integrator.dt

    # update time step of integrator to proposed next time step
    integrator.dt = get_proposed_dt(integrator)

    # adjust time step to bounds and time stops
    fix_dt_at_bounds!(integrator)
    modify_dt_for_tstops!(integrator)

    while integrator.tdir * integrator.dt > 0
        # calculate estimated value of next step and its residuals
        integrator(u, integrator.t + integrator.dt)

        # check whether time step is accepted
        isaccepted(u, abstol, f, args...) && break

        # adjust time step
        dtcache = integrator.dt
        integrator.dt *= scalefactor
        fix_dt_at_bounds!(integrator)
        modify_dt_for_tstops!(integrator)

        # abort iteration when time step is not changed
        if dtcache == integrator.dt
            if integrator.opts.verbose
                warn("Could not restrict values to domain. Iteration was canceled since ",
                     "time step dt = ", integrator.dt, " could not be reduced.")
            end
            break
        end
    end

    # update current and next time step
    set_proposed_dt!(integrator, integrator.dt)
    integrator.dt = dt
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
isaccepted(u, tolerance, ::AbstractDomainAffect, args...) = true

# specific method definitions for positive domain callback

function modify_u!(integrator, f::PositiveDomainAffect)
    modified = false

    # set all negative values to zero
    @inbounds for i in eachindex(integrator.u)
        if integrator.u[i] < 0
            integrator.u[i] = 0
            modified = true
        end
    end

    modified
end

# state vector is accepted if its entries are greater than -abstol
isaccepted(u, abstol::Number, ::PositiveDomainAffect) = all(x -> x + abstol > 0, u)
isaccepted(u, abstol, ::PositiveDomainAffect) = all(x + y > 0 for (x,y) in zip(u, abstol))

# specific method definitions for general domain callback

# create array of residuals
setup(f::GeneralDomainAffect, integrator) =
    typeof(f.resid) <: Void ? (similar(integrator.u),) : (f.resid,)

function isaccepted(u, abstol, f::GeneralDomainAffect, resid)
    # calculate residuals
    f.g(u, resid)

    # accept time step if residuals are smaller than the tolerance
    if typeof(abstol) <: Number
        all(x-> x < abstol, resid)
    else
        # element-wise comparison
        all(x < y for (x,y) in zip(resid, abstol))
    end
end

# callback definitions

function GeneralDomain(g, u=nothing; nlsolve=NLSOLVEJL_SETUP(), save=true,
                       abstol=nothing, scalefactor=nothing)
    if typeof(u) <: Void
        affect! = GeneralDomainAffect(g, abstol, scalefactor, nothing, nothing)
    else
        affect! = GeneralDomainAffect(g, abstol, scalefactor, deepcopy(u), deepcopy(u))
    end
    condition = (t,u,integrator) -> true
    CallbackSet(ManifoldProjection(g; nlsolve=nlsolve, save=false),
                DiscreteCallback(condition, affect!; save_positions=(false, save)))
end

function PositiveDomain(u=nothing; save=true, abstol=nothing, scalefactor=nothing)
    if typeof(u) <: Void
        affect! = PositiveDomainAffect(abstol, scalefactor, nothing)
    else
        affect! = PositiveDomainAffect(abstol, scalefactor, deepcopy(u))
    end
    condition = (t,u,integrator) -> true
    DiscreteCallback(condition, affect!; save_positions=(false, save))
end

export GeneralDomain, PositiveDomain
