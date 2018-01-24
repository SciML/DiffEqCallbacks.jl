mutable struct FunctionCallingAffect{funcFunc, funcatType, funcatCacheType}
    func::funcFunc
    funcat::funcatType
    funcat_cache::funcatCacheType
    func_everystep::Bool
    func_start::Bool
    funciter::Int
end

function (affect!::FunctionCallingAffect)(integrator,force_func = false)
    # see OrdinaryDiffEq.jl -> integrator_utils.jl, function funcvalues!
    while !isempty(affect!.funcat) && integrator.tdir*top(affect!.funcat) <= integrator.tdir*integrator.t # Perform funcat
        affect!.funciter += 1
        curt = pop!(affect!.funcat) # current time
        if curt != integrator.t # If <t, interpolate
            if typeof(integrator) <: ODEIntegrator
                # Expand lazy dense for interpolation
                ode_addsteps!(integrator)
            end
            if typeof(integrator.u) <: Union{Number,SArray}
                curu = integrator(curt)
            else
                curu = first(get_tmp_cache(integrator))
                integrator(curu,curt) # inplace since func_func allocates
            end
            affect!.func(curu, curt, integrator)
        else # ==t, just func
            affect!.func(integrator.u, integrator.t, integrator)
        end
    end
    if affect!.func_everystep || force_func
        affect!.funciter += 1
        affect!.func(integrator.u, integrator.t, integrator)
    end
    u_modified!(integrator, false)
end

function functioncalling_initialize(cb, u, t, integrator)
    if cb.affect!.funciter != 0
        if integrator.tdir > 0
            cb.affect!.funcat = binary_minheap(cb.affect!.funcat_cache)
        else
            cb.affect!.funcat = binary_maxheap(cb.affect!.funcat_cache)
        end
        cb.affect!.funciter = 0
    end
    cb.affect!.func_start && cb.affect!(integrator)
end


"""
    FunctionCallingCallback(func;
                    funcat=Vector{Float64}(),
                    func_everystep=isempty(funcat),
                    func_start = true,
                    tdir=1)

A `DiscreteCallback` applied after every step to call `func(u,t,integrator)`
If `func_everystep`, every step of the integrator gives a `func` call.
If `funcat` is specified, the function is called at the given times, using
interpolation if necessary.
If the time `tdir` direction is not positive, i.e. `tspan[1] > tspan[2]`,
`tdir = -1` has to be specified.
"""
function FunctionCallingCallback(func;
                        funcat=Vector{Float64}(),
                        func_everystep=isempty(funcat),
                        func_start = true,
                        tdir=1)
    # funcat conversions, see OrdinaryDiffEq.jl -> integrators/type.jl
    funcat_vec = collect(funcat)
    if tdir > 0
        funcat_internal = binary_minheap(funcat_vec)
    else
        funcat_internal = binary_maxheap(funcat_vec)
    end
    affect! = FunctionCallingAffect(func, funcat_internal,
                                    funcat_vec, func_everystep, func_start, 0)
    condtion = (u, t, integrator) -> true
    DiscreteCallback(condtion, affect!;
                     initialize = functioncalling_initialize,
                     save_positions=(false,false))
end


export FunctionCallingCallback
