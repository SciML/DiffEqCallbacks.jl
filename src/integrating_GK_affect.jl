# Integrating affect with Gauss-Kronrod

using QuadGK
gk_points = [kronrod(i,-1,1)[1] for i in 1:10]
gk_weights= [kronrod(i,-1,1)[2] for i in 1:10]
g_weights = [kronrod(i,-1,1)[3] for i in 1:10]


## Structures 

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
	gk_error_cache::IntegrandCacheType
end


# Calculates integral value over (bound_l,bound_r)
function integrate_gk!(affect!::SavingIntegrandGKAffect, bound_l, bound_r; order=3, tol=1e-10)
	affect!.gk_step_cache = sum(gk_weights[order].*affect!.integrand_func( (gk_points[order].+1).*((bound_r-bound_l)/2) .+ bound_l)) * (bound_r-bound_l)/2
	error = abs( affect!.gk_error_cache - sum(g_weights[order].*affect!.integrand_func((gk_points[order][2:2:end].+1).*((bound_r-bound_l)/2).+bound_l)*(bound_r-bound_l)/2) )
	if error<tol
		affect!.accumulation_cache += affect!.gk_step_cache
	else
		integrate_gk!(affect!, bound_l, (bound_l+bound_r)/2)
		integrate_gk!(affect!, (bound_l+bound_r)/2, bound_r)
	end
end


function (affect!::SavingIntegrandGKAffect)(integrator)
	n = 3 # alg order
	accumulation_cache = 0 # Resets the cache
	integrate_gk!(affect!, integrator.t_prev, integrator.t) # Calculates integral values for (t_prev,t) into acc_cache
	push!(affect!.integrand_values.ts, integrator.t)        # publishes t_steps
	push!(affect!.integrand_values.integrand, recursive_copy(accumulation_cache)) # publishes integral cache
	u_modified!(integrator, false)						      # ???
end


## Exports and inclusion


function IntegratingGKCallback(
	integrand_func, integrand_values::IntegrandValues, integrand_prototype)
    affect! = SavingIntegrandGKAffect(integrand_func, integrand_values, integrand_prototype,
        allocate_zeros(integrand_prototype), allocate_zeros(integrand_prototype))
    condition = true_condition
    DiscreteCallback(condition, affect!, save_positions = (false, false))
end

export IntegratingGKCallback




