module TuringABC

using Random, LinearAlgebra, Statistics


using AbstractMCMC: AbstractMCMC
using DynamicPPL: DynamicPPL, AbstractPPL, OrderedDict
using MCMCChains: MCMCChains

using DocStringExtensions

export ABC

"""
    ABC <: AbstractMCMC.AbstractSampler

Approximate Bayesian Computation (ABC) sampler.

# Fields
$(FIELDS)
"""
struct ABC{F,V,T} <: AbstractMCMC.AbstractSampler
    "distance and statistic method expecting two arguments: `data_true` and `data_proposed`"
    dist_and_stat::F
    "variable representing the data"
    data_variable::V
    "threshold used for comparison to decide whether to accept or reject"
    threshold::T
end

# Use the mean.
ABC(data_var, threshold) = ABC((data1, data2) -> mean(abs2, data1 .- data2), data_var, threshold)

"""
    split_latent_data(d::OrderedDict, data_variable)

Return a tuple with first element being variables and second being data.
"""
function split_latent_data(d::OrderedDict, data_variable::DynamicPPL.VarName)
    ks = collect(keys(d))
    data_keys = filter(ks) do k
        AbstractPPL.subsumes(data_variable, k)
    end
    θ = map(Base.Fix1(getindex, d), filter(∉(data_keys), ks))
    data = map(Base.Fix1(getindex, d), data_keys)

    return θ, data
end

"""
    split_latent_data(sampler::ABC, model::DynamicPPL.Model, x)

Return a tuple with first element being variables and second being data.
"""
split_latent_data(sampler::ABC, model::DynamicPPL.Model, x::OrderedDict) = split_latent_data(x, sampler.data_variable)

"""
    statistic_distance(sampler::ABC, data_true, data_candidate)

Return the distance between the statistics of `data_true` and `data_candidate`.
"""
statistic_distance(sampler::ABC, data_true, data_candidate) = sampler.dist_and_stat(data_true, data_candidate)

"""
    statistic_distance(sampler::ABC, model::DynamicPPL.Model, data_candidate)

Return the distance between observations in `model` and `data_candidate`.
"""
function statistic_distance(sampler::ABC, model::DynamicPPL.Model, data_candidate)
    return statistic_distance(sampler, observations(sampler, model), data_candidate)
end

"""
    make_joint_model(sampler::ABC, model::DynamicPPL.Model)

Return a model with observations now also considered random variables.
"""
make_joint_model(sampler::ABC, model::DynamicPPL.Model) = DynamicPPL.decondition(model, sampler.data_variable)

"""
    sample_from_joint(rng::Random.AbstractRNG, sampler::ABC, model::DynamicPPL.Model)

Sample from the joint model.

Defaults to `rand(rng, OrderedDict, make_joint_model(sampler, model))`.

See also: [`make_joint_model`](@ref).
"""
function sample_from_joint(rng::Random.AbstractRNG, sampler::ABC, model::DynamicPPL.Model)
    return rand(rng, OrderedDict, make_joint_model(sampler, model))
end

"""
    observations(sampler::ABC, model::DynamicPPL.Model)

Return the observations in `model`.
"""
function observations(sampler::ABC, model::DynamicPPL.Model)
    return DynamicPPL.getvalue(DynamicPPL.conditioned(model), sampler.data_variable)
end

# Implementation of the `AbstractMCMC` interface.
function AbstractMCMC.step(rng::Random.AbstractRNG, model::DynamicPPL.Model, sampler::ABC; kwargs...)
    # Sample from the joint model.
    d = sample_from_joint(rng, sampler, model)
    # Figure out which variables represents data.
    θ, _ = split_latent_data(d, sampler.data_variable)

    # TODO: Add statistics, etc.?
    return θ, θ
end

function AbstractMCMC.step(rng::Random.AbstractRNG, model::DynamicPPL.Model, sampler::ABC, θ_current; kwargs...)    
    # Sample from the joint model.
    d = sample_from_joint(rng, sampler, model)
    # Figure out which variables represents the data.
    θ_candidate, data_candidate = split_latent_data(sampler, model, d)
    # Compute the distance between the generated data and true.
    dist = statistic_distance(sampler, model, data_candidate)

    # TODO: Should `threshold` be adaptable?
    θ_next = dist < sampler.threshold ? θ_candidate : θ_current

    # TODO: Add statistics, etc.?
    return θ_next, θ_next
end

# Bundle the samples up nicely after calls to `sample`.
function AbstractMCMC.bundle_samples(
    samples::AbstractVector{<:AbstractVector{<:Real}},
    model::DynamicPPL.Model,
    sampler::ABC,
    ::Any,
    ::Type{MCMCChains.Chains};
    param_names=missing, discard_initial=0, thinning=1,
    kwargs...
)
    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(samples[1]))]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end

    return MCMCChains.Chains(samples, param_names, (parameters = param_names,); start=discard_initial + 1, thin=thinning)
end

end
