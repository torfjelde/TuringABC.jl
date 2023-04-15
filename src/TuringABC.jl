module TuringABC

using Random, LinearAlgebra, Statistics

using AbstractMCMC: AbstractMCMC
using DynamicPPL: DynamicPPL, AbstractPPL, OrderedDict
using MCMCChains: MCMCChains

using Distributions
using DocStringExtensions

export ABC

_default_dist_and_stat(data1, data2) = mean(abs2, flatten(data1) .- flatten(data2))

"""
    ABC <: AbstractMCMC.AbstractSampler

Approximate Bayesian Computation (ABC) sampler.

# Fields
$(FIELDS)
"""
Base.@kwdef struct ABC{F,T} <: AbstractMCMC.AbstractSampler
    "distance and statistic method expecting two arguments: `data_true` and `data_proposed`"
    dist_and_stat::F=_default_dist_and_stat
    "initial threshold used for comparison to decide whether to accept or reject"
    threshold_initial::T=10.0
    "final threshold used for comparison to decide whether to accept or reject"
    threshold_minimum::T=1e-2
end

# Use the mean.
ABC(threshold_initial::Real) = ABC(; threshold_initial=float(threshold_initial))

struct ABCState{A,T}
    "current parameter values"
    θ::A
    "current threshold"
    threshold::T
    "current iteration"
    iteration::Int
    "threshold history"
    threshold_history::Vector{T}
end

function adapt!!(sampler::ABC, model::DynamicPPL.Model, state::ABCState)
    # TODO: Do something more principled than this scheduling.
    new_threshold = max(state.threshold * (state.iteration / (1 + state.iteration))^(3/4), sampler.threshold_minimum)
    push!(state.threshold_history, new_threshold)
    return ABCState(state.θ, new_threshold, state.iteration, state.threshold_history)
end

"""
    varname_keys(x)

Like `keys(x)`, but returns a collection of `VarName` instead of `Symbol`.
"""
varname_keys(d::AbstractDict) = keys(d)
varname_keys(d::NamedTuple) = map(k -> DynamicPPL.VarName{k}(), keys(d))

"""
    flatten(x)

Return a flattened version of `x`.
"""
flatten(x) = map(vcat, flatten, x)
flatten(x::Real) = [x]
flatten(x::AbstractArray{<:Real}) = vec(x)
flatten(x::AbstractDict) = mapreduce(flatten, vcat, values(x))
flatten(x::NamedTuple) = mapreduce(flatten, vcat, values(x))

"""
    split_latent_data(d::OrderedDict, data_variables, data)

Return a 3-tuple with first element being variables, second being sampled data,
and third being the original data.

The original data returned should be in the same format as the data sampled.
"""
function split_latent_data(x::OrderedDict, data_variables, data)
    ks = collect(keys(x))
    data_keys = filter(ks) do k
        any(Base.Fix1(AbstractPPL.subsumes, k), data_variables)
    end
    latent_keys = filter(∉(data_keys), ks)
    θ = OrderedDict(zip(latent_keys, map(Base.Fix1(getindex, x), latent_keys)))

    # Need to also make sure `data` is in the same format as `data_sampled`.
    # TODO: We're iterating over both twice. We
    data_vns = map(data_keys) do vn_parent
        data_for_vn = DynamicPPL.getvalue(x, vn_parent)
        return DynamicPPL.varname_leaves(vn_parent, data_for_vn)
    end
    x_data_vals = map(vns -> map(Base.Fix1(DynamicPPL.getvalue, x), vns), data_vns)
    data_vals = map(vns -> map(Base.Fix1(DynamicPPL.getvalue, data), vns), data_vns)

    return θ, OrderedDict(zip(data_keys, x_data_vals)), OrderedDict(zip(data_keys, data_vals))
end

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
make_joint_model(sampler::ABC, model::DynamicPPL.Model) = DynamicPPL.decondition(model)

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
observations(sampler::ABC, model::DynamicPPL.Model) = DynamicPPL.conditioned(model)

function sample_latent_and_data(rng::Random.AbstractRNG, sampler::ABC, model::DynamicPPL.Model)
    # Extract the data (usually a `NamedTuple` or `OrderedDict`).
    data = observations(sampler, model)
    # Extract the varnames.
    data_vns = varname_keys(data)
    # Sample from the joint model.
    x = sample_from_joint(rng, sampler, model)
    # Split the sampled variables into latent and data.
    θ, data_sampled, data_true = split_latent_data(x, data_vns, data)

    return θ, data_sampled, data_true
end


# Implementation of the `AbstractMCMC` interface.
function AbstractMCMC.step(rng::Random.AbstractRNG, model::DynamicPPL.Model, sampler::ABC; kwargs...)
    θ, _, _ = sample_latent_and_data(rng, sampler, model)
    # TODO: Add statistics, etc.?
    return θ, ABCState(θ, sampler.threshold_initial, 1, [sampler.threshold_initial])
end

function AbstractMCMC.step(rng::Random.AbstractRNG, model::DynamicPPL.Model, sampler::ABC, state::ABCState; kwargs...)
    # Adapt the threshold.
    state = adapt!!(sampler, model, state)

    # Sample a new candidate.
    θ_candidate, data_candidate, data_true = sample_latent_and_data(rng, sampler, model)
    # Compute the distance between the generated data and true.
    dist = statistic_distance(sampler, data_true, data_candidate)

    # Accept or reject the candidate.
    θ_next = dist < state.threshold ? θ_candidate : state.θ

    # TODO: Add statistics, etc.?
    return θ_next, ABCState(θ_next, state.threshold, state.iteration + 1, state.threshold_history)
end

# Bundle the samples up nicely after calls to `sample`.
function AbstractMCMC.bundle_samples(
    samples::AbstractVector{<:AbstractDict},
    model::DynamicPPL.Model,
    sampler::ABC,
    state::ABCState,
    ::Type{MCMCChains.Chains};
    discard_initial=0, thinning=1,
    kwargs...
)
    # Determine the keys present.
    # NOTE: We can potentially have variables with stochastic lengths,
    # variables that only included in some of the samples, etc., hence
    # we need to first do a full pass through the samples to determine
    # the union of the keys present in the samples.
    has_missings = false
    # Use a dictionary and track the keys corresponding to each symbol
    # separately so we can later easily order them in a reasonable manner.
    # E.g. if
    # - All samples have key `@varname(y)`
    # - Some samples have key `@varname(x[1])`
    # - Some samples have key `@varname(x[2])`
    # then, if we didn't group the varnames by the symbol, we could end up
    # with an ordering looking like `[x[1], y, x[2]]`, which is not ideal.
    # Instead, we group the varnames by the symbol, i.e. `x` and `y`, and
    # then concatenate the varnames at the end leading to, in this case,
    # `[x[1], x[2], y]`.
    param_names = OrderedDict()
    for sample in samples
        vns = keys(sample)
        for vn in vns
            sym = AbstractPPL.getsym(vn)
            sym_vns = collect(DynamicPPL.varname_leaves(vn, DynamicPPL.getvalue(sample, vn)))
            if !haskey(param_names, sym)
                param_names[sym] = sym_vns
            else
                # Check that the keys are the same.
                if param_names[sym] != sym_vns
                    has_missings = true
                    # Take the union of the keys.
                    param_names[sym] = union(param_names[sym], sym_vns)
                end
            end
        end
    end

    param_names = reduce(vcat, values(param_names))

    @debug "Found $(length(param_names)) parameters" param_names

    # Pre-allocation the array to hold the samples.
    num_iters = length(samples)
    num_params = length(param_names) + 1  # +1 for the thresholds.
    num_chains = 1
    T = Union{Missing, Float64}
    A = Array{T,3}(undef, num_iters, num_params, num_chains)

    # Extract the values.
    for (iter_idx, sample) in enumerate(samples)
        for (param_idx, vn) in enumerate(param_names)
            if DynamicPPL.hasvalue(sample, vn)
                A[iter_idx, param_idx, 1] = DynamicPPL.getvalue(sample, vn)
            else
                A[iter_idx, param_idx, 1] = missing
            end
        end

        # Add the threshold.
        A[iter_idx, num_params, 1] = state.threshold_history[discard_initial + iter_idx]
    end

    # HACK: `map(identity, A)` to potentially narrow the type of `A`.
    return MCMCChains.Chains(
        map(identity, A),
        vcat(param_names, [:threshold]),
        (parameters=param_names, internals=[:threshold]);
        start=discard_initial + 1,
        thin=thinning
    )
end

export DiracDelta
include("dirac.jl")

end
