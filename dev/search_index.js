var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [TuringABC]","category":"page"},{"location":"api/#TuringABC.ABC","page":"API","title":"TuringABC.ABC","text":"ABC <: AbstractMCMC.AbstractSampler\n\nApproximate Bayesian Computation (ABC) sampler.\n\n\n\n\n\n","category":"type"},{"location":"api/#TuringABC.make_joint_model-Tuple{ABC, DynamicPPL.Model}","page":"API","title":"TuringABC.make_joint_model","text":"make_joint_model(sampler::ABC, model::DynamicPPL.Model)\n\nReturn a model with observations now also considered random variables.\n\n\n\n\n\n","category":"method"},{"location":"api/#TuringABC.observations-Tuple{ABC, DynamicPPL.Model}","page":"API","title":"TuringABC.observations","text":"observations(sampler::ABC, model::DynamicPPL.Model)\n\nReturn the observations in model.\n\n\n\n\n\n","category":"method"},{"location":"api/#TuringABC.sample_from_joint-Tuple{Random.AbstractRNG, ABC, DynamicPPL.Model}","page":"API","title":"TuringABC.sample_from_joint","text":"sample_from_joint(rng::Random.AbstractRNG, sampler::ABC, model::DynamicPPL.Model)\n\nSample from the joint model.\n\nDefaults to rand(rng, OrderedDict, make_joint_model(sampler, model)).\n\nSee also: make_joint_model.\n\n\n\n\n\n","category":"method"},{"location":"api/#TuringABC.split_latent_data-Tuple{ABC, DynamicPPL.Model, OrderedCollections.OrderedDict}","page":"API","title":"TuringABC.split_latent_data","text":"split_latent_data(sampler::ABC, model::DynamicPPL.Model, x)\n\nReturn a tuple with first element being variables and second being data.\n\n\n\n\n\n","category":"method"},{"location":"api/#TuringABC.split_latent_data-Tuple{OrderedCollections.OrderedDict, AbstractPPL.VarName}","page":"API","title":"TuringABC.split_latent_data","text":"split_latent_data(d::OrderedDict, data_variable)\n\nReturn a tuple with first element being variables and second being data.\n\n\n\n\n\n","category":"method"},{"location":"api/#TuringABC.statistic_distance-Tuple{ABC, Any, Any}","page":"API","title":"TuringABC.statistic_distance","text":"statistic_distance(sampler::ABC, data_true, data_candidate)\n\nReturn the distance between the statistics of data_true and data_candidate.\n\n\n\n\n\n","category":"method"},{"location":"api/#TuringABC.statistic_distance-Tuple{ABC, DynamicPPL.Model, Any}","page":"API","title":"TuringABC.statistic_distance","text":"statistic_distance(sampler::ABC, model::DynamicPPL.Model, data_candidate)\n\nReturn the distance between observations in model and data_candidate.\n\n\n\n\n\n","category":"method"},{"location":"getting-started/#Getting-started","page":"Getting started","title":"Getting started","text":"","category":"section"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"using Turing\n\n# Example model.\n@model function demo()\n    x ~ Normal()\n    y ~ Normal(x, 1)\nend\n\nmodel = demo() | (y = 2.0, )","category":"page"},{"location":"getting-started/","page":"Getting started","title":"Getting started","text":"using TuringABC\n\nspl = ABC(Turing.@varname(y), 0.1)\nsamples = AbstractMCMC.sample(model, spl, 10_000; chain_type=MCMCChains.Chains)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TuringABC","category":"page"},{"location":"#TuringABC","page":"Home","title":"TuringABC","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TuringABC is a simple implementation of Approximate Bayesian Inference (ABC) in a way compatible with Turing.jl-models.","category":"page"}]
}