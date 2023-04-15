using TuringABC, DynamicPPL, Distributions, LinearAlgebra, Random, MCMCChains
using Test

@model function demo1()
    x ~ Normal(0, 1)
    y ~ Normal(x, 1)
end

@model function demo2()
    x ~ Normal(0, 1)
    y ~ MvNormal(x .* zeros(10), I)
end

@model function demo3()
    x ~ MvNormal(zeros(10), 1)
    y ~ MvNormal(x , I)
    z ~ MvNormal(x, 100 * I)
end

# Model with latent variable `x` with random length.
@model function demo4()
    p ~ Beta(1, 5)
    k ~ Geometric(p)
    x ~ MvNormal(zeros(k + 1), 1)
    z ~ Normal(sum(x), 1)
end

const EXAMPLE_MODELS = (
    condition(demo1(), y = 1),
    condition(demo2(), y = zeros(10)),
    condition(demo3(), y = zeros(10)),
    condition(demo3(), y = zeros(10), z = ones(10)),
    condition(demo4(), z = 10),
)

@testset "TuringABC.jl" begin
    @testset "$(model.f)" for model in EXAMPLE_MODELS
        # This is just checking that we sort of move around so we'll use
        # a huge epsilon.
        sampler = ABC(100)
        chain = sample(model, sampler, 100; chain_type=MCMCChains.Chains)
        # Every model contains some `x` variable, so we'll just check this.
        sym = first(names(MCMCChains.group(chain, :x)))
        @test std(chain[sym]) > 0
    end

    @testset "sample_latent_and_data" begin
        @model function demo()
            x ~ Normal(0, 1)
            y ~ Normal(x, 1)
        end
        model = demo() | (y = 1.0,)
        sampler = ABC(0.1)
        chain = sample(model, sampler, 10_000; chain_type=MCMCChains.Chains, progress=false)
        posterior_true = Normal(1/2, 1/√2)
        @test mean(chain[:x]) ≈ mean(posterior_true) atol=0.05
        @test std(chain[:x]) ≈ std(posterior_true) atol=0.05
    end
end
