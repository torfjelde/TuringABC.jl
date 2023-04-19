# Getting started

First we define a model with Turing.jl.

```@example demo
using Turing

# Example model.
@model function demo()
    x ~ Normal()
    y ~ Normal(x, 1)
end

model = demo() | (y = 2.0, )
```

!!! warning
    TuringABC currently only supports conditioning of the form
    `model | (...)` or `condition(model, ...)`. That is, passing
    conditioned variables as inputs to the model is NOT supported (yet).

Let's sample with `NUTS` first to have something to compare to

```@example demo
samples_nuts = sample(model, NUTS(), 1_000)
```

Now we do `ABC`:

```@example demo
using TuringABC

spl = ABC(0.1)
samples = sample(model, spl, 10_000; chain_type=MCMCChains.Chains)
```

## More complex example
Now we're going to try something a bit more crazy: we'll run inference _within_ a model `outer_model`, and _then_ run inference over this!
Yes, you read that right: _inference-within-inference_.

!!! warning
    This is not something we recommend doing; this is just a demo of what one _could_ do!

```@example nested-sampling
using Turing, TuringABC, LinearAlgebra, Logging
using Turing.DynamicPPL
```

First we define the `inner_model`, i.e. the model we're going to do inference over within `outer_model`.

```@example nested-sampling
@model function inner_model(σ², N)
    x ~ MvNormal(zeros(N), I)
    y_inner ~ MvNormal(x, σ² * I)
end
```

Then we need a method which can convert the resulting approximation of the posterior of `inner_model` into some statistics that we can use as "observation" for the approximate posterior:

```@example nested-sampling
function f_default(samples::MCMCChains.Chains)
    # Use quantiles of the "posterior" (approximated by `samples`)
    return vec(mapreduce(Base.Fix2(quantile, [0.25, 0.5, 0.75]), hcat, eachcol(Array(samples))))
end
```

Now we can finally define the `outer_model`!

```@example nested-sampling
@model function outer_model(
    μ;
    f=f_default,
    # Sampler and number of samples for the inner model.
    # We'll use NUTS by default, but this will be expensive!
    inner_sampler=NUTS(),
    num_inner_samples=1000,
)
    N = length(μ)
    # Prior on the variance used.
    σ² ~ InverseGamma(2, 1)
    # Prior on the mean used.
    y ~ MvNormal(μ, σ² * I)
    # Obtain (approximate) posterior of the inner model conditioned
    # on the sampled `y` from above.
    inner_mdl = inner_model(σ², N) | (y_inner = y,)
    # Turn off logging for this inner sample since it will be called many times.
    posterior = with_logger(NullLogger()) do
        sample(inner_mdl, inner_sampler, num_inner_samples; chain_type=MCMCChains.Chains, progress=false)
    end
    # Since we're now working with an empirical approximation of the
    # posterior, we project `posterior` (usually samples) onto some statistics
    # using `f`, and then this we'll fix/condition to some value later.
    stat ~ DiracDelta(f(posterior))

    return (; posterior, stat)
end
```

```@example nested-sampling
# Let's generate some data.
μ = zeros(2)
model = outer_model(μ)

vars_true = (σ² = 1.0, y = 0.5 .* ones(length(μ)))
stat_true = rand(condition(model, vars_true)).stat
```

```@example nested-sampling
# Now condition the model on the true statistic.
conditioned_model = model | (stat = stat_true,)
# Now if we sample from it there is no `stat`.
rand(conditioned_model)
```

```@example nested-sampling
# We can now use ABC to sample.
# NOTE: This will take a few minutes to run since we're running NUTS in every ABC iteration.
chain = sample(
    conditioned_model,
    ABC(0.1),
    1000;
    discard_initial=1000,
    chain_type=MCMCChains.Chains,
    progress=true
)
```

```@example nested-sampling
quantile(chain)
```

```@example nested-sampling
using StatsPlots
plot(chain)
```

This is clearly not working very well:) But hey, at least it's possible!
