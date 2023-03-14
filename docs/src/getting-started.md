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

Let's sample with `NUTS` first to have something to compare to

```@example demo
samples_nuts = AbstractMCMC.sample(model, NUTS(), 1_000)
```

Now we do `ABC`:

```@example demo
using TuringABC

spl = ABC(Turing.@varname(y), 0.1)
samples = AbstractMCMC.sample(model, spl, 10_000; chain_type=MCMCChains.Chains)
```
