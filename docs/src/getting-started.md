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
samples_nuts = AbstractMCMC.sample(model, NUTS(), 1_000)
```

Now we do `ABC`:

```@example demo
using TuringABC

spl = ABC(0.1)
samples = AbstractMCMC.sample(model, spl, 10_000; chain_type=MCMCChains.Chains)
```
