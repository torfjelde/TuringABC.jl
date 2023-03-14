# Getting started

```@example demo
using Turing

# Example model.
@model function demo()
    x ~ Normal()
    y ~ Normal(x, 1)
end

model = demo() | (y = 2.0, )
```

```@example demo
using TuringABC

spl = ABC(Turing.@varname(y), 0.1)
samples = AbstractMCMC.sample(model, spl, 10_000; chain_type=MCMCChains.Chains)
```
