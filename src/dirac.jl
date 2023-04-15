struct DiracDelta{T,V} <: Distributions.DiscreteDistribution{V}
    value::T
end

variateform(x::Real) = Distributions.Univariate
variateform(x::AbstractArray{<:Real,N}) where {N} = Distributions.ArrayLikeVariate{N}

DiracDelta(x) = DiracDelta{typeof(x),variateform(x)}(x)

const UnivariateDiracDelta{T} = DiracDelta{T,Distributions.Univariate}
const ArrayVariateDiracDelta{T,N} = DiracDelta{T,Distributions.ArrayLikeVariate{N}}

Base.eltype(::Type{DiracDelta{T}}) where {T} = T

Distributions.insupport(d::UnivariateDiracDelta, x::Real) = x == d.value
Distributions.insupport(d::DiracDelta, x::AbstractArray) = x == d.value

Base.minimum(d::DiracDelta) = d.value
Base.maximum(d::DiracDelta) = d.value
Distributions.support(d::DiracDelta) = (d.value,)

#### Properties
Distributions.mean(d::DiracDelta) = d.value
Distributions.var(d::DiracDelta{T}) where {T} = zero(T)

Distributions.mode(d::DiracDelta) = d.value

Distributions.entropy(d::DiracDelta{T}) where {T} = zero(T)

#### Evaluation

function Distributions.pdf(d::ArrayVariateDiracDelta{<:Any,N}, x::AbstractArray{<:Real,N}) where {N}
    T = eltype(d.value)
    return insupport(d, x) ? one(T) : zero(T)
end
function Distributions.logpdf(d::ArrayVariateDiracDelta{<:Any,N}, x::AbstractArray{<:Real,N}) where {N}
    T = eltype(d.value)
    return insupport(d, x) ? zero(T) : -T(Inf)
end

Distributions.pdf(d::UnivariateDiracDelta, x::Real) = insupport(d, x) ? one(d.value) : zero(d.value)
Distributions.logpdf(d::UnivariateDiracDelta, x::Real) = insupport(d, x) ? one(d.value) : -typeof(one(d.value))(Inf)

Distributions.cdf(d::UnivariateDiracDelta, x::Real) = x < d.value ? 0.0 : isnan(x) ? NaN : 1.0
Distributions.logcdf(d::UnivariateDiracDelta, x::Real) = x < d.value ? -Inf : isnan(x) ? NaN : 0.0
Distributions.ccdf(d::UnivariateDiracDelta, x::Real) = x < d.value ? 1.0 : isnan(x) ? NaN : 0.0
Distributions.logccdf(d::UnivariateDiracDelta, x::Real) = x < d.value ? 0.0 : isnan(x) ? NaN : -Inf

Distributions.quantile(d::UnivariateDiracDelta{T}, p::Real) where {T} = 0 <= p <= 1 ? d.value : T(NaN)

Distributions.mgf(d::UnivariateDiracDelta, t) = exp(t * d.value)
Distributions.cgf(d::UnivariateDiracDelta, t) = t*d.value
Distributions.cf(d::UnivariateDiracDelta, t) = cis(t * d.value)

#### Sampling
Base.rand(rng::Random.AbstractRNG, d::UnivariateDiracDelta) = d.value
Base.rand(rng::Random.AbstractRNG, d::ArrayVariateDiracDelta) = d.value
