## sample space/domain

abstract type VariateForm end
mutable struct Univariate    <: VariateForm end
mutable struct Multivariate  <: VariateForm end
mutable struct Matrixvariate <: VariateForm end

abstract type ValueSupport end
mutable struct Discrete   <: ValueSupport end
mutable struct Continuous <: ValueSupport end

Base.eltype(::Type{Discrete}) = Int
Base.eltype(::Type{Continuous}) = Float64

## Sampleable

abstract type Sampleable{F<:VariateForm,S<:ValueSupport} end

"""
    length(s::Sampleable)

The length of each sample. Always returns `1` when `s` is univariate.
"""
Base.length(s::Sampleable) = prod(size(s))
Base.length(::Sampleable{Univariate}) = 1
Base.length(s::Sampleable{Multivariate}) = throw(MethodError(length, (s,)))

"""
    size(s::Sampleable)

The size (i.e. shape) of each sample. Always returns `()` when `s` is univariate, and
`(length(s),)` when `s` is multivariate.
"""
Base.size(s::Sampleable)
Base.size(s::Sampleable{Univariate}) = ()
Base.size(s::Sampleable{Multivariate}) = (length(s),)

"""
    eltype(s::Sampleable)

The default element type of a sample. This is the type of elements of the samples generated
by the `rand` method. However, one can provide an array of different element types to
store the samples using `rand!`.
"""
Base.eltype(s::Sampleable{F,S}) where {F,S} = eltype(S)
Base.eltype(s::Sampleable{F,Discrete}) where {F} = Int
Base.eltype(s::Sampleable{F,Continuous}) where {F} = Float64

"""
    nsamples(s::Sampleable)

The number of samples contained in `A`. Multiple samples are often organized into an array,
depending on the variate form.
"""
nsamples(t::Type{Sampleable}, x::Any)
nsamples{D<:Sampleable{Univariate}}(::Type{D}, x::Number) = 1
nsamples{D<:Sampleable{Univariate}}(::Type{D}, x::AbstractArray) = length(x)
nsamples{D<:Sampleable{Multivariate}}(::Type{D}, x::AbstractVector) = 1
nsamples{D<:Sampleable{Multivariate}}(::Type{D}, x::AbstractMatrix) = size(x, 2)
nsamples{D<:Sampleable{Matrixvariate}}(::Type{D}, x::Number) = 1
nsamples{D<:Sampleable{Matrixvariate},T<:Number}(::Type{D}, x::Array{Matrix{T}}) = length(x)

## Distribution <: Sampleable

abstract type Distribution{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S} end

const UnivariateDistribution{S<:ValueSupport}   = Distribution{Univariate,S}
const MultivariateDistribution{S<:ValueSupport} = Distribution{Multivariate,S}
const MatrixDistribution{S<:ValueSupport}       = Distribution{Matrixvariate,S}
const NonMatrixDistribution = Union{UnivariateDistribution, MultivariateDistribution}

const DiscreteDistribution{F<:VariateForm}   = Distribution{F,Discrete}
const ContinuousDistribution{F<:VariateForm} = Distribution{F,Continuous}

const DiscreteUnivariateDistribution     = Distribution{Univariate,    Discrete}
const ContinuousUnivariateDistribution   = Distribution{Univariate,    Continuous}
const DiscreteMultivariateDistribution   = Distribution{Multivariate,  Discrete}
const ContinuousMultivariateDistribution = Distribution{Multivariate,  Continuous}
const DiscreteMatrixDistribution         = Distribution{Matrixvariate, Discrete}
const ContinuousMatrixDistribution       = Distribution{Matrixvariate, Continuous}

variate_form{VF<:VariateForm,VS<:ValueSupport}(::Type{Distribution{VF,VS}}) = VF
variate_form{T<:Distribution}(::Type{T}) = variate_form(supertype(T))

value_support{VF<:VariateForm,VS<:ValueSupport}(::Type{Distribution{VF,VS}}) = VS
value_support{T<:Distribution}(::Type{T}) = value_support(supertype(T))

## TODO: the following types need to be improved
abstract type SufficientStats end
abstract type IncompleteDistribution end

const DistributionType{D<:Distribution} = Type{D}
const IncompleteFormulation = Union{DistributionType,IncompleteDistribution}
