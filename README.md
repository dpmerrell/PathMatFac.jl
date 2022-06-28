# PathwayMultiomics.jl

A matrix factorization model for multi-omic data and biological pathways&mdash;implemented in Julia.

Note: this code is under development and subject to change!

## Installation

Install in the usual Julia way:

```
julia> using Pkg; Pkg.add(url="https://github.com/dpmerrell/PathwayMultiomics.jl")
```

## Main Ideas

Suppose you have a big array \\(A\\) of multi-omic data.
Each row is a sample, and each column is a particular omic measurement.

This package uses the matrix factorization model of [MatFac.jl](https://github.com/dpmerrell/MatFac.jl) to model \\(A = X^T Y\\).

* We use biological pathways to regularize the \\(Y \\) matrix.
  This allows us to interpret each row of \\(Y\\) as a _pathway factor_.
* We allow biological conditions to regularize the \\(X\\) matrix.
  I.e., two samples belonging to the same biological condition are expected to have
  similar attributes.
* The model accounts for distributional differences between omic assays:
    - It includes column-specific _shift_ and _scale_ parameters ( \\(\mu\\) and \\(\sigma\\) ).
    - It assigns appropriate noise models to assays. E.g., log-transformed mRNAseq data are normally-distributed
      and somatic mutations are bernoulli-distributed.
* The model accounts for **batch effects**. If you provide batch identifiers for 
  each sample and each assay, the model will fit batch-specific _shift_ and _scale_ parameters.


## Basic Usage

TODO: fill this out as development converges


