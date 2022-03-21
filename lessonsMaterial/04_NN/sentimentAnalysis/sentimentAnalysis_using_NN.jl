cd(@__DIR__)
using Pkg
Pkg.activate(".")
using Random
Random.seed!(123)

using DelimitedFiles, CSV, DataFrames, Flux

