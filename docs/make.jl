cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Documenter, Literate, Euler2D

Literate.markdown("./src/array_sims.jl", "./src")

pages = ["Introduction" => "index.md", "Tutorial" => "array_sims.md", "API" => "api.md"]

makedocs(; pages, modules = [Euler2D], remotes=nothing)