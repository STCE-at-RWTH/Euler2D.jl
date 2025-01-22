### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 1b7ead20-2f14-11ef-28d1-03ad74bf304a
begin
	import Pkg
    Pkg.activate(Base.current_project())
    using Revise, Euler2D
	using ShockwaveProperties
	using LaTeXStrings
	using LinearAlgebra
	using Plots
	using PlutoUI
	using Printf
	using StaticArrays
	using Unitful
end

# ╔═╡ 1b1c0430-bacf-454a-a0bf-b00c7b723b48
esim = load_cell_sim("../data/circular_obstacle_primal.celltape")

# ╔═╡ 01380877-d1cb-4ac7-ad08-8c8c57d95dcc
begin
	count(≠(0), esim.cell_ids)
	print(esim.cell_ids)
end

# ╔═╡ 35061132-50bf-4dd8-8178-ddf25a78b2f6
function visualize_boundary_cells(esim) 
	data_field = sign.(esim.cell_ids)
	(xs, ys) = cell_centers(esim)
	return heatmap(xs, ys, data_field', aspect_ratio=:equal, dpi=600, size=(1000,1000), xlabel=L"x", ylabel=L"y", title="Boundary Cells", titlefontface="Computer Modern")
end

# ╔═╡ a123f9d8-e6fe-4beb-974c-5c97ac24a8ea
function circleShape(radius, center, start=-π, stop=0.0)
	θ = LinRange(start, stop, 500)
	center[1] .+ radius*sin.(θ), center[2] .+ radius*cos.(θ)
end

# ╔═╡ ab62cd5f-5724-4d82-86f3-832cb69936d6
begin
	RADIUS = 0.75
	CENTER = SVector(0.0, 0.0)
	p = visualize_boundary_cells(esim)
	plot!(p, circleShape(RADIUS, CENTER))
end

# ╔═╡ Cell order:
# ╠═1b7ead20-2f14-11ef-28d1-03ad74bf304a
# ╠═1b1c0430-bacf-454a-a0bf-b00c7b723b48
# ╠═01380877-d1cb-4ac7-ad08-8c8c57d95dcc
# ╠═35061132-50bf-4dd8-8178-ddf25a78b2f6
# ╠═a123f9d8-e6fe-4beb-974c-5c97ac24a8ea
# ╠═ab62cd5f-5724-4d82-86f3-832cb69936d6
