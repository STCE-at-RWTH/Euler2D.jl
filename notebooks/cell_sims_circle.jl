### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

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
count(≠(0), esim.cell_ids)

# ╔═╡ 59540798-0328-4df1-816f-ec5121c8ab16
bcurve(y) = ShockwaveProperties.BilligShockParametrization.shock_front(y, 4.0, 0.75)

# ╔═╡ cf9e166b-a3e7-4082-9e06-fe67ab66f762
billig_data = mapreduce(vcat, bcurve.(cell_centers(esim)[2])) do v
	v'
end

# ╔═╡ 89f497a5-2e05-44b5-a947-f3fcc31094ff
cell_centers(esim)

# ╔═╡ 35061132-50bf-4dd8-8178-ddf25a78b2f6
function plot_scalar_field(field_fn, esim, frame, title) 
	(t, _) = nth_step(esim, frame)
	data_field = reshape(field_fn(esim, frame), grid_size(esim))
	(xs, ys) = cell_centers(esim)
	tstring = @sprintf("%s; n=%4d t=% 2.4e",title, frame, t)
	heatmap(xs, ys, data_field', aspect_ratio=:equal, dpi=600, size=(1000,1000), xlabel=L"x", ylabel=L"y", title=tstring, titlefontface="Computer Modern")
end

# ╔═╡ 55c06a26-bc2e-45f4-bb6f-68e1f000aa57
@bind i_esim Slider(1:n_tsteps(esim))

# ╔═╡ 14085c9d-f7fa-42e8-a50d-027751243bec
plot_scalar_field(esim, i_esim, "Pressure") do esim, n
	pf = pressure_field(esim, n, DRY_AIR)
	map(pf) do val
		isnothing(val) ? 0. : val
	end
end

# ╔═╡ 5190ba18-858a-4937-bc98-adddd8e110f8
begin
	p = plot_scalar_field(esim, i_esim, "Density") do esim, n
		pf = density_field(esim, n)
		map(pf) do val
			isnothing(val) ? 0. : ustrip(val)
		end
	end
	plot!(p, @view(billig_data[:, 1]), @view(billig_data[:,2]), label="Billig Curve", lw=4, color=:red, ls=:dashdot)
end

# ╔═╡ 7e55f0d8-86ba-4f21-8121-b9d3a8d89c80
plot_scalar_field(esim, i_esim, "Mach Number") do esim, n
	mf = mach_number_field(esim, n, DRY_AIR)
	mapslices(mf, dims=(1,)) do m
		any(isnothing.(m)) ? 0. : norm(m)
	end
end

# ╔═╡ 61fed571-73fa-44b3-9c9d-360fee0a26d6
begin
	mf = mach_number_field(esim, i_esim, DRY_AIR)
	is_supersonic = reshape(mapslices(mf, dims=1) do m
		return !(any(isnothing, m) || norm(m) > 1.0)
	end, size(mf)[2:end])
	
	heatmap(cell_centers(esim)..., is_supersonic', aspect_ratio=:equal, xlims=(-2, 0), ylims=(-1.5, 1.5))
end

# ╔═╡ Cell order:
# ╠═1b7ead20-2f14-11ef-28d1-03ad74bf304a
# ╠═1b1c0430-bacf-454a-a0bf-b00c7b723b48
# ╠═01380877-d1cb-4ac7-ad08-8c8c57d95dcc
# ╠═59540798-0328-4df1-816f-ec5121c8ab16
# ╠═cf9e166b-a3e7-4082-9e06-fe67ab66f762
# ╠═89f497a5-2e05-44b5-a947-f3fcc31094ff
# ╠═35061132-50bf-4dd8-8178-ddf25a78b2f6
# ╠═14085c9d-f7fa-42e8-a50d-027751243bec
# ╠═5190ba18-858a-4937-bc98-adddd8e110f8
# ╠═7e55f0d8-86ba-4f21-8121-b9d3a8d89c80
# ╟─61fed571-73fa-44b3-9c9d-360fee0a26d6
# ╠═55c06a26-bc2e-45f4-bb6f-68e1f000aa57
