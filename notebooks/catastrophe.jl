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

# ╔═╡ 6f1542ea-a747-11ef-2466-fd7f67d1ef2c
begin
	using Euler2D
	using Euler2D: TangentQuadCell
	using ForwardDiff
	using LaTeXStrings
	using LinearAlgebra
	using Interpolations
	using Plots
	using PlutoUI
	using Printf
	using ShockwaveProperties
	using StaticArrays
	using Unitful
end

# ╔═╡ 31009964-3f32-4f97-8e4a-2b95be0f0037
using PlanePolygons

# ╔═╡ 88fd25c1-0df7-4e12-a3a0-95f2d7178829
using Accessors

# ╔═╡ d4157e10-eea4-4ebb-bb53-7474d2609241
using BenchmarkTools

# ╔═╡ 0679a676-57fa-45ee-846d-0a8961562db3
begin
	using Graphs
	using MetaGraphsNext
end

# ╔═╡ f5fd0c28-99a8-4c44-a5e4-d7b24e43482c
PlutoUI.TableOfContents()

# ╔═╡ e7001548-a4b4-4709-b10c-0633a11bd624
md"""
# Numerical GTVs

"""

# ╔═╡ c87b546e-8796-44bf-868c-b2d3ad340aa1
md"""
## Setup
Declare ``u(\vec{x}, 0; p) = u_0`` and provide a useful scale to to nondimensionalize the Euler equations.

The parameters (taken from a previously-done simulation) are:
 - ``ρ_\inf=0.662\frac{\mathrm{kg}}{\mathrm{m}^3}``
 - ``M_\inf=4.0``
 - ``T_\inf=220\mathrm{K}``

"""

# ╔═╡ afc11d27-1958-49ba-adfa-237ba7bbd186
function u0(x, p)
	# ρ, M, T -> ρ, ρv, ρE
    pp = PrimitiveProps(p[1], SVector(p[2], 0.0), p[3])
    return ConservedProps(pp, DRY_AIR)
end

# ╔═╡ 0df888bd-003e-4b49-9c2a-c28a7ccc33d2
const ambient = u0((-Inf, 0.), SVector(0.662, 4.0, 220.0))

# ╔═╡ 3fe1be0d-148a-43f2-b0a5-bb177d1c041d
sim_scale = EulerEqnsScaling(
	1.0u"m", 
	ShockwaveProperties.density(ambient),
	speed_of_sound(ambient,DRY_AIR),
)

# ╔═╡ b23f691e-371e-42c8-86a4-ef534587c699
ambient_pressure = Euler2D._pressure(nondimensionalize(ambient, sim_scale), DRY_AIR)

# ╔═╡ c1a81ef6-5e0f-4ad5-8e73-e9e7f09cefa6
function dimensionless_speed_of_sound(
    u_star::SVector{N,T},
    gas::CaloricallyPerfectGas,
) where {N,T}
    P_star = Euler2D._pressure(u_star, gas)
    return sqrt(gas.γ * (P_star / u_star[1]))
end

# ╔═╡ e55363f4-5d1d-4837-a30f-80b0b9ae7a8e
function dimensionless_mach_number(u_star::SVector{N, T}, gas::CaloricallyPerfectGas) where {N, T}
    a = dimensionless_speed_of_sound(u_star, gas)
    ρa = u_star[1] * a
    return Euler2D.select_middle(u_star) ./ ρa
end

# ╔═╡ d832aeb4-42d6-4b72-88ee-4cdd702a4f48
md"""
Load up a data file. This contains a forward-mode computation on a fine grid allowed to run to $T=20$.
"""

# ╔═╡ 90bf50cf-7254-4de8-b860-938430e121a9
tangent=load_cell_sim("../data/circular_obstacle_tangent_longtime.celltape");

# ╔═╡ 33e635b3-7c63-4b91-a1f2-49da93307f29
md"""
We also know that this simulation was done with a blunt, cylindrical obstacle of radius ``0.75`` located at the origin.
"""

# ╔═╡ 4dc7eebf-48cc-4474-aef0-0cabf1d8eda5
body = CircularObstacle(SVector(0.,0.), 0.75);

# ╔═╡ 8bd1c644-1690-46cf-ac80-60654fc6d8c0
md"""
## Pressure Field Sensitivities
This mirrors the declaration of `pressure_field`, but returns `missing` values when there's no pressure field value to compute.
"""

# ╔═╡ d14c3b81-0f19-4207-8e67-13c09fd7636a
md"""
Computing the full gradient ``\nabla_pP`` is a bit finicky, but ultimately works out to be repeated Jacobian-vector products over the pressure field.
"""

# ╔═╡ 893ec2c8-88e8-4d72-aab7-88a1efa30b47
function dPdp(sim::CellBasedEulerSim{T, C}, n) where {T, C<:Euler2D.TangentQuadCell}
	_, u_cells = nth_step(sim, n)
	res = Array{Union{T, Missing}}(missing, (3, grid_size(sim)...))
	for i∈eachindex(IndexCartesian(), sim.cell_ids)
		sim.cell_ids[i] == 0 && continue
		cell = u_cells[sim.cell_ids[i]]
		dP = ForwardDiff.gradient(cell.u) do u
			Euler2D._pressure(u, DRY_AIR)
		end
		res[:, i] = dP'*cell.u̇
	end
	return res 
end

# ╔═╡ ce89ba5f-4858-426d-88ec-51e1f03987e1
let
	tdata = map(pressure_field(tangent, n_tsteps(tangent), DRY_AIR)) do val
		isnothing(val) ? missing : val
	end
	tempplot = heatmap(tdata', xlims=(0,300), ylims=(0,450), aspect_ratio=:equal, title=L"P", size=(400, 500), dpi=1000)
	savefig(tempplot, "../gfx/pressure_plot_longtime.pdf")
end

# ╔═╡ 2e3b9675-4b66-4623-b0c4-01acdf4e158c
@bind n Slider(2:n_tsteps(tangent); show_value=true)

# ╔═╡ f6147284-02ec-42dd-9c2f-a1a7534ae9fa
pfield = map(pressure_field(tangent, n, DRY_AIR)) do val
	isnothing(val) ? missing : val
end;

# ╔═╡ cc53f78e-62f5-4bf8-bcb3-5aa72c5fde99
pressure_tangent = dPdp(tangent, n);

# ╔═╡ d5db89be-7526-4e6d-9dec-441f09606a04
begin
	pplot = heatmap(pfield', xlims=(0,300), ylims=(0,450), aspect_ratio=:equal, title=L"P")
	cbar_limits = (:auto, (-2, 10), :auto)
	titles = [L"\frac{\partial P}{\partial \rho_\inf}", L"\frac{\partial P}{\partial M_\inf}", L"\frac{\partial P}{\partial T_\inf}"]
	dpplot = [
		heatmap((@view(pressure_tangent[i, :, :]))', xlims=(0,300), ylims=(0,450), aspect_ratio=:equal, clims=cbar_limits[i], title=titles[i]) for i=1:3
	]
	plots = reshape(vcat(pplot, dpplot), (2,2))
	xlabel!.(plots, L"i")
	ylabel!.(plots, L"j")
	plot(plots..., size=(800,800), dpi=1000)
end

# ╔═╡ 4e9fb962-cfaa-4650-b50e-2a6245d4bfb4
@bind n2 confirm(Slider(1:n_tsteps(tangent), show_value=true))

# ╔═╡ bcdd4862-ac68-4392-94e2-30b1456d411a
let
	dPdM = dPdp(tangent, n2)
	title = L"\frac{\partial P}{\partial M_\inf}"
	p = heatmap((@view(dPdM[2, :, :]))', xlims=(0,300), ylims=(0,450), aspect_ratio=:equal, clims=(-10, 10), title=title, size=(450, 600))
	p
end

# ╔═╡ 7604c406-c0a3-45bb-9109-389c7a47b8b3
let
	n=268
	dPdM = dPdp(tangent, n)
	title = L"\frac{\partial P}{\partial \rho},\,n=%$(n)"
	p = heatmap((@view(dPdM[1, :, :]))', xlims=(0,300), ylims=(0,450), aspect_ratio=:equal, title=title, size=(450, 500), dpi=1000)
	#savefig(p, "../gfx/dpdrho_plot_$n.pdf")
	p
end

# ╔═╡ e2bdc923-53e6-4a7d-9621-4d3b356a6e41
md"""
## Shock Sensitivities
"""

# ╔═╡ 44ff921b-09d0-42a4-8852-e911212924f9
md"""
### Shock Sensor
Implementation of the technique proposed in _Canny-Edge-Detection/Rankine-Hugoniot-conditions unified shock sensor for inviscid and viscous flows_.
"""

# ╔═╡ 4f8b4b5d-58de-4197-a676-4090912225a1
md"""
---
"""

# ╔═╡ 706146ae-3dbf-4b78-9fcc-e0832aeebb28
_diff_op(T) = SVector{3,T}(one(T), zero(T), -one(T))

# ╔═╡ 9b6ab300-6434-4a96-96be-87e30e35111f
_avg_op(T) = SVector{3,T}(one(T), 2 * one(T), one(T))

# ╔═╡ 21cbdeec-3438-4809-b058-d23ebafc9ee2
function convolve_sobel(matrix::AbstractMatrix{T}) where {T}
    Gy = _avg_op(T) * _diff_op(T)'
    Gx = _diff_op(T) * _avg_op(T)'
    new_size = size(matrix) .- 2
    outX = similar(matrix, new_size)
    outY = similar(matrix, new_size)
    for i ∈ eachindex(IndexCartesian(), outX, outY)
        view_range = i:(i+CartesianIndex(2, 2))
        outX[i] = Gx ⋅ @view(matrix[view_range])
        outY[i] = Gy ⋅ @view(matrix[view_range])
    end
    return outX, outY
end

# ╔═╡ 90ff1023-103a-4342-b521-e229157001fc
function discretize_gradient_direction(θ)
    if -π / 8 ≤ θ < π / 8
        return 0
    elseif π / 8 ≤ θ < 3 * π / 8
        return π / 4
    elseif 3 * π / 8 ≤ θ < 5 * π / 8
        return π / 2
    elseif 5 * π / 8 ≤ θ < 7 * π / 8
        return 3 * π / 4
    elseif 7 * π / 8 ≤ θ
        return π
    elseif -3 * π / 8 ≤ θ < -π / 8
        return -π / 4
    elseif -5 * π / 8 ≤ θ < -3 * π / 8
        return -π / 2
    elseif -7 * π / 8 ≤ θ < -5 * π / 8
        return -3π / 4
    elseif θ < -7 * π / 8
        return -π
    end
end

# ╔═╡ 5c0be95f-3c4a-4062-afeb-3c1681cae549
function gradient_grid_direction(θ)
    if -π / 8 ≤ θ < π / 8
        return CartesianIndex(1, 0)
    elseif π / 8 ≤ θ < 3 * π / 8
        return CartesianIndex(1, 1)
    elseif 3 * π / 8 ≤ θ < 5 * π / 8
        return CartesianIndex(0, 1)
    elseif 5 * π / 8 ≤ θ < 7 * π / 8
        return CartesianIndex(-1, 1)
    elseif 7 * π / 8 ≤ θ
        return CartesianIndex(-1, 0)
    elseif -3 * π / 8 ≤ θ < -π / 8
        return CartesianIndex(1, -1)
    elseif -5 * π / 8 ≤ θ < -3 * π / 8
        return CartesianIndex(0, -1)
    elseif -7 * π / 8 ≤ θ < -5 * π / 8
        return CartesianIndex(-1, -1)
    elseif θ < -7 * π / 8
        return CartesianIndex(-1, 0)
    end
end

# ╔═╡ 88889293-9afc-4540-a2b9-f30afb62b1de
function mark_edge_candidate(dP2_view, Gx, Gy)
    grid_theta = gradient_grid_direction(atan(Gy, Gx))
    idx = CartesianIndex(2, 2)
    return dP2_view[idx+grid_theta] < dP2_view[idx] &&
           dP2_view[idx-grid_theta] < dP2_view[idx]
end

# ╔═╡ 6da05b47-9763-4d0c-99cc-c945630c770d
#assumes stationary shock "edge"
function rh_error_lab_frame(cell_front, cell_behind, θ, gas)
    m1 = dimensionless_mach_number(cell_front.u, gas)
    m2 = dimensionless_mach_number(cell_behind.u, gas)
    dir = sincos(θ)
    n̂ = SVector(dir[2], dir[1])
    m_ratio = ShockwaveProperties.shock_normal_mach_ratio(m1, n̂, gas)
    m1_norm = abs(m1 ⋅ n̂)
    m2_norm_rh = m1_norm * m_ratio
    m2_norm_sim = abs(m2 ⋅ n̂)
    return (
			abs(m2_norm_rh - m2_norm_sim) / m2_norm_sim, 
			abs(m1_norm / m2_norm_sim - 1)
		   )
end

# ╔═╡ 351d4e18-4c95-428e-a008-5128f547c66d
function find_shock_in_timestep(
    sim::CellBasedEulerSim{T,C},
    t,
    gas;
    rh_rel_error_max = 0.5,
    continuous_variation_thold = 0.01,
) where {T,C}
    # TODO really gotta figure out how to deal with nothings or missings in this matrix
    pfield = map(p -> isnothing(p) ? zero(T) : p, pressure_field(sim, t, gas))
    Gx, Gy = convolve_sobel(pfield)
    dP2 = Gx.^2+Gy.^2
    edge_candidates = Array{Bool,2}(undef, size(dP2) .- 2)
    window_size = CartesianIndex(2, 2)
    for i ∈ eachindex(IndexCartesian(), edge_candidates)
        edge_candidates[i] = mark_edge_candidate(
            @view(dP2[i:i+window_size]),
            Gx[i+CartesianIndex(1, 1)],
            Gy[i+CartesianIndex(1, 1)],
        )
    end
    @info "Number of candidates..." n_candidates = sum(edge_candidates)
    Gx_overlay = @view(Gx[2:end-1, 2:end-1])
    Gy_overlay = @view(Gy[2:end-1, 2:end-1])
    id_overlay = @view(sim.cell_ids[3:end-2, 3:end-2])
	num_except = 0
	num_reject_too_smooth = 0
	num_reject_rh_fail = 0
    for j ∈ eachindex(IndexCartesian(), edge_candidates, Gx_overlay, Gy_overlay, id_overlay)
        i = j + CartesianIndex(2, 2)
        if id_overlay[j] > 0 && edge_candidates[j]
            θ = atan(Gy_overlay[j], Gx_overlay[j])
            θ_disc = discretize_gradient_direction(θ)
            θ_grid = gradient_grid_direction(θ_disc)
            # gradient points in direction of steepest increase...
            # cell in "front" of shock should be opposite the gradient?
            id_front = sim.cell_ids[i-θ_grid]
            id_back = sim.cell_ids[i+θ_grid]
            if id_front == 0 || id_back == 0
                edge_candidates[j] = false
                continue
            end

            cell_front = sim.cells[t][id_front]
            cell_back = sim.cells[t][id_back]
            try
                rh_err, sim_err = rh_error_lab_frame(cell_front, cell_back, θ_disc, gas)
                if rh_err > rh_rel_error_max
                    # discard edge candidate
					num_reject_rh_fail += 1
                    edge_candidates[j] = false
				elseif sim_err < continuous_variation_thold
					num_reject_too_smooth += 1
					edge_candidates[j] = false
                end
            catch de
				if de isa DomainError
                #@warn "Cell shock comparison caused error" typ=typeof(de) j θ_grid
                edge_candidates[j] = false
				num_except += 1
				else
					rethrow()
				end
            end
        else
            edge_candidates[j] = false
        end
    end
    @info "Number of candidates after RH condition thresholding..." n_candidates =
        sum(edge_candidates) num_except num_reject_rh_fail num_reject_too_smooth
    return edge_candidates
end

# ╔═╡ bc0c6a41-adc8-4d18-9574-645704f54b72
md"""
---
"""

# ╔═╡ 4a5086bc-5c36-4e71-9d3a-8f77f48a90f9
md"""
The implemented shock sensor has some issues (we need to set ``TOL=0.7``), but produces a reasonable result on this test data. We can clearly see the shock front that forms in front of the body, and the other "shocks" are likely numerical effects from the terribly-done rasterization of the body.

``TOL`` is much more relaxed here than in the original paper, but the sensor was originally tested on data generated using a MUSCL implementation.
"""

# ╔═╡ 747f0b67-546e-4222-abc8-2007daa3f658
@bind rh_err Slider(0.0:0.01:2.5, show_value=true, default=0.7)

# ╔═╡ 2cb33e16-d735-4e60-82a5-aa22da0288fb
@bind smoothness_err Slider(0.000:0.005:0.2, show_value=true, default=0.1)

# ╔═╡ 4b036b02-1089-4fa8-bd3a-95659c9293cd
# ╠═╡ show_logs = false
sf = find_shock_in_timestep(tangent, 268, DRY_AIR; rh_rel_error_max=rh_err, continuous_variation_thold=smoothness_err);

# ╔═╡ 24da34ca-04cd-40ae-ac12-c342824fa26e
let
	data = map(sf, @view tangent.cell_ids[3:end-2, 3:end-2]) do v1, v2
		if v2 == 0
			missing
		else
			v1
		end
	end
	p = heatmap(cell_centers(tangent, 1)[3:end-2], cell_centers(tangent, 2)[3:end-2], data', cbar=false, aspect_ratio=:equal, xlims=(-2, 0), ylims=(-1.5, 1.5), xlabel=L"x", ylabel=L"y", size=(350, 500),dpi=1000)
	savefig(p, "../gfx/shock_sensor_07_01.pdf")
	p
end

# ╔═╡ 92044a9f-2078-48d1-8181-34be87b03c4c
md"""
### Deriving ``\xi``

We are primarily interested in what happens when we vary any of the parameters in the ambient flow change. 

If we take the cell-average data available from the simulation, we can choose a (perhaps even slightly) coarser grid than was used for the simulation, and extract a linear interpolation of the bow shock.
"""

# ╔═╡ 5268fe37-b827-42ad-9977-6efbf4ecfad1
md"""
It's also necessary to expand the existing interface to Euler2D to polygon types...
"""

# ╔═╡ 63b23272-bda5-4947-9baf-4825dbf7e6fe
function cell_boundary_polygon(cell::Euler2D.QuadCell)
	c = cell.center
	dx, dy = cell.extent/2
	return SClosedPolygon(c + SVector(dx, -dy), 
				 		c + SVector(-dx, -dy),
				 		c + SVector(-dx, dy),
						c + SVector(dx, dy))
end

# ╔═╡ 4d202323-e1a9-4b24-b98e-7d17a6cc144f
struct CoarseQuadCell{T, NS, NTAN}
	id::Int
	pts::SClosedPolygon{4, T}
	# wall_length::T
	# wall_normal::SVector{2, T}
	u::SVector{4, T}
	u̇::SMatrix{4, NS, T, NTAN}
end

# ╔═╡ 95947312-342f-44b3-90ca-bd8ad8204e18
function cell_boundary_polygon(cell::CoarseQuadCell)
	return cell.pts
end

# ╔═╡ eb5a2dc6-9c7e-4099-a564-15f1cec11caa
md"""
---
"""

# ╔═╡ 9c601619-aaf1-4f3f-b2e2-10422d8ac640
function shock_cells(sim, n, shock_field)
	sort(reduce(vcat, filter(!isnothing, map(enumerate(eachcol(shock_field))) do (j, col)
		i = findfirst(col)
		isnothing(i) && return nothing
		id = @view(sim.cell_ids[2:end-1, 2:end-1])[i, j]
		return sim.cells[n][id]
	end)); lt=(a, b) -> a.center[2] < b.center[2])
end

# ╔═╡ e0a934d6-3962-46d5-b172-fb970a537cc0
function shock_points(sim::CellBasedEulerSim{T, C}, n, shock_field) where {T, C}
	sp = shock_cells(sim, n, shock_field)
	res = Matrix{T}(undef, (length(sp), 2))
	for i ∈ eachindex(sp)
		res[i, :] = sp[i].center
	end
	#sort!(sp; lt=(a, b) -> a[2] < b[2])
	return res
end

# ╔═╡ 62ebd91b-8980-4dc5-b61b-ba6a21ac357d
all_shock_points = shock_points(tangent, 267, sf);

# ╔═╡ be8ba02d-0d31-4720-9e39-595b549814cc
sp_interp = linear_interpolation(all_shock_points[:, 2], all_shock_points[:,1]);

# ╔═╡ a1dc855f-0b24-4373-ba00-946719d38d95
md"""
---
"""

# ╔═╡ 93043797-66d1-44c3-b8bb-e17deac70cfa
md"""
If we take a set of points along the ``y``-axis, we can create cells that have vertices on any of the:
 - computational domain boundaries
 - bow shock
 - blunt body

We can construct cells from these points and apply the conservation law again to compute ``εξ``.
"""

# ╔═╡ b10f8077-1daf-4dae-9c39-88aab5a1a3bb
slope_above = (all_shock_points[end, 2] - all_shock_points[end-2, 2])/(all_shock_points[end, 1]-all_shock_points[end-2, 1])

# ╔═╡ 2f088a0c-165e-47f9-aaeb-6e4ab31c9d26
slope_below = (all_shock_points[3, 2] - all_shock_points[1, 2])/(all_shock_points[3, 1]-all_shock_points[1, 1])

# ╔═╡ 2577067f-a880-4990-9340-a8face5ee5b4
function x_shock(y)
	if all_shock_points[1, 2] < y < all_shock_points[end, 2]
		return sp_interp(y)
	elseif all_shock_points[1, 2] ≥ y
		return (y-all_shock_points[1,2])/slope_below+all_shock_points[1,1]
	else
		return (y-all_shock_points[end,2])/slope_above+all_shock_points[end,1]
	end	
end

# ╔═╡ 7468fbf2-aa57-4505-934c-baa4dcb646fc
cell_width_at_shock = 0.025

# ╔═╡ 24fa22e6-1cd0-4bcb-bd6d-5244037e58e2
x_midleft(y) = x_shock(y) - cell_width_at_shock

# ╔═╡ cd312803-3819-4451-887b-ce2b53bb6e1b
x_between(y) = x_shock(y) + cell_width_at_shock

# ╔═╡ ac412980-1013-450f-bb23-0dc7c2b3f199
function x_body(y)
	if y > 0.75 || y < -0.75
		return 0.0
	else
		return -sqrt(0.75^2 - y^2)
	end
end

# ╔═╡ d44322b1-c67f-4ee8-b168-abac75fb42a1
ypts2 = range(all_shock_points[1, 2], all_shock_points[end, 2]; length=60);

# ╔═╡ 2ae618e1-89a8-45ac-ae60-77ab24ec3b56
polys_farleft = [SClosedPolygon(SVector(
	Point(x_midleft(y1), y1), 
	Point(-1.5, y1), 
	Point(-1.5, y2), 
	Point(x_midleft(y2), y2))) 
for (y1, y2) ∈ zip(ypts2[1:end-1], ypts2[2:end])]

# ╔═╡ 6f034bbe-bd04-4be4-af53-b53b3ec17942
polys_midleft = [SClosedPolygon(SVector(
	Point(x_shock(y1), y1), 
	Point(x_midleft(y1), y1), 
	Point(x_midleft(y2), y2), 
	Point(x_shock(y2), y2))) 
for (y1, y2) ∈ zip(ypts2[1:end-1], ypts2[2:end])]

# ╔═╡ 63edc638-b7aa-4a63-8c93-e860dd4d58f5
polys_between = [SClosedPolygon(
	Point(x_between(y1), y1), 
	Point(x_shock(y1), y1), 
	Point(x_shock(y2), y2), 
	Point(x_between(y2), y2))
for (y1, y2) ∈ zip(ypts2[1:end-1], ypts2[2:end])]

# ╔═╡ 7c394fb3-a75a-4bfd-a781-f31845569693
polys_body = [SClosedPolygon(
	Point(x_body(y1), y1), 
	Point(x_between(y1), y1), 
	Point(x_between(y2), y2), 
	Point(x_body(y2), y2)) 
for (y1, y2) ∈ zip(ypts2[1:end-1], ypts2[2:end])]

# ╔═╡ 729ebc48-bba1-4858-8369-fcee9f133ee0
function is_cell_contained_by(
	cell::Union{Euler2D.QuadCell, CoarseQuadCell}, closed_poly::ClockwiseOrientedPolygon
)
	return all(edge_starts(cell_boundary_polygon(cell))) do p
		PlanePolygons.point_inside(closed_poly, p)
	end
end

# ╔═╡ 5cffaaf5-9a5e-4839-a056-30e238308c51
function is_cell_overlapping(
	cell::Union{Euler2D.QuadCell, CoarseQuadCell}, closed_poly::ClockwiseOrientedPolygon
)
	contained = is_cell_contained_by(cell, closed_poly)
	if contained
		return false
	end
	return !isnothing(poly_intersection(cell_boundary_polygon(cell), closed_poly))
end

# ╔═╡ f252b8d0-f067-468b-beb3-ff6ecaeca722
function all_cells_contained_by(poly, sim)
	_, cells = nth_step(sim, 1)
	return filter(sim.cell_ids) do id
		id == 0 && return false
		return is_cell_contained_by(cells[id], poly)
	end
end

# ╔═╡ 571b1ee7-bb07-4b30-9870-fbd18349a2ef
function all_cells_overlapping(poly, sim)
	_, cells = nth_step(sim, 1)
	return filter(sim.cell_ids) do id
		id == 0 && return false
		return is_cell_overlapping(cells[id], poly)
	end
end

# ╔═╡ 80cde447-282a-41e5-812f-8eac044b0c15
function overlapping_cell_area(cell1, cell2)
	isect = poly_intersection(cell_boundary_polygon(cell1), cell_boundary_polygon(cell2))
	if isnothing(isect)
		return 0.
	end
	return poly_area(isect)
end

# ╔═╡ 5d9e020f-e35b-4325-8cc1-e2a2b3c246c9
function compute_coarse_cell_contents(coarse_cell::CoarseQuadCell{T, NS, NTAN}, sim::CellBasedEulerSim{T, TangentQuadCell{T, NS, NTAN}}, n) where {T, NS, NTAN}
	contained = all_cells_contained_by(coarse_cell.pts, sim)
	overlapped = all_cells_overlapping(coarse_cell.pts, sim)
	u_a = mapreduce(+, contained) do id
		_, cs = nth_step(sim, n)
		return Euler2D.cell_volume(cs[id]) * cs[id].u
	end
	u_b = mapreduce(+, overlapped) do id
		_, cs = nth_step(sim, n)
		A = overlapping_cell_area(cs[id], coarse_cell)
		return A*cs[id].u
	end
	u̇_a = mapreduce(+, contained) do id
		_, cs = nth_step(sim, n)
		return Euler2D.cell_volume(cs[id]) * cs[id].u̇
	end
	u̇_b = mapreduce(+, overlapped) do id
		_, cs = nth_step(sim, n)
		A = overlapping_cell_area(cs[id], coarse_cell)
		return A*cs[id].u̇
	end
	return (u_a+u_b , u̇_a + u̇_b) ./ poly_area(cell_boundary_polygon(coarse_cell))
end

# ╔═╡ 71ceb9c6-1451-43b1-b558-cf969fb9758d
Euler2D.nondimensionalize(ambient, Euler2D._SI_DEFAULT_SCALE)

# ╔═╡ 3b49eb2a-81be-413a-b06f-5ee70896bd99
testp = SClosedPolygon(SVector(SVector(0., 0.), SVector(-1.0, 0.), SVector.(-1., -1.), SVector(0., -1), SVector(0., 0.)))

# ╔═╡ c1abff36-74a1-4631-ab64-d106c21168e0
poly_area(testp)

# ╔═╡ f30619a3-5344-4e81-a4b5-6a11100cd056
empty_coarse = Dict([id=>CoarseQuadCell(id, poly, zero(SVector{4, Float64}), zero(SMatrix{4, 3, Float64, 12})) for (id, poly)∈enumerate(vcat(polys_farleft, polys_midleft,polys_between,polys_body))])

# ╔═╡ 5d77d782-2def-4b3a-ab3a-118bf8e96b6b
coarse_cells = let
	d = Dict([id=>CoarseQuadCell(id, poly, zero(SVector{4, Float64}), zero(SMatrix{4, 3, Float64, 12})) for (id, poly)∈enumerate(vcat(polys_farleft, polys_midleft,polys_between,polys_body))]);
	for c∈keys(d)
		v1, v2 = compute_coarse_cell_contents(d[c], tangent, 267)
		@reset d[c].u = v1
		@reset d[c].u̇ = v2
	end
	d
end

# ╔═╡ 875bb4ce-41ab-4b01-9b3b-5f7dcac4c006
(sum(all_cells_contained_by(coarse_cells[22].pts, tangent)) do id
	_, c = nth_step(tangent, 267)
	return c[id].u * Euler2D.cell_volume(c[id])
end + sum(all_cells_overlapping(coarse_cells[22].pts, tangent)) do id
	_, c = nth_step(tangent, 267)
	A = overlapping_cell_area(c[id], coarse_cells[22])
	return c[id].u * A
end) / poly_area(coarse_cells[22].pts)

# ╔═╡ c6e3873e-7fef-4c38-bf3f-de71f866057f
let
	xc = body.center[1] .+ body.radius .* cos.(0:0.01:2π)
	yc = body.center[2] .+ body.radius .* sin.(0:0.01:2π)
	p = plot(xc, yc, aspect_ratio=:equal, xlims=(-2, 0), ylims=(-1.5, 1.5), label=false, fill=true, dpi=1000, size=(800, 1000))
	id = 0
	maxdensity = maximum(coarse_cells) do (_, c)
		c.u[1]
	end
	for (id, cell) ∈ coarse_cells
		poly = cell.pts
		pl = reduce(hcat, edge_starts(poly))
		plot!(p, @view(pl[1, :]), @view(pl[2, :]) , lw=0.5, fill=true, fillalpha=(cell.u[1]/maxdensity), label=false, color=:red, seriestype=:shape)
		v = sum(eachcol(pl))/4
		annotate!(p,v..., Plots.text(L"P_{%$id}", 8))
	end
	spys = range(-1.5, 1.5; length=50)
	spxs = x_shock.(spys)
	plot!(p, spxs, spys, label="Strong Shock Front", lw=4)
	#savefig(p, "../gfx/silly_rectangles.pdf")
	p
end

# ╔═╡ 8f36cc9d-c2c5-4bea-9dc7-e5412a2960f9
let
	poly = cell_boundary_polygon(empty_coarse[1])
	pts = mapreduce(pt->pt', vcat, edge_starts(poly))
	# pts = vcat(pts, first(edge_starts(poly))')
	p = plot(pts[:,1], pts[:,2], lw=0.5, fill=true, ylims=(-1.0, -0.8), xlims=(-1.6, -0.9), dpi=1000, fillalpha=0.5, seriestype=:shape)
	foreach(all_cells_overlapping(empty_coarse[1].pts, tangent)) do id
		_, c = nth_step(tangent, 267)
		data = mapreduce(pt->pt', vcat, edge_starts(cell_boundary_polygon(c[id])))
		plot!(p, data[:,1], data[:,2], lw=0.5, fill=true, fillcolor=:red, fillalpha=0.5, seriestype=:shape, label=false)
	end
	for ell ∈ edge_lines(poly)
		x1 = ell.p[1] - 10*ell.dir[1]
		x2 = ell.p[1] + 10*ell.dir[1]
		y1 = ell.p[2] - 10*ell.dir[2]
		y2 = ell.p[2] + 10*ell.dir[2]
		plot!(p, [x1, x2], [y1,y2], color=:black, label=false, ls=:dash)
	end
	data = mapreduce(vcat, all_cells_contained_by(empty_coarse[1].pts, tangent)) do id
		_, c = nth_step(tangent, 267)
		return Vector(c[id].center)'
	end
	scatter!(p, data[:, 1], data[:,2], marker = :x, ms=2)
	p
end

# ╔═╡ bec2e760-1bb1-4ba0-8f17-7d28f9b17d01
with_dims = Euler2D.redimensionalize(coarse_cells[17].u, Euler2D._SI_DEFAULT_SCALE)

# ╔═╡ 2860e012-c8a7-4794-9aeb-d9ae53228482
coarse_cells[17].u

# ╔═╡ ac6224b2-b4f1-4e75-b8b3-15f95dceff87
pressure(with_dims, DRY_AIR) |> Base.Fix1(uconvert, u"Pa")

# ╔═╡ 6394c62c-9b9f-4b0c-9bdd-ec2b171f366e
pressure(ambient, DRY_AIR) |> Base.Fix1(uconvert, u"Pa")

# ╔═╡ 1d6fe61f-2ae8-464f-bed6-1f9a50a5cf9e
begin
	INTERNAL_CODE::Int = 0
	EAST_CODE::Int = length(coarse_cells)+1
	WEST_CODE::Int = length(coarse_cells)+2
	SOUTH_CODE::Int = length(coarse_cells)+3
	NORTH_CODE::Int = length(coarse_cells)+4
	HARDWALL_CODE::Int = length(coarse_cells)+5
	edge_codes = (EAST_CODE, WEST_CODE, SOUTH_CODE, NORTH_CODE)
end

# ╔═╡ 0e0a049b-e2c3-4fe9-8fb8-186cdeb60485
function are_coarse_neighbors(c1, c2)
	e1 = hcat(c1.pts, c1.pts[:, 1])
	e2 = reduce(hcat, Iterators.reverse(eachcol(hcat(c2.pts, c2.pts[:, 1]))))
	select_starts = SVector(ntuple(i->i, 4))
	select_ends = SVector(ntuple(i->i+1, 3)..., 1)

	#clockwise around c1
	for (p1, p2) ∈ zip(eachcol(e1[:, select_starts]), eachcol(e1[:, select_ends]))
		# counterclockwise around c2
		for (q1, q2) ∈ zip(eachcol(e2[:, select_starts]), eachcol(e2[:, select_ends]))
			if (
				isapprox(norm(p1-q1), 0.; atol=1.0e-9) && 
				isapprox(norm(p2-q2), 0.; atol=1.0e-9)
			)
				return (true, norm(p1-p2))
			end
		end
	end
	return (false, 0.0)
end

# ╔═╡ 148c3af7-f815-4a89-ae6c-7d6134432984
coarse_dual = let
	g = MetaGraph(
		Graph(),
		Int,
		Union{Int, valtype(coarse_cells)},
		Float64,
	)
	for (k, v) ∈ coarse_cells
		g[k] = v
	end
	for c1 ∈ labels(g)
		for c2 ∈ labels(g)
			a, b = are_coarse_neighbors(g[c1], g[c2])
			if a
				g[c1, c2] = b
			end
		end
	end
	g
end

# ╔═╡ ed865040-1d8e-4c93-849c-55b0da6ccee1
function get_edge_code(p1, p2)
	ll = tangent.bounds[1]
	ur = tangent.bounds[2]
	if p1[1] == ll[1] && p2[1] == ll[1]
		return EAST_CODE
	elseif p1[2] == ll[2] && p2[2] == ll[2]
		return SOUTH_CODE
	elseif p1[1] == ur[1] && p2[1] == ur[1]
		return WEST_CODE
	elseif p1[2] == ur[2] && p2[2] == ur[2]
		return NORTH_CODE
	elseif norm(p1) ≈ 0.75 && norm(p2) ≈ 0.75
		return HARDWALL_CODE
	end
	return 0
end

# ╔═╡ 5b7a3783-ef40-468f-93ac-91cb46929bd6
ne(coarse_dual)

# ╔═╡ 74525445-19f6-471f-878e-a60f07ba9f01
nv(coarse_dual)

# ╔═╡ d9be4c0f-4dc4-4538-8f6e-be246404334f


# ╔═╡ 3a8cd7e2-fae9-4e70-8c92-004b17352506
md"""
## Solving for ``\dot x``

Each of the points on the shock can be used to define new cells ``P_i``. For each of the original cells, as well as the new cells, we know that:
```math
\oint_{\partial P_i} F(\bar{u}_i)\cdot\hat n\,ds = 0
```

This defines a system of equations, where only the ``x``-coordinate of the points on the shock is free. ``P_{N, S, E, W}`` is the neighbor cell to the north, south, east , or west respectively.

```math
	\left(F(\bar u_i) - F(\bar u_S)\right)\hat n_{i,S}L_{i, S} + F(\bar u_i)\hat n_{i,E}L_{i, E} + F(\bar u_i)\hat n_{i,N}L_{i, N} + F(\bar u_i)\hat n_{i,W}L_{i, W} = 0
```

We can stack all of these equations into ``\mathcal G``, and then use the implicit function theorem:
```math
\begin{aligned}
0 &= \mathcal {G}(\bar u, x)\\
0 &= \nabla_u\mathcal{G}(\bar u, x)\dot u + \nabla_x\mathcal{G}(\bar u, x)\dot x\\
-\nabla_x\mathcal{G}(\bar u, x)\dot x &= \nabla_u\mathcal{G}(\bar u, x)\dot u
\end{aligned}
```

``\bar u``, ``x``, and ``\dot u`` are known, so we _should_ be able to solve this system of equations.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
"""

# ╔═╡ Cell order:
# ╠═6f1542ea-a747-11ef-2466-fd7f67d1ef2c
# ╠═f5fd0c28-99a8-4c44-a5e4-d7b24e43482c
# ╟─e7001548-a4b4-4709-b10c-0633a11bd624
# ╟─c87b546e-8796-44bf-868c-b2d3ad340aa1
# ╠═0df888bd-003e-4b49-9c2a-c28a7ccc33d2
# ╠═afc11d27-1958-49ba-adfa-237ba7bbd186
# ╠═3fe1be0d-148a-43f2-b0a5-bb177d1c041d
# ╠═b23f691e-371e-42c8-86a4-ef534587c699
# ╠═c1a81ef6-5e0f-4ad5-8e73-e9e7f09cefa6
# ╠═e55363f4-5d1d-4837-a30f-80b0b9ae7a8e
# ╟─d832aeb4-42d6-4b72-88ee-4cdd702a4f48
# ╟─90bf50cf-7254-4de8-b860-938430e121a9
# ╠═33e635b3-7c63-4b91-a1f2-49da93307f29
# ╠═4dc7eebf-48cc-4474-aef0-0cabf1d8eda5
# ╟─8bd1c644-1690-46cf-ac80-60654fc6d8c0
# ╠═f6147284-02ec-42dd-9c2f-a1a7534ae9fa
# ╟─d14c3b81-0f19-4207-8e67-13c09fd7636a
# ╟─893ec2c8-88e8-4d72-aab7-88a1efa30b47
# ╠═cc53f78e-62f5-4bf8-bcb3-5aa72c5fde99
# ╠═ce89ba5f-4858-426d-88ec-51e1f03987e1
# ╠═2e3b9675-4b66-4623-b0c4-01acdf4e158c
# ╟─d5db89be-7526-4e6d-9dec-441f09606a04
# ╠═4e9fb962-cfaa-4650-b50e-2a6245d4bfb4
# ╠═bcdd4862-ac68-4392-94e2-30b1456d411a
# ╠═7604c406-c0a3-45bb-9109-389c7a47b8b3
# ╟─e2bdc923-53e6-4a7d-9621-4d3b356a6e41
# ╟─44ff921b-09d0-42a4-8852-e911212924f9
# ╟─4f8b4b5d-58de-4197-a676-4090912225a1
# ╠═706146ae-3dbf-4b78-9fcc-e0832aeebb28
# ╠═9b6ab300-6434-4a96-96be-87e30e35111f
# ╠═21cbdeec-3438-4809-b058-d23ebafc9ee2
# ╟─90ff1023-103a-4342-b521-e229157001fc
# ╟─5c0be95f-3c4a-4062-afeb-3c1681cae549
# ╠═88889293-9afc-4540-a2b9-f30afb62b1de
# ╠═6da05b47-9763-4d0c-99cc-c945630c770d
# ╠═351d4e18-4c95-428e-a008-5128f547c66d
# ╟─bc0c6a41-adc8-4d18-9574-645704f54b72
# ╟─4a5086bc-5c36-4e71-9d3a-8f77f48a90f9
# ╠═747f0b67-546e-4222-abc8-2007daa3f658
# ╠═2cb33e16-d735-4e60-82a5-aa22da0288fb
# ╠═4b036b02-1089-4fa8-bd3a-95659c9293cd
# ╟─24da34ca-04cd-40ae-ac12-c342824fa26e
# ╟─92044a9f-2078-48d1-8181-34be87b03c4c
# ╟─5268fe37-b827-42ad-9977-6efbf4ecfad1
# ╠═31009964-3f32-4f97-8e4a-2b95be0f0037
# ╠═63b23272-bda5-4947-9baf-4825dbf7e6fe
# ╠═4d202323-e1a9-4b24-b98e-7d17a6cc144f
# ╠═95947312-342f-44b3-90ca-bd8ad8204e18
# ╟─eb5a2dc6-9c7e-4099-a564-15f1cec11caa
# ╟─9c601619-aaf1-4f3f-b2e2-10422d8ac640
# ╟─e0a934d6-3962-46d5-b172-fb970a537cc0
# ╠═62ebd91b-8980-4dc5-b61b-ba6a21ac357d
# ╠═be8ba02d-0d31-4720-9e39-595b549814cc
# ╟─a1dc855f-0b24-4373-ba00-946719d38d95
# ╟─93043797-66d1-44c3-b8bb-e17deac70cfa
# ╠═a0ae957d-26aa-48a5-a642-56cdbf1b8012
# ╠═cc0ead94-8af4-4c6b-ad05-79a2539a3271
# ╠═d8a94eb8-2752-4273-bc7b-405c6416f2b2
# ╠═8960c7e3-2234-46f7-9c5d-d41f656fe48a
# ╠═44b27e39-b2b7-4548-8091-7479fffbc470
# ╠═3323f6be-deca-4780-a877-d018b0651aeb
# ╠═38c1679e-803a-49e9-b4cc-b47b1d1ec954
# ╠═641efc27-f5ba-4970-94b1-cc8761407564
# ╠═6aa1657f-1ee3-4055-bc58-17325f894d5b
# ╠═0bfde9c6-4f22-4caf-9628-7d84706109ca
# ╠═6c8bf0e2-68df-4a09-a33f-13924560a871
# ╠═5033ce87-7695-4f21-acfd-ac952e429393
# ╠═0e4858f9-b4e4-4036-ba2f-22df03e4ecf3
# ╠═5d025c37-dde6-48fe-91f5-546a03f61309
# ╠═9564dc10-f407-4b24-a27b-f0c3e5993bb8
# ╠═d77581ac-1393-491a-b07b-462cce95f39d
# ╠═6d4e503f-24ff-49a0-b2db-7e23d14e61ee
# ╠═d9fb063e-51ce-4ace-8e84-1d9d5c120e8b
# ╠═3accd3b6-395f-4548-9b28-8d89a6be446d
# ╠═88fd25c1-0df7-4e12-a3a0-95f2d7178829
# ╠═d4157e10-eea4-4ebb-bb53-7474d2609241
# ╠═729ebc48-bba1-4858-8369-fcee9f133ee0
# ╠═5cffaaf5-9a5e-4839-a056-30e238308c51
# ╠═f252b8d0-f067-468b-beb3-ff6ecaeca722
# ╠═571b1ee7-bb07-4b30-9870-fbd18349a2ef
# ╠═80cde447-282a-41e5-812f-8eac044b0c15
# ╠═5d9e020f-e35b-4325-8cc1-e2a2b3c246c9
# ╠═875bb4ce-41ab-4b01-9b3b-5f7dcac4c006
# ╠═71ceb9c6-1451-43b1-b558-cf969fb9758d
# ╠═3b49eb2a-81be-413a-b06f-5ee70896bd99
# ╠═c1abff36-74a1-4631-ab64-d106c21168e0
# ╠═f30619a3-5344-4e81-a4b5-6a11100cd056
# ╠═5d77d782-2def-4b3a-ab3a-118bf8e96b6b
# ╠═4d202323-e1a9-4b24-b98e-7d17a6cc144f
# ╟─5d9e020f-e35b-4325-8cc1-e2a2b3c246c9
# ╠═c6e3873e-7fef-4c38-bf3f-de71f866057f
# ╟─3a8cd7e2-fae9-4e70-8c92-004b17352506
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
