
using StaticArrays, PlanePolygons, BenchmarkTools
using Graphs, MetaGraphsNext

using Euler2D

sim1 = load_cell_sim("x-data/60691549/11/bow_shock_t20.celltape")
sim2 = load_cell_sim("x-data/60691549/16/bow_shock_t20.celltape")

data1 = Euler2D.CannyShockSensor.find_shock_in_timestep(sim1, 1, DRY_AIR)
data2 = Euler2D.CannyShockSensor.find_shock_in_timestep(sim2, 1, DRY_AIR)

function inform_sensor_info(infos)
    @info "Shock Info:" nshock = infos.n_candidate_cells nmaxima = infos.n_thinned nrh_fail =
        infos.n_rejected_rh nsmooth = infos.n_rejected_smooth
end

inform_sensor_info(data1)
inform_sensor_info(data2)

interp1 = Euler2D.CannyShockSensor.extract_bow_shock_interpolation(sim1, data1)
interp2 = Euler2D.CannyShockSensor.extract_bow_shock_interpolation(sim2, data2)

function plot_poly(poly; kwargs...)
    xs = [pt[1] for pt ∈ edge_starts(poly)]
    ys = [pt[2] for pt ∈ edge_starts(poly)]
    return plot(xs, ys; kwargs...)
end

function plot_poly!(canvas, poly; kwargs...)
    xs = [pt[1] for pt ∈ edge_starts(poly)]
    ys = [pt[2] for pt ∈ edge_starts(poly)]
    plot!(canvas, xs, ys; kwargs...)
    return canvas
end

free_stream = SVector(1.0, 4.0, 0.0, 9.78587)
bcs = (
    ExtrapolateToPhantom(),
    StrongWall(),
    ExtrapolateToPhantom(),
    ExtrapolateToPhantom(),
    StrongWall(),
)

coarse1 = Euler2D.LocalPseudoinversion.make_coarse_cell_graph(
    interp1,
    sim1,
    1,
    free_stream,
    bcs,
    DRY_AIR,
    8,
    0.075,
)
