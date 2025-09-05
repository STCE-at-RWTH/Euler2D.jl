using Revise

using Euler2D

sim = load_cell_sim("x-data/60691549/11/bow_shock_t20.celltape")

data = Euler2D.CannyShockSensor.find_shock_in_timestep(sim, 1, DRY_AIR)

function inform_sensor_info(infos)
    @info "Shock Info:" nshock = infos.n_candidate_cells nerrs = infos.n_erred nmaxima =
        infos.n_thinned nrh_fail = infos.n_rejected_rh nsmooth = infos.n_rejected_smooth
end

inform_sensor_info(data)
