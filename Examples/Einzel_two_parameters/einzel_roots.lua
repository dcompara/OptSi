-- Match signed position and angle differences; V2 in V, r2 in mm.
simion.workbench_program()
package.path = '../../Algorithms/?.lua;' .. package.path
local detector_x = 80 -- mm; both rays start at x=20 mm
local V2, current_radius, pa, flights, refines
local hits, previous, launched, terminated, result_file
-- Prescribed signed position and angle targets (mm, mrad):
local target_y, target_angle = 0.9, -6
local tolerance_y, tolerance_angle = 0.001, 0.01

function segment.fast_adjust()
    adj_elect01, adj_elect02, adj_elect03 = 0, V2 or 30, 0
end

local function sample()
    return {x=ion_px_mm, y=ion_py_mm, vx=ion_vx_mm, vy=ion_vy_mm}
end

-- Mirror the example's result lines to a text file.
local function report(line)
    print(line)
    result_file:write(line, '\n')
    result_file:flush()
end

function segment.initialize_run()
    hits, previous, launched, terminated = {}, {}, 0, 0
    sim_trajectory_quality = 3
end

function segment.initialize()
    launched = launched + 1
    previous[ion_number] = sample()
end

function segment.other_actions()
    local id, b = ion_number, sample()
    local a = previous[id]
    if a and not hits[id] and a.x<detector_x and b.x>=detector_x and ion_splat==0 then
        local t = (detector_x-a.x)/(b.x-a.x)
        local y = a.y + t*(b.y-a.y)
        local vx, vy = a.vx+t*(b.vx-a.vx), a.vy+t*(b.vy-a.vy)
        assert(vx>0 and vx<math.huge and math.abs(vy)<math.huge, 'Invalid arrival velocity')
        hits[id] = {y=y, angle=1000*math.atan(vy/vx)} -- forward XY angle, mrad
        ion_splat = 1 -- virtual detector stop, after measuring the crossing
    end
    previous[id] = b
    assert(ion_time_of_flight<100, 'Ray did not reach the detector')
end

function segment.terminate()
    terminated = terminated + 1
end

local function update_geometry(radius)
    if radius==current_radius then return end -- voltage-only changes reuse the family
    _G.einzel_r2 = radius -- read by einzel.gem
    local built = simion.open_gem('einzel.gem'):to_pa()
    built:save('working.pa#')
    built:close()
    pa:load('working.pa#')
    local delta = pa:refine{convergence=1e-5, max_overrelaxation=0.95}
    assert(pa.refined and delta<=1e-5, 'Refine did not converge')
    pa:load('working.pa0') -- load the complete refined family before Fast Adjust
    current_radius, refines = radius, refines+1
end

local function evaluate(x)
    update_geometry(x[2])
    V2, flights = x[1], flights+1
    run() -- actual SIMION flight of both rays
    assert(launched==2 and terminated==2 and hits[1] and hits[2], 'Both rays must arrive')
    local dy = hits[2].y - hits[1].y
    local da = hits[2].angle - hits[1].angle
    assert(math.abs(dy)<math.huge and math.abs(da)<math.huge, 'Nonfinite measurement')
    report(string.format('flight=%d V2=%.3f r2=%.4f dy_mm=%+.4f dtheta_mrad=%+.3f arrivals=2/2',
        flights, V2, x[2], dy, da))
    return dy, da -- signed ray 2 minus ray 1, in mm and mrad
end

function segment.flym()
    result_file = assert(io.open('results.txt', 'w')) -- replaced on each run
    pa = simion.wb.instances[1].pa
    pa:load('einzel.pa0') -- included family at r2=4 mm; never overwrite it
    current_radius, flights, refines = 4, 0, 0
    sim_rerun_flym, sim_grouped, sim_trajectory_image_control = 0, 0, 1
    local opt = require('Optimization_Algorithms'){
        algorithm='Newton', start={68,4.1}, step={2,0.5}, maxcalls=40, -- or 'Broyden'
        options={lower={30,3.4}, upper={85,4.8}, residual_scale={tolerance_y,tolerance_angle},
            ftol=1, xtol=1e-5, fd_step=0.05}}
    while opt:running() do
        local dy, da = evaluate{opt:values()}
        opt:result{dy-target_y, da-target_angle} -- retain both signs
    end
    current_radius = nil -- force an independent field rebuild and Refine
    sim_trajectory_image_control = 0
    local dy, da = evaluate(assert(opt:best()))
    report(string.format('Signed target errors: %+.4g mm, %+.4g mrad', dy-target_y, da-target_angle))
    assert(opt:status()=='converged' and math.abs(dy-target_y)<=tolerance_y
        and math.abs(da-target_angle)<=tolerance_angle, 'Root targets not verified')
    report(string.format('status=%s requests=%d flights=%d family_refines=%d',
        opt:status(), opt:ncalls(), flights, refines))
    result_file:close()
end
