-- Teaching problem: one voltage, two actual SIMION rays and one separation cost.
-- The evaluator below owns the trajectory run and rejects missing detector hits.
-- Scalar options set voltage bounds; the optimizer receives no trajectory objects.
-- Each values() vector needs one result(cost); the final display flight is extra.

-- Work in Examples/Einzel_minimal; open einzel.iob with einzel.fly2.
simion.workbench_program()
package.path = '../../Algorithms/?.lua;' .. package.path
local Optimizer = require 'Optimization_Algorithms'
adjustable optimize = 1       -- 1: optimize; 0: fly once at view_voltage
adjustable view_voltage = 30  -- V; useful for viewing before/after trajectories
local V2, detector_x = 30, 80 -- V, mm
local y, previous, launched, flights

local function spot_cost(positions)
    assert(positions[1] and positions[2], 'Both rays must reach the detector')
    return math.abs(positions[2] - positions[1])
end

function segment.fast_adjust()
    adj_elect01, adj_elect02, adj_elect03 = 0, V2, 0
end

function segment.initialize_run()
    y, previous, launched = {}, {}, 0 -- never reuse a previous candidate's hits
    sim_trajectory_quality = 3
end

function segment.initialize()
    assert(ion_number == 1 or ion_number == 2, 'Select einzel.fly2')
    launched = launched + 1
    previous[ion_number] = {ion_px_mm, ion_py_mm}
end

function segment.other_actions()
    local id, x, yy = ion_number, ion_px_mm, ion_py_mm
    local p = previous[id]
    if p and p[1] < detector_x and x >= detector_x and ion_splat == 0 then
        local fraction = (detector_x-p[1])/(x-p[1])
        y[id] = p[2] + fraction*(yy-p[2]) -- interpolate at exactly X=80 mm
        ion_splat = 1 -- virtual stop; no detector electrode or extra electric field
    end
    previous[id] = {x, yy}
    assert(ion_time_of_flight < 100, 'Ray did not reach the detector in 100 us')
end

local function evaluate(voltage)
    V2 = voltage
    run() -- SIMION calculates both trajectories using Fast Adjust, never Refine
    flights = flights + 1
    assert(launched == 2, 'Expected exactly two launched rays')
    local cost = spot_cost(y) -- a missing ray stops the example, never a good cost
    print(string.format('flight=%d V2=%.12g y1=%.12g y2=%.12g separation_mm=%.12g arrivals=2/2',
        flights, V2, y[1], y[2], cost))
    return cost
end

function segment.flym()
    flights = 0
    sim_rerun_flym, sim_grouped = 0, 0
    if optimize == 0 then
        sim_trajectory_image_control = 0 -- view and retain this single flight
        evaluate(view_voltage)
        return
    end
    local opt = Optimizer {
        algorithm = 'Pattern_Search',
        start = {30},
        step = {15},
        maxcalls = 32,
        options = {lower={0}, upper={90}}
    }
    sim_trajectory_image_control = 1 -- view, but do not retain trial trajectories
    while opt:running() do
        local x = {opt:values()}
        opt:result(evaluate(x[1]))
    end
    local point, cost = opt:best()
    assert(point, 'No valid solution')
    print(string.format('best V2=%.12g separation_mm=%.12g status=%s requests=%d',
        point[1], cost, opt:status(), opt:ncalls()))
    sim_trajectory_image_control = 0
    evaluate(point[1]) -- one extra flight to display the best, outside the budget
end
