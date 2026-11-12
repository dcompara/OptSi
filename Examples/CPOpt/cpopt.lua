-- Advanced teaching demo: nominal mode or six-voltage scalar search at fixed radii.
-- One objective dispatch runs the selected 601-ion ensemble and measures its hits.
-- The physics evaluator owns Fast Adjust, detector validity and the scalar cost.
-- values()/result() exchange parameters and one measured cost; count flights apart.
-- Fixed-geometry CPOpt demonstration. Work in Examples/CPOpt; see README.md.
-- Only this file is the workbench program. Edit controls in cpopt_config.lua.
simion.workbench_program()
local C = dofile('cpopt_config.lua')
package.path = '../../Algorithms/?.lua;' .. package.path
local Optimizer = require 'Optimization_Algorithms'
local metrics = dofile('cpopt_metrics.lua')
local Detector = dofile('cpopt_detector.lua')
local unpack = unpack or table.unpack
local voltage = { unpack(C.voltage_start) }
local max_tof_us, tqual = C.max_tof_us, C.trajectory_quality
local max_step_mm, detector_step_control = C.max_step_mm, C.detector_step_control
local spot_weight = C.spot_weight
local hits, total, reference_hit, detector, result

function segment.fast_adjust()
    adj_elect01, adj_elect05, adj_elect09, adj_elect10 = 0, 0, 0, 0
    adj_elect02, adj_elect03, adj_elect04 = voltage[1], voltage[2], voltage[3]
    adj_elect06, adj_elect07, adj_elect08 = voltage[4], voltage[5], voltage[6]
end

function segment.initialize_run()
    hits, total, reference_hit = {}, 0, false
    detector = Detector.new {
        detector_x = C.detector_x,
        detector_radius = C.detector_radius,
        max_tof_us = max_tof_us
    }
    sim_trajectory_quality = tqual
end

function segment.initialize()
    detector:initialize(ion_number, ion_px_mm, ion_py_mm, ion_pz_mm)
end

function segment.tstep_adjust()
    -- SIMION has already chosen its normal time step. These optional limits
    -- only shorten it; the default max_step_mm=0 adds no global spatial cap.
    if max_step_mm > 0 then
        local speed = math.sqrt(ion_vx_mm ^ 2 + ion_vy_mm ^ 2 + ion_vz_mm ^ 2)
        if speed > 0 then
            ion_time_step = math.min(ion_time_step, max_step_mm / speed)
        end
    end
    -- T.Qual does not know about our virtual measuring plane. This local
    -- constant-velocity estimate reduces overshoot; crossing interpolation
    -- still handles acceleration and small numerical differences.
    if detector_step_control == 1 and ion_vx_mm > 0 and ion_px_mm < C.detector_x then
        ion_time_step = math.min(ion_time_step, (C.detector_x - ion_px_mm) / ion_vx_mm)
    end
end

function segment.other_actions()
    if
        detector:advance(
            ion_number,
            ion_px_mm,
            ion_py_mm,
            ion_pz_mm,
            ion_vx_mm,
            ion_time_of_flight,
            ion_splat ~= 0
        )
    then
        ion_splat = 1
    end
end

function segment.terminate()
    detector:terminate(ion_number)
end

function segment.terminate_run()
    detector:check_complete()
    hits, total, reference_hit = detector.hits, detector.total, detector.reference_hit
    assert(total > 0, "No particles initialized: check the FLY2 file and PA placement.")
    -- EDIT HERE for another experiment: collect its measurements, then pass
    -- them to your own metrics function. See cpopt_metrics.lua for the contract.
    result = metrics.measure(hits, total, reference_hit, C.dmax_mm, spot_weight)
end

-- The geometry is fixed: prepare.lua builds its complete family once, before IOB loading.
-- Each call below changes voltages by Fast Adjust and runs the unchanged detector.
function segment.flym()
    assert(C.mode == 0 or C.mode == 1, 'Use mode 0 (nominal) or 1 (fixed geometry)')
    assert(C.spot_weight > 0 and C.spot_weight < 1, 'spot_weight must be between zero and one')
    assert(
        C.maxcalls >= 1 and C.maxcalls == math.floor(C.maxcalls),
        'maxcalls must be a positive integer'
    )
    local fields = dofile('cpopt_fields.lua')
    local gem = assert(io.open('cpopt.gem', 'rb'))
    assert(
        fields.gem == gem:read('*a') and fields.signature == C.field_signature(),
        'Geometry inputs changed; prepare a fresh source copy'
    )
    gem:close()
    assert(
        fields.convergence == C.refine_convergence_V and fields.mesh == C.default_mesh_mm,
        'Prepared fields use another profile; use a fresh source copy and prepare again'
    )
    for i = 1, 6 do
        assert(
            fields.radii[i] == C.radius_start[i],
            'Prepared geometry differs from configured radii'
        )
    end
    for i = 0, 10 do
        local f =
            assert(io.open('cpopt.pa' .. i, 'rb'), 'Run prepare.lua before opening the workbench')
        f:close()
    end
    sim_rerun_flym, sim_grouped = 0, 0
    sim_trajectory_image_control = C.mode == 0 and 0 or 3
    local prefix = 'cpopt_result_' .. os.date('%Y%m%d_%H%M%S')
    local serial = 0
    while true do
        local f = io.open(prefix .. '.csv', 'r')
        if not f then
            break
        end
        f:close()
        serial = serial + 1
        prefix = 'cpopt_result_' .. os.date('%Y%m%d_%H%M%S') .. '_' .. serial
    end
    local log = assert(io.open(prefix .. '.csv', 'w'))
    log:write(
        'evaluation,request,V2_V,V3_V,V4_V,V6_V,V7_V,V8_V,FW50_mm,FW50_y_mm,FW50_z_mm,transmission,cost,status,reference_hit,detected,launched,detector_miss,timed_out,returned_upstream,stopped_elsewhere,physical_flights\n'
    )
    local calls, flights, best, best_voltage, best_result = 0, 0, nil, nil, nil

    local function evaluate(values, request)
        calls = calls + 1
        voltage = { unpack(values) }
        local outside = 0
        for i = 1, 6 do
            outside = outside
                + (math.max(
                        C.voltage_bounds[1] - voltage[i],
                        0,
                        voltage[i] - C.voltage_bounds[2]
                    ) / C.voltage_step_V)
                    ^ 2
        end
        if outside > 0 then
            local row = { calls, request }
            for _, v in ipairs(voltage) do
                row[#row + 1] = string.format('%.17g', v)
            end
            for i = 9, 22 do
                row[i] = ''
            end
            row[13], row[14], row[22] = 2 + outside, 'out_of_bounds', flights
            log:write(table.concat(row, ','), '\n')
            log:flush()
            return 2 + outside -- Historical finite penalty; no fabricated optics.
        end
        local applied = { [1] = 0, [5] = 0, [9] = 0, [10] = 0 }
        for i, e in ipairs(C.electrodes) do
            applied[e] = voltage[i]
        end
        simion.wb.instances[1].pa:fast_adjust(applied)
        run()
        flights = flights + 1
        assert(total == C.launched_particles, 'Incomplete initialized particle ensemble')
        local status = not reference_hit and 'reference_lost'
            or (result.f1 == 0 and 'spot_penalized' or 'ok')
        if reference_hit and (not best or result.cost < best) then
            best, best_voltage, best_result = result.cost, { unpack(voltage) }, result
        end
        local row = { calls, request }
        for _, v in ipairs(voltage) do
            row[#row + 1] = string.format('%.17g', v)
        end
        for _, key in ipairs { 'fw50_mm', 'fw50_y_mm', 'fw50_z_mm', 'transmission', 'cost' } do
            row[#row + 1] = string.format('%.17g', result[key])
        end
        row[#row + 1] = status
        row[#row + 1] = reference_hit and 1 or 0
        row[#row + 1], row[#row + 2] = #hits, total
        for _, key in ipairs {
            'detector_miss',
            'timed_out',
            'returned_upstream',
            'stopped_elsewhere'
        } do
            row[#row + 1] = detector.counts[key]
        end
        row[#row + 1] = flights
        log:write(table.concat(row, ','), '\n')
        log:flush()
        print(
            string.format(
                'Evaluation %d: cost=%.12g FW50=%.12g mm transmission=%d/%d %s',
                calls,
                result.cost,
                result.fw50_mm,
                #hits,
                total,
                status
            )
        )
        return result.cost
    end
    local status, requests, rejected = 'nominal', 0, 0
    if C.mode == 0 then
        evaluate(C.voltage_start, 0)
    else
        local lower, upper, step = {}, {}, {}
        for i = 1, 6 do
            lower[i], upper[i], step[i] = C.voltage_bounds[1], C.voltage_bounds[2], C.voltage_step_V
        end
        local options = { lower = lower, upper = upper, seed = C.seed }
        for key, value in pairs(C.options) do
            options[key] = value
        end
        local opt = Optimizer {
            algorithm = C.algorithm,
            start = C.voltage_start,
            step = step,
            maxcalls = C.maxcalls,
            minradius = C.minradius,
            options = options
        }
        while opt:running() do
            opt:result(evaluate({ opt:values() }, opt:ncalls()))
        end
        status, requests, rejected = opt:status(), opt:ncalls(), opt:nrejected()
    end
    log:close()
    local summary = assert(io.open(prefix .. '_best.txt', 'w'))
    summary:write(
        'status=',
        status,
        '\nalgorithm=',
        C.algorithm,
        '\nmode=',
        C.mode,
        '\nseed=',
        C.seed,
        '\nbudget=',
        C.maxcalls,
        '\noptimizer_requests=',
        requests,
        '\nphysical_flights=',
        flights,
        '\nFast_Adjust_API_calls=',
        flights,
        '\nflight_Refines=0\npreparation_Refines=1\nrejected_requests=',
        rejected,
        '\n'
    )
    if best then
        summary:write(
            string.format(
                'cost=%.17g\nFW50_mm=%.17g\ntransmission=%.17g\n',
                best,
                best_result.fw50_mm,
                best_result.transmission
            ),
            'V2,V3,V4,V6,V7,V8=',
            table.concat(best_voltage, ','),
            '\n'
        )
        print('Best voltages (V2,V3,V4,V6,V7,V8): ' .. table.concat(best_voltage, ', '))
    end
    summary:close()
    print('Results: ' .. prefix .. '.csv and ' .. prefix .. '_best.txt')
end
