-- Teaching configuration, not a manuscript campaign or published optimum.
-- mode=0 measures the nominal point; mode=1 searches six voltages at fixed radii.
-- maxcalls=10 is an initialization smoke test; PSO's default population here is 12.
-- For a separate learning run, set maxcalls=100; scalar options belong in options.
-- Fixed-geometry CPOpt example. All voltages are V and lengths are mm.
-- Algorithm controls and physical inputs are configured in this file only.
local C = {
    -- Edit optimization controls here. mode=0: nominal; mode=1: six-voltage search.
    mode = 0,
    algorithm = 'PSO',
    maxcalls = 10,
    seed = 1,
    voltage_step_V = 100,
    minradius = 1e-9,
    options = {}, -- Optional method-specific settings; see ../../REFERENCE.md.
    spot_weight = 0.5,

    -- Validated numerical profile; no Einzel Refine settings are used here.
    refine_convergence_V = 0.005,
    trajectory_quality = 3,
    max_step_mm = 0,
    detector_step_control = 1,
    max_tof_us = 1000,
    launched_particles = 601,
    electrodes = { 2, 3, 4, 6, 7, 8 },
    voltage_start = { 1400, 400, 1000, 600, -1000, 1000 },
    radius_start = { 6.0, 6.0, 6.0, 6.0, 6.0, 6.0 },
    voltage_bounds = { -2000, 2000 },
    radius_bounds = { 3, 10 }, -- Sec. 4.1.
    x_begin = { 80, 105, 120, 135, 150, 175, 190, 205, 220 },
    x_end = { 100, 115, 130, 145, 170, 185, 200, 215, 240 },
    outer_radius = 15,
    end_radius = 5,
    aperture_radius = 1,
    aperture_web_begin = 158,
    aperture_web_end = 162,
    aperture_flange_radius = 12, -- Fig. 10: flange outside the thin central web.
    detector_x = 272,
    detector_radius = 20,
    -- Assumed grounded detector backing, behind the numerical measuring plane.
    detector_backing_begin = 272.1,
    detector_backing_end = 277,
    domain_x = 290,
    domain_r = 30,
    default_mesh_mm = 0.1, -- Sec. 4.1; not the earlier Sabouri mesh.
    dmax_mm = 2, -- Eq. (1).
}

-- Fields depend on geometry, not on the selected optimizer or voltage start.
function C.field_signature()
    local parts = {}
    for _, key in ipairs {
        'radius_start',
        'x_begin',
        'x_end',
        'outer_radius',
        'end_radius',
        'aperture_radius',
        'aperture_web_begin',
        'aperture_web_end',
        'aperture_flange_radius',
        'detector_radius',
        'detector_backing_begin',
        'detector_backing_end',
        'domain_x',
        'domain_r',
        'default_mesh_mm'
    } do
        local v = C[key]
        if type(v) ~= 'table' then
            v = { v }
        end
        for _, x in ipairs(v) do
            parts[#parts + 1] = string.format('%.17g', x)
        end
    end
    return table.concat(parts, ',')
end
return C
