-- Fresh source-copy field preparation only; rerun flights through cpopt.iob later.
-- Existing PA files are refused before field generation; never use this to rerun Fly'm.
-- Run in native SIMION Lua BEFORE opening cpopt.iob; see ../../REFERENCE.md.
-- Builds the fixed canonical geometry; no workbench, particles or optimizer is loaded.
local C = dofile('cpopt_config.lua')
for _, suffix in ipairs { '#', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10' } do
    local f = io.open('cpopt.pa' .. suffix, 'rb')
    if f then
        f:close()
        error(
            'Fields already exist. Preparation never overwrites them; use a fresh source copy to rebuild.'
        )
    end
end
_G.cpopt_geometry = { r = C.radius_start, h = C.default_mesh_mm }
local built = simion.open_gem('cpopt.gem'):to_pa()
built:save('cpopt.pa#')
built:close()
local pa = simion.pas:open('cpopt.pa#')
-- Only convergence is explicit. Keep every other native Refine default unchanged.
local delta = pa:refine { convergence = C.refine_convergence_V }
assert(pa.refined and delta <= C.refine_convergence_V, 'Refine did not converge')
local applied = { [1] = 0, [5] = 0, [9] = 0, [10] = 0 }
for i, e in ipairs(C.electrodes) do
    applied[e] = C.voltage_start[i]
end
pa:fast_adjust(applied)
pa:save('cpopt.pa0')
pa:close()
for i = 0, 10 do
    local f = assert(io.open('cpopt.pa' .. i, 'rb'), 'Missing native PA companion')
    f:close()
end
local gem = assert(io.open('cpopt.gem', 'rb'))
local gem_source = gem:read('*a')
gem:close()
local f = assert(io.open('cpopt_fields.lua', 'w'))
f:write(
    'return {gem=',
    string.format('%q', gem_source),
    ',signature=',
    string.format('%q', C.field_signature()),
    ',convergence=',
    string.format('%.17g', C.refine_convergence_V),
    ',mesh=',
    string.format('%.17g', C.default_mesh_mm),
    ',radii={',
    table.concat(C.radius_start, ','),
    '}}\n'
)
f:close()
print(
    string.format('CPOpt prepared: one family Refine, final Delta=%.17g V. Open cpopt.iob.', delta)
)
