-- Measurement-to-objective boundary: 601 launched ions, detected hits and validity.
-- The workbench sends one scalar cost to this example's optimizer, not raw hits.
-- Physical outputs and Eq. (1) of CPOpt. Independent of SIMION for verification.
-- FW50 convention: diameter of an AXIS-CENTRED circle containing half the hits.
-- Linear interpolation of sample quantiles. The paper does not specify this detail.
local function quantile(a, p)
    if #a == 0 then
        return 0
    end
    table.sort(a)
    local t = 1 + (#a - 1) * p
    local k = math.floor(t)
    return a[k] + (t - k) * ((a[k + 1] or a[k]) - a[k])
end

-- Model contract: measure() returns raw diagnostics, cost (scalar minimization)
-- and objectives (vector minimization). CPOpt scores are defined only here.
-- A new model may replace these formulas and inputs without editing Algorithms/.
-- Keep objective order/length fixed and use finite penalties for invalid flights;
-- do not let missing detector hits look like a perfect zero-width spot.
local function measure(hits, total, reference_hit, dmax, weight)
    local radii, y, z = {}, {}, {}
    for i, h in ipairs(hits) do
        radii[i] = math.sqrt(h[1] ^ 2 + h[2] ^ 2);
        y[i], z[i] = h[1], h[2]
    end
    local d = 2 * quantile(radii, 0.5)
    local transmission = total > 0 and #hits / total or 0
    -- Physical transmission includes every launched ion in its denominator.
    -- The reference gate below affects the optimization scores, not this count.
    local f1, f2 = 0, 0
    if reference_hit then
        if d > 0 and d < dmax then
            f1 = 1 - d / dmax
        end
        f2 = transmission
    end
    -- Return raw measurements alongside the scores, so a small spot caused by
    -- heavy particle loss remains visible rather than looking unconditionally good.
    return {
        fw50_mm = d,
        fw50_y_mm = quantile(y, 0.75) - quantile(y, 0.25),
        fw50_z_mm = quantile(z, 0.75) - quantile(z, 0.25),
        transmission = transmission,
        f1 = f1,
        f2 = f2,
        objectives = { 1 - f1, 1 - f2 },
        cost = weight * (1 - f1) + (1 - weight) * (1 - f2)
    }
end
return { measure = measure }
