-- Scalar Efficient Global Optimization, entirely Lua 5.1.
-- Jones, Schonlau & Welch (1998), doi:10.1023/A:1008306431147.
-- Uses ParEGO's unchanged ordinary-kriging GP and stable logEI (Ament et al.,
-- NeurIPS 2023). Only self:f calls are physical objective requests.
local U = require 'optimizer_utils'
local GP = require 'surrogate'
local M = {}

local function distance(a, b)
    local value = 0
    for j = 1, #a do value = value + (a[j] - b[j]) ^ 2 end
    return value
end

local function key(x)
    local parts = {}
    for j = 1, #x do parts[j] = string.format('%.17g', x[j]) end
    return table.concat(parts, ':')
end

local function reflect(x)
    local t = x % 2
    return t <= 1 and t or 2 - t
end

-- Keep the measured incumbent, newest valid point and a spread in observation
-- order. The complete measured best/archive is independent of this fit limit.
local function subset(points, values, limit, best)
    if #points <= limit then return points, values end
    local indexes, seen = {}, {}
    local function add(i)
        if not seen[i] and #indexes < limit then
            indexes[#indexes + 1], seen[i] = i, true
        end
    end
    add(best); add(#points); add(1)
    for k = 0, limit - 1 do add(1 + math.floor(k * (#points - 1) / (limit - 1) + 0.5)) end
    for k = #points, 1, -1 do add(k) end
    table.sort(indexes)
    local x, y = {}, {}
    for k, i in ipairs(indexes) do x[k], y[k] = points[i], values[i] end
    return x, y
end

function M.run(self)
    local o = U.options(self, {
        seed = 1, gp_nugget = 1e-9, gp_sweeps = 2, gp_max_points = 80,
        gp_refit_interval = 5, acquisition_starts = 4, acquisition_iterations = 16
    })
    local d = U.domain(self, o)
    local n, budget = #d.active, self._maxcalls
    assert(U.finite(budget) and budget > 0 and budget == math.floor(budget),
        'EGO requires a finite positive integer maxcalls budget')
    if o.initial_points == nil then
        o.initial_points = math.min(2 * n + 1, math.max(1, math.floor(budget / 3)))
    end
    if o.candidates == nil then o.candidates = math.max(128, 64 * n) end
    U.number(o, 'initial_points', 1, 1000000, true)
    U.number(o, 'candidates', 8, 100000, true)
    U.number(o, 'gp_nugget', 1e-15, 0.1)
    U.number(o, 'gp_sweeps', 0, 20, true)
    U.number(o, 'gp_max_points', 3, 1000000, true)
    U.number(o, 'gp_refit_interval', 1, 1000000, true)
    U.number(o, 'acquisition_starts', 1, 100, true)
    U.number(o, 'acquisition_iterations', 1, 100, true)
    local random, normal = U.generator(o.seed)
    local lo, span, start = {}, {}, {}
    self._radius = 0
    for i, j in ipairs(d.active) do
        assert(o.lower and o.upper and U.finite(o.lower[j]) and U.finite(o.upper[j]),
            'EGO requires finite lower and upper bounds on every active variable')
        lo[i], span[i] = o.lower[j], o.upper[j] - o.lower[j]
        assert(U.finite(span[i]) and span[i] > 0, 'invalid EGO interval width')
        start[i] = (self._start[j] - lo[i]) / span[i]
        self._radius = math.max(self._radius, span[i])
    end
    local points, values, attempted, seen = {}, {}, {}, {}
    local best_index, invalid, iteration, warm_length = nil, 0, 0, nil
    local function physical(z)
        local x = U.copy(self._start)
        for i, j in ipairs(d.active) do
            x[j] = math.max(o.lower[j], math.min(o.upper[j], lo[i] + span[i] * z[i]))
        end
        return x
    end
    local function nearest(z)
        local result = math.huge
        for _, p in ipairs(attempted) do result = math.min(result, distance(z, p)) end
        return result
    end
    local function novel(z)
        return nearest(z) >= 1e-14 and not seen[key(physical(z))]
    end
    local function evaluate(z, info, x)
        if not d.available() then return false end
        x = x or physical(z)
        self._request_info = info
        -- No pcall: Lua callback exceptions retain the shared abort contract.
        local value = self:f(x)
        attempted[#attempted + 1], seen[key(x)] = U.copy(z), true
        if U.finite(value) then
            points[#points + 1], values[#values + 1] = U.copy(z), value
            if not best_index or value < values[best_index] then best_index = #values end
        else
            invalid = invalid + 1
        end
        return true
    end
    -- Use the exact supplied start, avoiding a normalization round trip.
    evaluate(start, { phase = 'initial', iteration = 0 }, U.copy(self._start))
    if n == 0 then self._stop_reason = 'fixed_variables'; return end

    local count = math.min(o.initial_points - 1, budget - self:ncalls())
    local design = {}
    for k = 1, count do design[k] = {} end
    for j = 1, n do
        local bins = {}
        for k = 1, count do bins[k] = k end
        for k = count, 2, -1 do
            local h = math.floor(random() * k) + 1
            bins[k], bins[h] = bins[h], bins[k]
        end
        for k = 1, count do design[k][j] = (bins[k] - 1 + random()) / count end
    end
    for k = 1, count do
        if novel(design[k]) then evaluate(design[k], { phase = 'ego_design', iteration = 0, member = k }) end
    end
    while d.available() do
        iteration = iteration + 1
        local model, training_count, used_nugget = nil, 0, o.gp_nugget
        local refit = not warm_length or (iteration - 1) % o.gp_refit_interval == 0
        local best = 0
        if #values >= 2 then
            -- Affine objective normalization only, never log(physical cost).
            -- Divide before subtracting to avoid overflow for opposite extremes.
            local magnitude, low, high = 0, math.huge, -math.huge
            for _, v in ipairs(values) do magnitude = math.max(magnitude, math.abs(v)) end
            magnitude = magnitude > 0 and magnitude or 1
            for _, v in ipairs(values) do low, high = math.min(low, v / magnitude), math.max(high, v / magnitude) end
            local range, normalized = high - low, {}
            for k, v in ipairs(values) do normalized[k] = range > 0 and (v / magnitude - low) / range or 0 end
            best = normalized[best_index]
            local fit_points, fit_values = subset(points, normalized, o.gp_max_points, best_index)
            training_count = #fit_points
            -- Bounded nugget escalation if Cholesky fails; no fabricated data.
            while true do
                model = GP.fit(fit_points, fit_values, {
                    nugget = used_nugget, sweeps = refit and o.gp_sweeps or 0,
                    length = warm_length, fixed_length = warm_length ~= nil and not refit
                })
                if model and U.finite(model.mean) and U.finite(model.variance)
                    and U.finite(model.likelihood) then break end
                model = nil
                if used_nugget >= 0.1 then break end
                used_nugget = math.min(0.1, used_nugget * 100)
            end
            if model then warm_length = U.copy(model.length) end
        end
        local function acquisition(z)
            if not novel(z) or not model then return -math.huge end
            local mean, variance = model:predict(z)
            if not U.finite(mean) or not U.finite(variance) or variance < 0 then return -math.huge end
            return GP.log_expected_improvement(mean, variance, best)
        end
        local candidates, farthest, separation = {}, nil, -1
        local function consider(z)
            if not novel(z) then return end
            local dist = nearest(z)
            if dist > separation then farthest, separation = U.copy(z), dist end
            candidates[#candidates + 1] = { x = z, value = acquisition(z), index = #candidates + 1 }
        end
        for k = 1, o.candidates do
            local z = {}
            for j = 1, n do
                if n == 1 and k % 2 == 1 then
                    z[j] = (k - 1) / (o.candidates - 1)
                elseif best_index and k % 2 == 1 then
                    local base = k % 4 == 1 and points[best_index] or points[math.floor(random() * #points) + 1]
                    z[j] = reflect(base[j] + 0.2 * normal())
                else z[j] = random() end
            end
            consider(z)
        end
        for j = 1, n do
            for _, bound in ipairs { 0, 1 } do
                local z = U.copy(best_index and points[best_index] or start)
                z[j] = bound; consider(z)
            end
        end
        table.sort(candidates, function(a, b)
            return a.value > b.value or (a.value == b.value and a.index < b.index)
        end)
        local proposal, maximum = farthest, -math.huge
        if #candidates > 0 then proposal, maximum = candidates[1].x, candidates[1].value end
        if model then
            for k = 1, math.min(o.acquisition_starts, #candidates) do
                local center, value, mesh = U.copy(candidates[k].x), candidates[k].value, 0.2
                for _ = 1, o.acquisition_iterations do
                    local improved = false
                    for j = 1, n do
                        for _, sign in ipairs { -1, 1 } do
                            local z = U.copy(center)
                            z[j] = math.max(0, math.min(1, z[j] + sign * mesh))
                            local score = acquisition(z)
                            if score > value then center, value, improved = z, score, true end
                        end
                    end
                    if not improved then mesh = mesh * 0.5 end
                end
                if value > maximum then proposal, maximum = center, value end
            end
        end
        if maximum == -math.huge then proposal = farthest end
        if not proposal or not novel(proposal) then self._stop_reason = 'design_resolution'; break end
        self._ego_state = {
            iteration = iteration, valid_observations = #values, invalid_evaluations = invalid,
            gp_training_points = training_count, gp_nugget = used_nugget,
            gp_hyperparameter_refit = refit, log_expected_improvement = maximum,
            length_scales = warm_length and U.copy(warm_length)
        }
        evaluate(proposal, {
            phase = maximum > -math.huge and 'ego_logei' or 'ego_space_filling', iteration = iteration,
            gp_training_points = training_count, invalid_evaluations = invalid,
            gp_hyperparameter_refit = refit and 1 or 0, log_expected_improvement = maximum
        })
    end
    if not self._stop_reason then self._stop_reason = 'budget' end
end

return M
