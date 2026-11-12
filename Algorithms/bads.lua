-- BADS for deterministic objectives in Lua 5.1, with seeded randomized searches.
-- Following Acerbi & Ma (2017), Algorithm 1
-- and Supplement A/B: https://arxiv.org/html/1705.04405v2
-- Local ARD-RQ GP / MAP, LCB, two-generation ES search, diagonal and weighted
-- covariance with Exp3 hedge, length-scaled randomized LTMADS poll, mesh updates.
-- Compact implementation choices: Latin hypercube instead of Sobol design;
-- coordinate MAP optimization, rebuilt GP Cholesky instead of rank-one updates;
-- default 256 ES candidates/generation instead of 2048; a large predictive
-- residual triggers refitting instead of the paper's Shapiro-Wilk test.
-- This implements deterministic bound-constrained BADS, not all MATLAB/PyBADS
-- features: noisy objectives, periodic variables and nonlinear barriers absent.
-- Cost differences of interest should be order one: use objective_scale.
local U = require 'optimizer_utils'
local GP = require 'bads_gp'
local M = {}

local function shuffle(a, random)
    for i = #a, 2, -1 do
        local j = math.floor(random() * i) + 1;
        a[i], a[j] = a[j], a[i]
    end
end

local function identity(n)
    local a = {}
    for i = 1, n do
        a[i] = {};
        for j = 1, n do
            a[i][j] = i == j and 1 or 0
        end
    end
    return a
end

-- Duplicate detection uses this existing numerical key precision.
local function key(x)
    local t = {};
    for j = 1, #x do
        t[j] = string.format('%.15g', x[j])
    end
    return table.concat(t, ':')
end

-- A randomly permuted lower-triangular basis plus one dense special column.
-- Together with its negatives this is a positive spanning set of 2*n vectors.
-- The special column is retained at a given refinement level, as in LTMADS.
local function lt_basis(n, level, random, special)
    local q = 2 ^ level
    local item = special[level]

    local function signed()
        return random() < 0.5 and -q or q
    end

    local function integer()
        return math.floor(random() * (2 * q - 1)) - q + 1
    end
    if not item then
        item = { row = math.floor(random() * n) + 1, column = {} }
        for j = 1, n do
            item.column[j] = j == item.row and signed() or integer()
        end
        special[level] = item
    end
    local rows = {}
    for j = 1, n do
        if j ~= item.row then
            rows[#rows + 1] = j
        end
    end
    shuffle(rows, random)
    local columns = {}
    for j = 1, n - 1 do
        local v = {}
        for i = 1, n do
            v[i] = 0
        end
        for i = j, n - 1 do
            v[rows[i]] = i == j and signed() or integer()
        end
        columns[#columns + 1] = v
    end
    columns[#columns + 1] = U.copy(item.column)
    shuffle(columns, random)
    return columns, q
end

function M.run(self)
    local o = U.options(
        self,
        {
            seed = 1,
            objective_scale = 1,
            min_poll = 1e-6,
            search_candidates = 256,
            gp_sweeps = 3,
            log_transform = true,
            deterministic = true
        }
    )
    local d = U.domain(self, o)
    local n = #d.active
    assert(
        o.deterministic == true,
        'this Lua BADS implementation supports deterministic objectives only'
    )
    assert(
        o.noise == nil and o.periodic == nil and o.constraints == nil,
        'noise, periodic variables and nonlinear constraints are not supported by this BADS implementation'
    )
    assert(type(o.log_transform) == 'boolean', 'log_transform must be boolean')
    U.number(o, 'objective_scale', 1e-300, 1e300)
    U.number(o, 'min_poll', 1e-12, 0.1)
    U.number(o, 'search_candidates', 8, 100000, true)
    U.number(o, 'gp_sweeps', 1, 20, true)
    o.search_steps = o.search_steps or math.max(n, 3 + math.floor(n / 2))
    o.max_stall = o.max_stall or 4 + math.floor(n / 2)
    U.number(o, 'search_steps', 1, 1000, true)
    U.number(o, 'max_stall', 1, 10000, true)
    for _, name in ipairs { 'plausible_lower', 'plausible_upper' } do
        assert(not o[name] or #o[name] == #self._start, name .. ' length must match start')
    end
    local random, normal = U.generator(o.seed)
    -- Hard bounds constrain requests; plausible ranges and optional log transforms
    -- define internal coordinates. They are not the usual abs(step) coordinates.
    local offset, scale, lower, upper, logs, start, ranges = {}, {}, {}, {}, {}, {}, {}
    for i, j in ipairs(d.active) do
        local lo = o.lower and o.lower[j] or -math.huge
        local hi = o.upper and o.upper[j] or math.huge
        local pl = o.plausible_lower and o.plausible_lower[j]
            or (U.finite(lo) and lo or self._start[j] - 2 * math.abs(self._step[j]))
        local pu = o.plausible_upper and o.plausible_upper[j]
            or (U.finite(hi) and hi or self._start[j] + 2 * math.abs(self._step[j]))
        assert(
            U.finite(pl) and U.finite(pu) and lo <= pl and pl < pu and pu <= hi,
            'BADS requires finite plausible bounds inside hard bounds'
        )
        local transform = o.log_transform and lo > 0 and hi >= 10 * lo
        logs[i] = transform
        if transform then
            lo, hi, pl, pu = math.log(lo), math.log(hi), math.log(pl), math.log(pu)
        end
        offset[i], scale[i] = (pl + pu) / 2, (pu - pl) / 2
        assert(
            U.finite(offset[i]) and U.finite(scale[i]) and scale[i] > 0,
            'invalid BADS parameter scale'
        )
        lower[i], upper[i] = (lo - offset[i]) / scale[i], (hi - offset[i]) / scale[i]
        ranges[i] = upper[i] - lower[i]
        start[i] = ((transform and math.log(self._start[j]) or self._start[j]) - offset[i])
            / scale[i]
    end

    local function physical(z)
        local x = U.copy(self._start)
        for i, j in ipairs(d.active) do
            local v = offset[i] + scale[i] * z[i]
            if logs[i] then
                v = math.exp(v)
            end
            if o.lower then
                v = math.max(o.lower[j], v)
            end
            if o.upper then
                v = math.min(o.upper[j], v)
            end
            x[j] = v
        end
        return x
    end
    local mesh, poll = 2 ^ -10, 1
    self._radius = d.max_step
    local center, cost = U.copy(start), math.huge
    local cache, seen = {}, {}
    local model, force_refit, last_fit = nil, true, -math.huge
    local stats = {
        search_evaluations = 0,
        poll_evaluations = 0,
        gp_fits = 0,
        diagonal_searches = 0,
        covariance_searches = 0,
        poll_successes = 0,
        contractions = 0
    }
    -- Diagnostic state separates search/poll requests from cheap GP/acquisition work.
    self._bads_state = { statistics = stats, mesh = mesh, poll = poll }
    local iteration = 0

    local function evaluate(z, phase)
        local id = key(z)
        if seen[id] then
            return seen[id].value, false
        end
        if not d.available() then
            return nil, false
        end
        self._request_info =
            { phase = phase, iteration = iteration, mesh = mesh, poll = poll, search_scale = poll }
        local value = self:f(physical(z)) / o.objective_scale
        if not U.finite(value) then
            value = math.huge
        end
        local entry = { x = U.copy(z), value = value, index = #cache + 1 }
        seen[id] = entry
        if U.finite(value) then
            if model then
                local mu, var = model:predict(z)
                if math.abs(value - mu) > 4 * math.sqrt(var + model.noise) then
                    force_refit = true
                end
            end
            cache[#cache + 1] = entry
        end
        if phase == 'bads_search' then
            stats.search_evaluations = stats.search_evaluations + 1
        end
        if phase == 'bads_poll' then
            stats.poll_evaluations = stats.poll_evaluations + 1
        end
        return value, true
    end
    local initial = evaluate(start, 'initial')
    if initial == nil then
        return
    end
    cost = initial
    if n == 0 then
        self._radius = 0;
        self._stop_reason = 'fixed_variables';
        return
    end

    local function project(z, origin)
        local projected = {}
        for j = 1, n do
            local index = math.floor((z[j] - origin[j]) / mesh + 0.5)
            index = math.max(
                math.ceil((lower[j] - origin[j]) / mesh),
                math.min(math.floor((upper[j] - origin[j]) / mesh), index)
            )
            projected[j] = origin[j] + index * mesh
            if not U.finite(projected[j]) then
                return nil
            end
        end
        return projected
    end
    local design = {}
    for k = 1, n do
        design[k] = {}
    end
    for j = 1, n do
        local bins = {}
        for k = 1, n do
            bins[k] = k
        end
        shuffle(bins, random)
        for k = 1, n do
            design[k][j] = -1 + 2 * (bins[k] - 1 + random()) / n
        end
    end
    for k = 1, n do
        if not d.available() then
            break
        end
        local z = project(design[k], start)
        if z then
            local value = evaluate(z, 'bads_design')
            if value and value < cost then
                center, cost = z, value
            end
        end
    end

    local function rebuild()
        if #cache == 0 then
            return
        end
        local ranked = {}
        for k, entry in ipairs(cache) do
            local r = 0
            for j = 1, n do
                r = r + ((entry.x[j] - center[j]) / (model and model.length[j] or 1)) ^ 2
            end
            ranked[k] = { entry = entry, r = r }
        end
        table.sort(ranked, function(a, b)
            return a.r < b.r or (a.r == b.r and a.entry.index < b.entry.index)
        end)
        local shape = model and model.shape or 1
        local radius2 = 9 * shape * (math.exp(1 / shape) - 1)
        local points, values = {}, {}
        for k, item in ipairs(ranked) do
            if k > 50 and (k > 50 + 10 * n or item.r > radius2) then
                break
            end
            points[#points + 1] = item.entry.x;
            values[#values + 1] = item.entry.value
        end
        local refit = force_refit or self:ncalls() - last_fit >= 2 * n
        local updated = GP.fit(
            points,
            values,
            {
                previous = model,
                refit = refit,
                poll = poll,
                ranges = ranges,
                min_poll = o.min_poll,
                sweeps = o.gp_sweeps
            }
        )
        if updated then
            model = updated
            if refit then
                stats.gp_fits = stats.gp_fits + 1;
                last_fit = self:ncalls();
                force_refit = false
            end
        end
    end

    local function lcb(z)
        if seen[key(z)] then
            return math.huge
        end
        if not model then
            return 0
        end
        local mu, variance = model:predict(z)
        local beta = 2 * math.log(n * math.max(1, self:ncalls()) ^ 2 * math.pi ^ 2 / (6 * 0.1))
        return mu - math.sqrt(0.2 * beta * variance)
    end

    local function covariance(strategy)
        local a = identity(n)
        if strategy == 1 or not model then
            local trace = 0
            for j = 1, n do
                a[j][j] = (model and model.length[j] or 1) ^ 2;
                trace = trace + a[j][j]
            end
            for j = 1, n do
                a[j][j] = a[j][j] / trace
            end
            return assert(GP.cholesky(a))
        end
        local ranked = {}
        for k = 1, #model.points do
            ranked[k] = { x = model.points[k], value = model.values[k], index = k }
        end
        table.sort(ranked, function(x, y)
            return x.value < y.value or (x.value == y.value and x.index < y.index)
        end)
        local mean, weights, total = {}, {}, 0
        for j = 1, n do
            mean[j] = 0
        end
        for k, item in ipairs(ranked) do
            weights[k] = math.log(#ranked + 1) - math.log(k);
            total = total + weights[k]
            for j = 1, n do
                mean[j] = mean[j] + weights[k] * item.x[j]
            end
        end
        for j = 1, n do
            mean[j] = mean[j] / total
        end
        local trace = 0
        for i = 1, n do
            for j = 1, n do
                local value = 0
                for k, item in ipairs(ranked) do
                    value = value
                        + weights[k] * (item.x[i] - mean[i]) * (item.x[j] - mean[j]) / total
                end
                if i == j then
                    value = value + 1e-6 * model.length[i] ^ 2;
                    trace = trace + value
                end
                a[i][j] = value
            end
        end
        for i = 1, n do
            for j = 1, n do
                a[i][j] = a[i][j] / trace
            end
        end
        return GP.cholesky(a) or covariance(1)
    end
    local gains = { 0, 0 }

    local function search_oracle()
        local difference = math.max(-700, math.min(700, gains[2] - gains[1]))
        local p1 = 0.125 + 0.75 / (1 + math.exp(difference))
        local strategy = random() < p1 and 1 or 2
        if strategy == 1 then
            stats.diagonal_searches = stats.diagonal_searches + 1
        else
            stats.covariance_searches = stats.covariance_searches + 1
        end
        local l = covariance(strategy)

        local function sample(origin, zoom)
            local z, x = {}, {}
            for j = 1, n do
                z[j] = normal()
            end
            for i = 1, n do
                local delta = 0
                for j = 1, i do
                    delta = delta + l[i][j] * z[j]
                end
                x[i] = origin[i] + zoom * poll * delta
            end
            return project(x, center)
        end
        local first = {}
        for k = 1, o.search_candidates do
            local z = sample(center, 1)
            if z then
                first[#first + 1] = { x = z, value = lcb(z), index = k }
            end
        end
        if #first == 0 then
            return nil, strategy, p1
        end
        table.sort(first, function(a, b)
            return a.value < b.value or (a.value == b.value and a.index < b.index)
        end)
        local cumulative, total = {}, 0
        for k = 1, #first do
            total = total + 1 / math.sqrt(k);
            cumulative[k] = total
        end
        local proposal, best = nil, math.huge
        for k = 1, o.search_candidates do
            -- Systematic rank-weighted offspring allocation, with a random shift.
            local target = (k - 1 + random()) / o.search_candidates * total
            local parent = 1
            while cumulative[parent] < target do
                parent = parent + 1
            end
            local z = sample(first[parent].x, 0.25)
            if z then
                local score = lcb(z)
                if score < best then
                    proposal, best = z, score
                end
            end
        end
        if not proposal and first[1].value < math.huge then
            proposal = first[1].x
        end
        return proposal, strategy, p1
    end
    local special, stall = {}, 0
    while d.available() do
        iteration = iteration + 1
        self._bads_state.iteration, self._bads_state.mesh, self._bads_state.poll =
            iteration, mesh, poll
        local radius = 0
        local center_physical = physical(center)
        for i, j in ipairs(d.active) do
            local plus, minus = U.copy(center), U.copy(center)
            plus[i] = math.min(upper[i], center[i] + poll)
            minus[i] = math.max(lower[i], center[i] - poll)
            radius = math.max(
                radius,
                math.abs(physical(plus)[j] - center_physical[j]),
                math.abs(physical(minus)[j] - center_physical[j])
            )
        end
        self._radius = radius
        if poll < o.min_poll or radius < (self._minradius or 1e-7) then
            self._stop_reason = 'radius';
            break
        end
        if stall > o.max_stall then
            self._stop_reason = 'stall';
            break
        end
        local before = cost
        local bestx, bestcost = center, cost
        local search_success = false
        for attempt = 1, o.search_steps do
            if not d.available() then
                break
            end
            rebuild()
            local proposal, strategy, p1 = search_oracle()
            if not proposal then
                break
            end
            local value = evaluate(proposal, 'bads_search')
            if value then
                local reward = U.finite(before) and U.finite(value) and math.max(0, before - value)
                    or 0
                local decay = 0.1 ^ (1 / (2 * n))
                gains[1], gains[2] = decay * gains[1], decay * gains[2]
                gains[strategy] = gains[strategy]
                    + reward / ((strategy == 1 and p1 or 1 - p1) * poll)
                if value < bestcost then
                    bestx, bestcost = proposal, value
                end
                if before - value >= poll ^ 1.5 then
                    search_success = true;
                    break
                end
            end
        end
        local poll_success = false
        if not search_success and d.available() then
            rebuild()
            local level = math.max(0, math.ceil(-math.log(mesh) / math.log(4)))
            local columns, q = lt_basis(n, level, random, special)
            local gm = 0
            for j = 1, n do
                gm = gm + math.log(model and model.length[j] or 1) / n
            end
            gm = math.exp(gm)
            local candidates = {}
            for k, v in ipairs(columns) do
                for _, sign in ipairs { -1, 1 } do
                    local z, valid = {}, true
                    for j = 1, n do
                        local omega = math.min(
                            math.max(1e-6, mesh, (model and model.length[j] or 1) / gm),
                            U.finite(ranges[j]) and ranges[j] or 2
                        )
                        local displacement = sign * poll * v[j] / q * omega
                        local integer = math.floor(displacement / mesh + 0.5)
                        -- Preserve every nonzero triangular pivot after mesh
                        -- discretization, even for very unequal GP length scales.
                        if integer == 0 and v[j] ~= 0 then
                            integer = sign * v[j] > 0 and 1 or -1
                        end
                        z[j] = center[j] + integer * mesh
                        if z[j] < lower[j] or z[j] > upper[j] then
                            valid = false
                        end
                    end
                    if valid and not seen[key(z)] then
                        candidates[#candidates + 1] =
                            { x = z, value = lcb(z), index = 2 * k + (sign == 1 and 1 or 0) }
                    end
                end
            end
            table.sort(candidates, function(a, b)
                return a.value < b.value or (a.value == b.value and a.index < b.index)
            end)
            for _, candidate in ipairs(candidates) do
                if not d.available() then
                    break
                end
                local value = evaluate(candidate.x, 'bads_poll')
                if value and value < bestcost then
                    bestx, bestcost = candidate.x, value
                end
                if value and value < before then
                    poll_success = true;
                    break
                end
            end
        end
        center, cost = bestx, bestcost
        if before - cost >= poll ^ 1.5 then
            stall = 0
        else
            stall = stall + 1
        end
        if poll_success then
            stats.poll_successes = stats.poll_successes + 1
            poll, mesh = 2 * poll, 2 * mesh
        elseif not search_success then
            local contraction = stall > 3 and 4 or 2
            poll, mesh = poll / contraction, mesh / contraction
            stats.contractions = stats.contractions + 1
        end
        self._bads_state.center = physical(center)
        self._bads_state.gains = U.copy(gains)
        self._bads_state.length_scales = model and U.copy(model.length)
    end
    self._bads_state.mesh, self._bads_state.poll = mesh, poll
    if not self._stop_reason then
        self._stop_reason = 'budget'
    end
end
return M
