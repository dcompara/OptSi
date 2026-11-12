-- Single-objective evolutionary minimizers for Optimization_Algorithms.lua.
-- All evaluations, including the initial population, go through self:f().
-- start is evaluated first; step sets each coordinate's search scale.
-- minradius is a radius in caller units, not a population size.
--
-- GA design reference: the SIMION geneticlib.lua example by D. Manura,
-- 2016-12-06, (c) 2016 Scientific Instrument Services, Inc. (Licensed SIMION 8.1).
-- This module implements roulette selection with distinct parents, whole
-- arithmetic crossover, elitism and Gaussian mutation of one active coordinate.
-- The reference example is not distributed here. See ALGORITHM_DETAILS_AND_VALIDATION.md
-- for the distinction between this historical acknowledgment and code licensing.
--
-- JADE: Zhang and Sanderson (2009), adaptive differential evolution with
-- current-to-pbest/1 mutation and an optional external archive (enabled here).
-- Population and archive updates are deferred until the generation ends.
-- Ties may replace a target, but only strict improvements train F/CR or enter
-- the archive. This conservative tie rule matters for penalized objectives.
local M = {}
local unpack = unpack or table.unpack

local function copy(x)
    return { unpack(x) }
end

local function finite(x)
    return type(x) == 'number' and x == x and x > -math.huge and x < math.huge
end

local function generator(seed)
    assert(finite(seed) and seed == math.floor(seed), 'evolutionary seed must be an integer')
    local state = seed % 2147483647
    if state == 0 then
        state = 1
    end

    local function random()
        local high = math.floor(state / 127773)
        state = 16807 * (state - high * 127773) - 2836 * high
        if state <= 0 then
            state = state + 2147483647
        end
        return state / 2147483647
    end

    local function normal()
        return math.sqrt(-2 * math.log(random())) * math.cos(2 * math.pi * random())
    end
    return random, normal
end

function M.run(self, variant)
    assert(variant == 'ga' or variant == 'jade', 'unknown evolutionary variant')
    -- All method settings use the same public options table as other solvers.
    local o = self._options or {}
    local start, step = self._start, self._step
    local random, normal = generator(o.seed or 1)
    local scale, lo, hi, active = {}, {}, {}, {}
    local dimension = #start
    if o.lower then
        assert(#o.lower == dimension, 'lower bounds have the wrong length')
    end
    if o.upper then
        assert(#o.upper == dimension, 'upper bounds have the wrong length')
    end
    for j = 1, dimension do
        assert(finite(start[j]) and finite(step[j]), 'start and step must be finite')
        local lower = o.lower and o.lower[j] or -math.huge
        local upper = o.upper and o.upper[j] or math.huge
        assert(
            type(lower) == 'number' and type(upper) == 'number' and lower <= upper,
            'invalid evolutionary bounds'
        )
        assert(start[j] >= lower and start[j] <= upper, 'start is outside evolutionary bounds')
        scale[j] = math.abs(step[j])
        if scale[j] > 0 and lower < upper then
            active[#active + 1] = j
            lo[j], hi[j] = (lower - start[j]) / scale[j], (upper - start[j]) / scale[j]
        else
            scale[j], lo[j], hi[j] = 0, 0, 0
        end
    end
    local size = o.population or math.max(8, math.min(24, 2 * #active))
    assert(
        finite(size) and size == math.floor(size) and size >= (variant == 'jade' and 4 or 2),
        'population must be an integer >= 4 for JADE or >= 2 for GA'
    )
    local spread = o.initial_spread or 2
    assert(finite(spread) and spread > 0, 'initial_spread must be positive')

    local function option(name, default, lower, upper, integer)
        local v = o[name]
        if v == nil then
            v = default
        end
        assert(
            finite(v) and v >= lower and v <= upper and (not integer or v == math.floor(v)),
            'invalid evolutionary option: ' .. name
        )
        return v
    end
    local crossover = option('crossover', 0.8, 0, 1)
    local mutation = option('mutation', 0.3, 0, 1)
    local mutation_scale = option('mutation_scale', 1, 0, math.huge)
    local elite = option('elite', 1, 1, size - 1, true)
    local pbest_fraction = option('pbest_fraction', 0.1, 1e-12, 1)
    local adaptation = option('adaptation', 0.1, 0, 1)
    local initial_F = option('initial_F', 0.5, 1e-12, 1)
    local initial_CR = option('initial_CR', 0.5, 0, 1)
    local F_spread = option('F_spread', 0.1, 0, math.huge)
    local CR_spread = option('CR_spread', 0.1, 0, math.huge)
    self._resolved_options = {
        seed = o.seed or 1,
        population = size,
        initial_spread = spread,
        lower = o.lower,
        upper = o.upper
    }
    local resolved = self._resolved_options
    if variant == 'ga' then
        resolved.crossover, resolved.mutation, resolved.mutation_scale =
            crossover, mutation, mutation_scale
        resolved.elite = elite
    else
        resolved.pbest_fraction, resolved.adaptation = pbest_fraction, adaptation
        resolved.initial_F, resolved.initial_CR = initial_F, initial_CR
        resolved.F_spread, resolved.CR_spread = F_spread, CR_spread
    end
    local budget = self._maxcalls or math.huge
    local population, archive = {}, {}
    local generation = 0

    local function available()
        return self:ncalls() < budget
    end

    local function physical(z)
        local x = {}
        for j = 1, dimension do
            x[j] = start[j] + scale[j] * z[j]
            -- Suppress roundoff outside supplied physical bounds.
            if o.lower then
                x[j] = math.max(o.lower[j], x[j])
            end
            if o.upper then
                x[j] = math.min(o.upper[j], x[j])
            end
        end
        return x
    end

    local function reflected(value, j)
        local lower, upper = lo[j], hi[j]
        if lower > -math.huge and upper < math.huge then
            local width = upper - lower
            local offset = (value - lower) % (2 * width)
            return lower + (offset <= width and offset or 2 * width - offset)
        elseif value < lower then
            return 2 * lower - value
        elseif value > upper then
            return 2 * upper - value
        end
        return value
    end

    local function refresh_radius(points)
        local centroid, raw = {}, {}
        for j = 1, dimension do
            centroid[j] = 0
        end
        for i, member in ipairs(points) do
            raw[i] = physical(member.z)
            for j = 1, dimension do
                centroid[j] = centroid[j] + raw[i][j] / #points
            end
        end
        local radius2 = 0
        for _, x in ipairs(raw) do
            local distance2 = 0
            for j = 1, dimension do
                distance2 = distance2 + (x[j] - centroid[j]) ^ 2
            end
            radius2 = math.max(radius2, distance2)
        end
        self._radius = math.sqrt(radius2)
        -- Retain physical-coordinate population snapshots for diagnostics; these
        -- are observations of the current population, not extra evaluations.
        self._points, self._values = raw, {}
        for i, member in ipairs(points) do
            self._values[i] = member.f
        end
    end

    local function evaluate(z, phase, index, factor, crossover)
        if not available() then
            return nil
        end
        self._request_info = {
            phase = phase,
            generation = generation,
            iteration = generation,
            population_size = size,
            candidate_index = index,
            population_radius = self._radius or 0,
            mutation_factor = factor,
            crossover_rate = crossover
        }
        local value = self:f(physical(z))
        return { z = copy(z), f = finite(value) and value or math.huge, finite = finite(value) }
    end

    local function sorted_indices(points)
        local indices = {}
        for i = 1, #points do
            indices[i] = i
        end
        table.sort(indices, function(a, b)
            if points[a].f == points[b].f then
                return a < b
            end
            return points[a].f < points[b].f
        end)
        return indices
    end

    -- Finite costs use scaled cost differences; a nonfinite member switches to
    -- rank weights. Equal weights fall back to uniform selection.
    local function roulette(points, excluded)
        local weights, largest, worst, count, all_finite = {}, 1, -math.huge, 0, true
        for i, member in ipairs(points) do
            if i ~= excluded then
                count = count + 1
                all_finite = all_finite and member.finite
                if member.finite then
                    largest = math.max(largest, math.abs(member.f))
                    worst = math.max(worst, member.f)
                end
            end
        end
        if all_finite then
            for i, member in ipairs(points) do
                if i ~= excluded then
                    weights[i] = worst / largest - member.f / largest
                end
            end
        else
            -- Rank fallback avoids NaN weights when an objective returns infinity.
            local indices = sorted_indices(points)
            local rank, previous, last_weight = 0, nil, nil
            for _, i in ipairs(indices) do
                if i ~= excluded then
                    rank = rank + 1
                    if previous ~= points[i].f then
                        last_weight = count - rank + 1
                    end
                    weights[i] = last_weight
                    previous = points[i].f
                end
            end
        end
        local sum = 0
        for i = 1, #points do
            sum = sum + (weights[i] or 0)
        end
        if sum == 0 then
            for i = 1, #points do
                weights[i] = i ~= excluded and 1 or 0
            end
            sum = count
        end
        local target, accumulated, last = random() * sum, 0, nil
        for i = 1, #points do
            if i ~= excluded then
                accumulated = accumulated + (weights[i] or 0)
                last = i
                if target < accumulated then
                    return i
                end
            end
        end
        return last
    end

    self._radius = 0
    local zero = {}
    for j = 1, dimension do
        zero[j] = 0
    end
    population[1] = evaluate(zero, 'initial_population', 1)
    if not population[1] then
        return
    end
    refresh_radius(population)
    if #active == 0 then
        return
    end
    for i = 2, size do
        local z = copy(zero)
        for _, j in ipairs(active) do
            z[j] = reflected(spread * (2 * random() - 1), j)
        end
        local member = evaluate(z, 'initial_population', i)
        if not member then
            return
        end
        population[i] = member
        refresh_radius(population)
    end

    local mean_F, mean_CR = initial_F, initial_CR
    while available() and self._radius >= self._minradius do
        generation = generation + 1
        local order = sorted_indices(population)
        local next_population = {}
        for i, member in ipairs(population) do
            next_population[i] = member
        end
        if variant == 'ga' then
            -- Keep the requested elite members without reevaluating them.
            for i, index in ipairs(order) do
                next_population[i] = population[index]
            end
            for i = elite + 1, size do
                local first = roulette(population)
                local second = roulette(population, first)
                local child = copy(population[first].z)
                if random() < crossover then
                    local fraction = random()
                    for _, j in ipairs(active) do
                        child[j] = fraction * child[j] + (1 - fraction) * population[second].z[j]
                    end
                end
                local mutated = 0
                if random() < mutation then
                    mutated = 1
                    local j = active[math.floor(random() * #active) + 1]
                    child[j] = reflected(child[j] + mutation_scale * normal(), j)
                end
                local member = evaluate(child, 'ga_offspring', i, mutated, crossover)
                if not member then
                    return
                end
                next_population[i] = member
                refresh_radius(next_population)
            end
        else
            local success_F, success_CR, new_archive = {}, {}, {}
            local top = math.max(2, math.ceil(pbest_fraction * size))
            for i, target in ipairs(population) do
                local F
                repeat
                    F = mean_F + F_spread * math.tan(math.pi * (random() - 0.5))
                until F > 0
                F = math.min(F, 1)
                local CR = math.max(0, math.min(1, mean_CR + CR_spread * normal()))
                local pbest = population[order[math.floor(random() * top) + 1]]
                local r1
                repeat
                    r1 = math.floor(random() * size) + 1
                until r1 ~= i
                local r2
                repeat
                    r2 = math.floor(random() * (size + #archive)) + 1
                until r2 ~= i and r2 ~= r1
                local other = r2 <= size and population[r2].z or archive[r2 - size]
                local forced = active[math.floor(random() * #active) + 1]
                local child = copy(target.z)
                for _, j in ipairs(active) do
                    if j == forced or random() < CR then
                        local value = target.z[j]
                            + F * (pbest.z[j] - target.z[j])
                            + F * (population[r1].z[j] - other[j])
                        -- JADE midpoint repair, between the target and crossed bound.
                        if value < lo[j] then
                            value = 0.5 * (lo[j] + target.z[j])
                        end
                        if value > hi[j] then
                            value = 0.5 * (hi[j] + target.z[j])
                        end
                        child[j] = value
                    end
                end
                local member = evaluate(child, 'jade_trial', i, F, CR)
                if not member then
                    return
                end
                if member.f <= target.f then
                    next_population[i] = member
                end
                if member.f < target.f then
                    new_archive[#new_archive + 1] = copy(target.z)
                    success_F[#success_F + 1] = F
                    success_CR[#success_CR + 1] = CR
                end
                refresh_radius(next_population)
            end
            for _, z in ipairs(new_archive) do
                archive[#archive + 1] = z
            end
            while #archive > size do
                table.remove(archive, math.floor(random() * #archive) + 1)
            end
            if #success_F > 0 then
                local sum_F, sum_F2, sum_CR = 0, 0, 0
                for i, F in ipairs(success_F) do
                    sum_F, sum_F2, sum_CR = sum_F + F, sum_F2 + F * F, sum_CR + success_CR[i]
                end
                mean_F = (1 - adaptation) * mean_F + adaptation * sum_F2 / sum_F
                mean_CR = (1 - adaptation) * mean_CR + adaptation * sum_CR / #success_CR
            end
        end
        population = next_population
    end
end

return M
