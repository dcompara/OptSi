-- Pareto-based population searches, Lua 5.1; all objective components minimized.
-- GA in multiobjective mode: NSGA-II, Deb et al. (2002),
-- https://doi.org/10.1109/4235.996017
-- Nondominated sorting, normalized crowding, binary tournaments, elitist
-- parent/offspring selection, bounded SBX crossover and polynomial mutation.
-- PSO in multiobjective mode: a crowding-based MOPSO variant, inspired by
-- Coello Coello et al. (2004), https://doi.org/10.1109/TEVC.2004.826067
-- This variant uses crowding tournaments rather than the original adaptive
-- grid; it is not a reproduction of that paper's exact algorithm/settings.
--
-- Shared options: population=20, seed=1, finite lower/upper bounds on active
-- parameters. Start is evaluated first; remaining particles are uniform over
-- those bounds. step=0/equal bounds fix coordinates. Active coordinates use
-- [0,1] internally, so nonzero step magnitudes do not set the search spread.
-- GA: crossover=0.9 per pair, crossover_eta=15, mutation=1/n per coordinate,
-- mutation_eta=20. Larger eta means more local variation. n=active dimension.
-- PSO: inertia=0.7298, cognitive=social=1.49618, velocity_limit=0.2 of each
-- bound width, mutation=1/n per coordinate, mutation_eta=20, archive_size=100.
-- archive_size caps only the PSO leader set; front() retains ALL observed
-- nondominated objective vectors (one representative for exact equality).
-- Both searches stop at maxcalls, or after one evaluation if all inputs are
-- fixed. minradius is not a Pareto-convergence criterion and is not used here.
-- No global RNG changes, external executable, or simulator-specific physics.
local U = require 'optimizer_utils'
local M = {}

function M.dominates(a, b)
    local strict = false
    for j = 1, #a do
        if a[j] > b[j] then
            return false
        end
        if a[j] < b[j] then
            strict = true
        end
    end
    return strict
end

local function equal(a, b)
    for j = 1, #a do
        if a[j] ~= b[j] then
            return false
        end
    end
    return true
end

-- Copies parameter/objective vectors into the observed front; exact objective
-- equality keeps the first representative, without a dominance tolerance.
function M.update_front(front, entry)
    -- A point is retained unless another observed point is no worse in every
    -- objective and strictly better in at least one. Equal vectors share one entry.
    for _, p in ipairs(front) do
        if M.dominates(p.objectives, entry.objectives) or equal(p.objectives, entry.objectives) then
            return
        end
    end
    for i = #front, 1, -1 do
        if M.dominates(entry.objectives, front[i].objectives) then
            table.remove(front, i)
        end
    end
    front[#front + 1] = { x = U.copy(entry.x), objectives = U.copy(entry.objectives) }
end

-- Constant objectives contribute zero density information. Scale before
-- subtracting to avoid overflow for finite objectives of opposite signs.
-- Mutates crowding metadata on these entries, leaving measured objectives intact.
function M.crowding(points)
    for _, p in ipairs(points) do
        p.crowding = 0
    end
    if #points == 0 then
        return
    end
    for j = 1, #points[1].objectives do
        local order = {}
        for i = 1, #points do
            order[i] = i
        end
        table.sort(order, function(a, b)
            local x, y = points[a].objectives[j], points[b].objectives[j]
            return x < y or (x == y and a < b)
        end)
        local low = points[order[1]].objectives[j]
        local high = points[order[#order]].objectives[j]
        if high > low then
            local scale = math.max(math.abs(low), math.abs(high))
            local range = high / scale - low / scale
            points[order[1]].crowding = math.huge
            points[order[#order]].crowding = math.huge
            for k = 2, #order - 1 do
                local gap = (
                    points[order[k + 1]].objectives[j] / scale
                    - points[order[k - 1]].objectives[j] / scale
                ) / range
                points[order[k]].crowding = points[order[k]].crowding + gap
            end
        end
    end
end

-- O(m*N^2) nondominated sorting, assigning every member a rank and crowding.
-- Mutates rank/crowding metadata and returns fronts containing the same entries.
function M.rank(points)
    local dominated, counts, first = {}, {}, {}
    for i, p in ipairs(points) do
        dominated[i], counts[i] = {}, 0
        for j, q in ipairs(points) do
            if M.dominates(p.objectives, q.objectives) then
                dominated[i][#dominated[i] + 1] = j
            elseif M.dominates(q.objectives, p.objectives) then
                counts[i] = counts[i] + 1
            end
        end
        if counts[i] == 0 then
            first[#first + 1] = i
        end
    end
    local fronts, indices, rank = {}, first, 1
    while #indices > 0 do
        local front, next_indices = {}, {}
        for _, i in ipairs(indices) do
            points[i].rank = rank;
            front[#front + 1] = points[i]
            for _, j in ipairs(dominated[i]) do
                counts[j] = counts[j] - 1
                if counts[j] == 0 then
                    next_indices[#next_indices + 1] = j
                end
            end
        end
        M.crowding(front);
        fronts[#fronts + 1] = front
        indices, rank = next_indices, rank + 1
    end
    return fronts
end

local function crowded_first(a, b)
    return a.crowding > b.crowding or (a.crowding == b.crowding and a.id < b.id)
end

local function environmental_selection(combined, size)
    local selected = {}
    for _, front in ipairs(M.rank(combined)) do
        if #selected + #front > size then
            table.sort(front, crowded_first)
        end
        for _, p in ipairs(front) do
            if #selected == size then
                break
            end
            selected[#selected + 1] = p
        end
        if #selected == size then
            break
        end
    end
    return selected
end

local function tournament(points, random, use_rank)
    local a = points[math.floor(random() * #points) + 1]
    local b = points[math.floor(random() * #points) + 1]
    if use_rank and a.rank ~= b.rank then
        return a.rank < b.rank and a or b
    end
    if a.crowding ~= b.crowding then
        return a.crowding > b.crowding and a or b
    end
    return random() < 0.5 and a or b
end

local function clip(v)
    return math.max(0, math.min(1, v))
end

-- Deb's bounded polynomial mutation, in normalized parameter coordinates.
local function mutate(z, probability, eta, random)
    for j, y in ipairs(z) do
        if random() < probability then
            local r = random();
            local power = 1 / (eta + 1);
            local delta
            if r <= 0.5 then
                delta = (2 * r + (1 - 2 * r) * (1 - y) ^ (eta + 1)) ^ power - 1
            else
                delta = 1 - (2 * (1 - r) + 2 * (r - 0.5) * y ^ (eta + 1)) ^ power
            end
            z[j] = clip(y + delta)
        end
    end
end

local function children(a, b, o, random)
    local c, d = U.copy(a.z), U.copy(b.z)
    if random() < o.crossover then
        for j = 1, #c do
            if random() <= 0.5 and math.abs(a.z[j] - b.z[j]) > 1e-14 then
                local low, high = math.min(a.z[j], b.z[j]), math.max(a.z[j], b.z[j])
                local r, power = random(), 1 / (o.crossover_eta + 1)

                local function beta_q(distance)
                    local beta = 1 + 2 * distance / (high - low)
                    local alpha = 2 - beta ^( -(o.crossover_eta + 1))
                    if r <= 1 / alpha then
                        return (r * alpha) ^ power
                    end
                    return (1 / (2 - r * alpha)) ^ power
                end
                c[j] = clip(0.5 * (low + high - beta_q(low) * (high - low)))
                d[j] = clip(0.5 * (low + high + beta_q(1 - high) * (high - low)))
                if random() < 0.5 then
                    c[j], d[j] = d[j], c[j]
                end
            end
        end
    end
    mutate(c, o.mutation, o.mutation_eta, random)
    mutate(d, o.mutation, o.mutation_eta, random)
    return c, d
end

function M.run(self, algorithm)
    assert(algorithm == 'GA' or algorithm == 'PSO', 'unknown Pareto population method')
    local o = U.options(self, { seed = 1, population = 20, mutation_eta = 20 })
    local d = U.domain(self, o)
    local n = #d.active
    local size = U.number(o, 'population', 2, math.huge, true)
    if o.mutation == nil then
        o.mutation = 1 / math.max(1, n)
    end
    U.number(o, 'mutation', 0, 1);
    U.number(o, 'mutation_eta', 0, 1e6)
    if algorithm == 'GA' then
        if o.crossover == nil then
            o.crossover = 0.9
        end
        if o.crossover_eta == nil then
            o.crossover_eta = 15
        end
        U.number(o, 'crossover', 0, 1);
        U.number(o, 'crossover_eta', 0, 1e6)
    else
        local defaults = {
            inertia = 0.7298,
            cognitive = 1.49618,
            social = 1.49618,
            velocity_limit = 0.2,
            archive_size = 100
        }
        for k, v in pairs(defaults) do
            if o[k] == nil then
                o[k] = v
            end
        end
        U.number(o, 'inertia', 0, 1);
        U.number(o, 'cognitive', 0, 1e6);
        U.number(o, 'social', 0, 1e6)
        U.number(o, 'velocity_limit', 1e-12, 1)
        U.number(o, 'archive_size', 2, math.huge, true)
    end
    local random = U.generator(o.seed)
    local lower, width = {}, {}
    for k, j in ipairs(d.active) do
        assert(
            o.lower and o.upper and U.finite(o.lower[j]) and U.finite(o.upper[j]),
            'multiobjective ' .. algorithm .. ' requires finite bounds on every active variable'
        )
        lower[k], width[k] = o.lower[j], o.upper[j] - o.lower[j]
        assert(U.finite(width[k]) and width[k] > 0, 'invalid multiobjective interval width')
    end

    local function physical(z)
        local x = U.copy(self._start)
        for k, j in ipairs(d.active) do
            x[j] = math.max(o.lower[j], math.min(o.upper[j], lower[k] + width[k] * clip(z[k])))
        end
        return x
    end

    local function normalized(x)
        local z = {};
        for k, j in ipairs(d.active) do
            z[k] = (x[j] - lower[k]) / width[k]
        end
        return z
    end
    local generation = 0

    local function evaluate(z, phase, index)
        if not d.available() then
            return nil
        end
        self._request_info = {
            phase = phase,
            generation = generation,
            iteration = generation,
            candidate_index = index,
            population_size = size
        }
        local x = (generation == 0 and index == 1) and U.copy(self._start) or physical(z)
        local values = self:f(x) -- Public API validates and archives it.
        return { z = U.copy(z), objectives = values, id = self:ncalls() }
    end
    local population = {}
    for i = 1, size do
        local z = normalized(self._start)
        if i > 1 then
            for j = 1, n do
                z[j] = random()
            end
        end
        local member = evaluate(z, 'initial_population', i)
        if not member then
            return
        end
        population[#population + 1] = member
        if n == 0 then
            self._radius = 0;
            self._stop_reason = 'fixed_variables';
            return
        end
    end
    if algorithm == 'GA' then
        M.rank(population)
        -- NSGA-II competes parents and offspring together; crowding preserves
        -- spread when only part of the last non-dominated layer can survive.
        while d.available() do
            generation = generation + 1
            local combined = {}
            for i, p in ipairs(population) do
                combined[i] = p
            end
            local count = 0
            while count < size and d.available() do
                local a = tournament(population, random, true)
                local b = tournament(population, random, true)
                local c, e = children(a, b, o, random)
                for _, z in ipairs { c, e } do
                    if count == size or not d.available() then
                        break
                    end
                    count = count + 1
                    combined[#combined + 1] = evaluate(z, 'nsga2_offspring', count)
                end
            end
            population = environmental_selection(combined, size)
        end
    else
        for _, p in ipairs(population) do
            p.best = U.copy(p.z);
            p.best_objectives = U.copy(p.objectives);
            p.velocity = {}
            for j = 1, n do
                p.velocity[j] = 0
            end
        end
        while d.available() do
            generation = generation + 1
            -- Freeze leaders for a generation. Rebuild from the full observed
            -- front, so a discarded working leader never loses its public record.
            local leaders = {}
            for i, p in ipairs(self._pareto) do
                leaders[i] = { z = normalized(p.x), objectives = p.objectives, id = i }
            end
            M.crowding(leaders)
            while #leaders > o.archive_size do
                table.sort(leaders, crowded_first);
                table.remove(leaders)
                M.crowding(leaders)
            end
            for i, p in ipairs(population) do
                if not d.available() then
                    break
                end
                local leader = tournament(leaders, random, false)
                local z = U.copy(p.z)
                for j = 1, n do
                    local v = o.inertia * p.velocity[j]
                        + o.cognitive * random() * (p.best[j] - p.z[j])
                        + o.social * random() * (leader.z[j] - p.z[j])
                    v = math.max(-o.velocity_limit, math.min(o.velocity_limit, v))
                    local proposal = p.z[j] + v
                    z[j] = clip(proposal)
                    p.velocity[j] = proposal ~= z[j] and -0.5 * v or v
                end
                local before = U.copy(z)
                mutate(z, o.mutation, o.mutation_eta, random)
                for j = 1, n do
                    if z[j] ~= before[j] then
                        p.velocity[j] = 0
                    end
                end
                local candidate = evaluate(z, 'mopso_move', i)
                p.z, p.objectives = candidate.z, candidate.objectives
                -- Strictly better replaces memory; mutually nondominated
                -- alternatives use an unbiased tie. Exact equality keeps memory.
                if
                    M.dominates(p.objectives, p.best_objectives)
                    or (
                        not M.dominates(p.best_objectives, p.objectives)
                        and not equal(p.objectives, p.best_objectives)
                        and random() < 0.5
                    )
                then
                    p.best, p.best_objectives = U.copy(p.z), U.copy(p.objectives)
                end
            end
        end
    end
    self._stop_reason = 'budget'
end
return M
