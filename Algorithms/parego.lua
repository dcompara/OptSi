-- ParEGO for deterministic, expensive vector objectives, all minimized.
-- Knowles (2006), IEEE Transactions on Evolutionary Computation 10(1):50-66.
-- https://staff.cs.manchester.ac.uk/~jknowles/parego/
-- Latin hypercube initialization; changing simplex-lattice weights; augmented
-- Tchebycheff scalarization of normalized observations; refitted GP and EI.
-- This compact version uses an ARD squared-exponential GP and multistart
-- bounded pattern optimization of EI, rather than the paper's inner GA.
-- Budget counts expensive vector evaluations, never GP/acquisition calls.
-- The default initial design has 5*n+1 points for n active parameters.
-- Set options.initial_points below maxcalls to leave evaluations for GP/EI
-- search. Every observation is rescalarized as the weights and objective
-- ranges change. Finite bounds and finite objective vectors are required.
-- Like NSGA-II and MOPSO, the public evaluator returns the complete vector.
-- Scalarization is internal: changing weights changes the search preference,
-- while the public Pareto archive retains the original measured objectives.
local U = require 'optimizer_utils'
local GP = require 'surrogate'
local M = {}

local function shuffle(a, random)
    for i = #a, 2, -1 do
        local j = math.floor(random() * i) + 1;
        a[i], a[j] = a[j], a[i]
    end
end

local function distance(a, b)
    local value = 0
    for j = 1, #a do
        value = value + (a[j] - b[j]) ^ 2
    end
    return value
end

local function reflect(value)
    local t = value % 2
    return t <= 1 and t or 2 - t
end

-- Exact simplex lattice, traversed in shuffled cycles to cover tradeoffs.
local function weight_lattice(objectives, divisions)
    local list, current = {}, {}

    local function fill(k, left)
        if k == objectives then
            current[k] = left / divisions
            list[#list + 1] = U.copy(current)
            assert(#list <= 4096, 'too many ParEGO weights; reduce weight_divisions')
        else
            for value = 0, left do
                current[k] = value / divisions;
                fill(k + 1, left - value)
            end
        end
    end
    fill(1, divisions)
    return list
end

function M.run(self)
    local o = U.options(
        self,
        {
            seed = 1,
            rho = 0.05,
            gp_nugget = 1e-9,
            gp_sweeps = 2,
            gp_refit_interval = 1,
            acquisition_starts = 4,
            acquisition_iterations = 12
        }
    )
    local d = U.domain(self, o)
    local n = #d.active
    local budget = self._maxcalls
    assert(
        U.finite(budget) and budget >= 0 and budget == math.floor(budget),
        'ParEGO requires a finite integer maxcalls budget'
    )
    U.number(o, 'rho', 1e-12, 1)
    U.number(o, 'gp_nugget', 1e-15, 0.1)
    U.number(o, 'gp_sweeps', 0, 20, true)
    U.number(o, 'gp_refit_interval', 1, 1000000, true)
    if o.gp_max_points ~= nil then
        U.number(o, 'gp_max_points', 3, 1000000, true)
    end
    U.number(o, 'acquisition_starts', 1, 100, true)
    U.number(o, 'acquisition_iterations', 1, 100, true)
    if o.initial_points == nil then
        o.initial_points = 5 * n + 1
    end
    if o.candidates == nil then
        o.candidates = math.max(64, 64 * n)
    end
    U.number(o, 'initial_points', 1, math.huge, true)
    U.number(o, 'candidates', 8, 100000, true)
    local random, normal = U.generator(o.seed)
    local lo, span = {}, {}
    self._radius = 0
    for i, j in ipairs(d.active) do
        assert(
            o.lower and o.upper and U.finite(o.lower[j]) and U.finite(o.upper[j]),
            'ParEGO requires finite lower and upper bounds on every active variable'
        )
        lo[i], span[i] = o.lower[j], o.upper[j] - o.lower[j]
        assert(U.finite(span[i]) and span[i] > 0, 'invalid ParEGO interval width')
        self._radius = math.max(self._radius, span[i])
    end
    local archive, front, points = {}, {}, {}
    self._parego_archive, self._pareto = archive, front
    local dimensions

    local function physical(z)
        local x = U.copy(self._start)
        for i, j in ipairs(d.active) do
            local value = lo[i] + span[i] * math.max(0, math.min(1, z[i]))
            -- Reconstruction can round past a physical endpoint even when
            -- z is in [0,1]. Preserve the public evaluator's exact bounds.
            x[j] = math.max(o.lower[j], math.min(o.upper[j], value))
        end
        return x
    end

    local function evaluate(z, info)
        if not d.available() then
            return false
        end
        local x = physical(z)
        self._request_info = info
        local values = self:f(x)
        assert(type(values) == 'table' and #values >= 2, 'ParEGO expects at least two objectives')
        dimensions = dimensions or #values
        assert(#values == dimensions, 'the number of ParEGO objectives changed')
        for j = 1, dimensions do
            assert(U.finite(values[j]), 'ParEGO objectives must be finite numbers')
        end
        local entry = { x = U.copy(x), objectives = U.copy(values) }
        archive[#archive + 1] = entry
        points[#points + 1] = U.copy(z)
        -- self:f() has already updated the shared public Pareto archive.
        return true
    end

    local function training_subset(all_points, all_values, max_points, best_index)
        local total = #all_points
        if not max_points or total <= max_points then
            return all_points, all_values, total
        end
        local indexes, seen = {}, {}

        local function add(index)
            index = math.max(1, math.min(total, math.floor(index + 0.5)))
            if not seen[index] then
                seen[index] = true;
                indexes[#indexes + 1] = index
            end
        end
        -- Always keep the current scalar best, the newest observation and the
        -- beginning of the design, then fill the remaining slots uniformly in
        -- observation order. This is deterministic, cheap, and preserves the
        -- full archive separately for Pareto reporting and rescalarization.
        add(best_index);
        add(total);
        add(1)
        if max_points > 1 then
            for k = 0, max_points - 1 do
                if #indexes >= max_points then
                    break
                end
                add(1 + k * (total - 1) / (max_points - 1))
            end
        end
        if #indexes < max_points then
            for index = total, 1, -1 do
                add(index)
                if #indexes >= max_points then
                    break
                end
            end
        end
        table.sort(indexes)
        local subset_points, subset_values = {}, {}
        for k = 1, #indexes do
            local index = indexes[k]
            subset_points[k] = all_points[index]
            subset_values[k] = all_values[index]
        end
        return subset_points, subset_values, #subset_points
    end

    local start = {}
    for i, j in ipairs(d.active) do
        start[i] = (self._start[j] - lo[i]) / span[i]
    end
    if not evaluate(start, { phase = 'initial', iteration = 0 }) then
        return
    end
    if n == 0 then
        self._stop_reason = 'fixed_variables';
        return
    end

    -- Start counts as one initialization point. The remaining design is a LHS.
    local count = math.min(o.initial_points - 1, math.floor(budget - self:ncalls()))
    if count > 0 then
        local design = {}
        for k = 1, count do
            design[k] = {}
        end
        for j = 1, n do
            local bins = {}
            for k = 1, count do
                bins[k] = k
            end
            shuffle(bins, random)
            for k = 1, count do
                design[k][j] = (bins[k] - 1 + random()) / count
            end
        end
        for k = 1, count do
            evaluate(design[k], { phase = 'parego_design', iteration = 0, member = k })
        end
    end
    if not d.available() then
        self._stop_reason = 'budget';
        return
    end
    if o.weight_divisions == nil then
        o.weight_divisions = dimensions == 2 and 10 or (dimensions == 3 and 4 or 2)
    end
    U.number(o, 'weight_divisions', 1, 100, true)
    local weights = weight_lattice(dimensions, o.weight_divisions)
    shuffle(weights, random)
    local weight_index, iteration, warm_length = 1, 0, nil
    while d.available() do
        iteration = iteration + 1
        local weight = weights[weight_index]
        weight_index = weight_index + 1
        if weight_index > #weights then
            weight_index = 1;
            shuffle(weights, random)
        end
        local ideal, nadir = {}, {}
        for j = 1, dimensions do
            ideal[j], nadir[j] = math.huge, -math.huge
        end
        for _, entry in ipairs(archive) do
            for j = 1, dimensions do
                ideal[j] = math.min(ideal[j], entry.objectives[j])
                nadir[j] = math.max(nadir[j], entry.objectives[j])
            end
        end
        local scalar, best, best_index = {}, math.huge, 1
        for k, entry in ipairs(archive) do
            local largest, sum = -math.huge, 0
            for j = 1, dimensions do
                local range = nadir[j] - ideal[j]
                local normalized = range > 0 and (entry.objectives[j] - ideal[j]) / range or 0
                local weighted = weight[j] * normalized
                largest, sum = math.max(largest, weighted), sum + weighted
            end
            scalar[k] = largest + o.rho * sum
            if scalar[k] < best then
                best, best_index = scalar[k], k
            end
        end
        -- A training limit reduces only the GP input, never the measured archive.
        local fit_points, fit_scalar, training_count =
            training_subset(points, scalar, o.gp_max_points, best_index)
        local refit = not warm_length or ((iteration - 1) % o.gp_refit_interval == 0)
        local model = GP.fit(
            fit_points,
            fit_scalar,
            {
                nugget = o.gp_nugget,
                sweeps = refit and o.gp_sweeps or 0,
                length = warm_length,
                fixed_length = warm_length ~= nil and not refit
            }
        )
        if model then
            warm_length = model.length
        end

        local function nearest(z)
            local v = math.huge
            for _, p in ipairs(points) do
                v = math.min(v, distance(z, p))
            end
            return v
        end

        -- These surrogate predictions do not call the expensive objective.
        local function acquisition(z)
            if nearest(z) < 1e-14 then
                return -1
            end
            if not model then
                return 0
            end
            local mean, variance = model:predict(z)
            return GP.expected_improvement(mean, variance, best)
        end
        local candidates, farthest, far_distance = {}, nil, -1

        local function consider(z)
            local separation = nearest(z)
            if separation > far_distance then
                farthest, far_distance = U.copy(z), separation
            end
            candidates[#candidates + 1] = { x = z, value = acquisition(z), index = #candidates + 1 }
        end
        for k = 1, o.candidates do
            local z = {}
            if n == 1 then
                z[1] = (k - 1) / (o.candidates - 1)
            elseif k % 2 == 0 then
                for j = 1, n do
                    z[j] = random()
                end
            else
                local base = k % 4 == 1 and points[best_index]
                    or points[math.floor(random() * #points) + 1]
                for j = 1, n do
                    z[j] = reflect(base[j] + 0.2 * normal())
                end
            end
            consider(z)
        end
        -- End points along each axis are useful when a weight selects a single
        -- objective; every acquisition candidate is cheap, even at a boundary.
        for j = 1, n do
            for _, bound in ipairs { 0, 1 } do
                local z = U.copy(points[best_index]);
                z[j] = bound;
                consider(z)
            end
        end
        table.sort(candidates, function(a, b)
            return a.value > b.value or (a.value == b.value and a.index < b.index)
        end)
        local proposal, maximum = U.copy(candidates[1].x), candidates[1].value
        for k = 1, math.min(o.acquisition_starts, #candidates) do
            local center, value = U.copy(candidates[k].x), candidates[k].value
            local mesh = 0.2
            for _ = 1, o.acquisition_iterations do
                local improved = false
                for j = 1, n do
                    for _, sign in ipairs { -1, 1 } do
                        local z = U.copy(center)
                        z[j] = math.max(0, math.min(1, z[j] + sign * mesh))
                        local score = acquisition(z)
                        if score > value then
                            center, value, improved = z, score, true
                        end
                    end
                end
                if not improved then
                    mesh = mesh * 0.5
                end
            end
            if value > maximum then
                proposal, maximum = center, value
            end
        end
        -- On a constant or numerically flat scalarization, EI can be zero.
        -- Space filling remains well-defined and avoids re-evaluating a point.
        if maximum <= 1e-15 then
            proposal = farthest
        end
        if not proposal or nearest(proposal) < 1e-14 then
            self._stop_reason = 'design_resolution';
            break
        end
        -- Retain the search preference/model diagnostics alongside each new request.
        self._parego_state = {
            iteration = iteration,
            weights = U.copy(weight),
            length_scales = warm_length and U.copy(warm_length),
            expected_improvement = math.max(0, maximum),
            gp_training_points = training_count,
            gp_hyperparameter_refit = refit
        }
        evaluate(
            proposal,
            {
                phase = 'parego_ei',
                iteration = iteration,
                weights = U.copy(weight),
                expected_improvement = math.max(0, maximum),
                gp_training_points = training_count,
                gp_hyperparameter_refit = refit and 1 or 0
            }
        )
    end
    if not self._stop_reason then
        self._stop_reason = 'budget'
    end
end

return M
