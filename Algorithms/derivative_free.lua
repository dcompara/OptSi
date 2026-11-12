-- Scalar minimizers with the same values()/result() interface as SPSA/GA.
-- Every measured candidate goes through self:f(), including initialization.
-- Pattern: opportunistic bounded compass search, with rotated polling order.
-- PSO: synchronous global-best inertia PSO, private RNG and velocity cap.
-- Annealing: Gaussian proposals, Metropolis acceptance, geometric cooling.
local U = require 'optimizer_utils'
local M = {}

local function pattern(self)
    local o = U.options(self, { shrink = 0.5, expansion = 1, opportunistic = true })
    U.number(o, 'shrink', 1e-12, 1 - 1e-12)
    U.number(o, 'expansion', 1, math.huge)
    assert(type(o.opportunistic) == 'boolean', 'opportunistic must be a boolean')
    local d = U.domain(self, o)
    local x = U.copy(d.zero)
    self._radius = d.max_step
    local f = d.evaluate(x, { phase = 'initial', iteration = 0, search_scale = 1 })
    if not f or #d.active == 0 then
        self._radius = 0;
        return
    end
    -- Pattern radius is mesh times the largest physical characteristic step.
    local mesh, iteration, first = 1, 0, 1
    while d.available() and mesh * d.max_step >= self._minradius do
        iteration = iteration + 1
        local best_x, best_f, best_slot = x, f, nil
        for k = 0, #d.active - 1 do
            local slot = (first + k - 1) % #d.active + 1
            local j = d.active[slot]
            for _, sign in ipairs { 1, -1 } do
                local trial = U.copy(x)
                trial[j] = d.clip(x[j] + sign * mesh, j)
                if trial[j] ~= x[j] then
                    local value = d.evaluate(
                        trial,
                        {
                            phase = 'pattern_poll',
                            iteration = iteration,
                            search_scale = mesh,
                            center_cost_before = f,
                            direction = sign * j
                        }
                    )
                    if value == nil then
                        return
                    end
                    if value < best_f then
                        best_x, best_f, best_slot = trial, value, slot
                    end
                    if best_slot and o.opportunistic then
                        break
                    end
                end
            end
            if best_slot and o.opportunistic then
                break
            end
        end
        if best_slot then
            x, f = best_x, best_f
            first = best_slot % #d.active + 1
            mesh = mesh * o.expansion
        else
            mesh = mesh * o.shrink
        end
        self._radius = mesh * d.max_step
    end
end

local function pso(self)
    local o = U.options(
        self,
        {
            seed = 1,
            initial_spread = 2,
            inertia = 0.7298,
            cognitive = 1.49618,
            social = 1.49618,
            velocity_limit = 4
        }
    )
    local d = U.domain(self, o)
    if o.population == nil then
        o.population = math.max(8, math.min(24, 2 * #d.active))
    end
    local size = U.number(o, 'population', 2, math.huge, true)
    U.number(o, 'initial_spread', 1e-12, math.huge)
    U.number(o, 'inertia', 0, 1)
    U.number(o, 'cognitive', 0, math.huge);
    U.number(o, 'social', 0, math.huge)
    U.number(o, 'velocity_limit', 1e-12, math.huge)
    local random = U.generator(o.seed)
    local swarm, global, global_f = {}, nil, math.huge
    local generation, population_radius = 0, 0

    local function refresh()
        local center = {}
        for j = 1, #d.zero do
            center[j] = 0
        end
        for _, p in ipairs(swarm) do
            for j = 1, #d.zero do
                center[j] = center[j] + p.x[j] / #swarm
            end
        end
        -- Both spread about the swarm centroid and velocity use caller-coordinate
        -- scales. The stop radius is the larger of their maximum Euclidean norms.
        local radius2, speed2 = 0, 0
        for _, p in ipairs(swarm) do
            local r, v = 0, 0
            for j = 1, #d.zero do
                r = r + ((p.x[j] - center[j]) * d.scale[j]) ^ 2
                v = v + (p.velocity[j] * d.scale[j]) ^ 2
            end
            radius2, speed2 = math.max(radius2, r), math.max(speed2, v)
        end
        population_radius = math.sqrt(radius2)
        -- A collapsed swarm with moving particles is not yet stationary.
        self._radius = math.sqrt(math.max(radius2, speed2))
    end

    local function info(phase, i)
        return {
            phase = phase,
            generation = generation,
            iteration = generation,
            population_size = size,
            candidate_index = i,
            population_radius = population_radius
        }
    end
    for i = 1, size do
        local x = U.copy(d.zero)
        if i > 1 then
            for _, j in ipairs(d.active) do
                x[j] = d.reflect(o.initial_spread * (2 * random() - 1), j)
            end
        end
        local f = d.evaluate(x, info('initial_population', i))
        if f == nil then
            return
        end
        swarm[i] = { x = x, f = f, best = U.copy(x), best_f = f, velocity = U.copy(d.zero) }
        if not global or f < global_f then
            global, global_f = U.copy(x), f
        end
        refresh()
        if #d.active == 0 then
            return
        end
    end
    while d.available() and self._radius >= self._minradius do
        generation = generation + 1
        local guide = U.copy(global) -- Fixed until every particle has been moved.
        for i, p in ipairs(swarm) do
            local trial = U.copy(p.x)
            for _, j in ipairs(d.active) do
                local v = o.inertia * p.velocity[j]
                    + o.cognitive * random() * (p.best[j] - p.x[j])
                    + o.social * random() * (guide[j] - p.x[j])
                v = math.max(-o.velocity_limit, math.min(o.velocity_limit, v))
                local proposed = p.x[j] + v
                trial[j] = d.reflect(proposed, j)
                -- Reset only on a genuine bound crossing. Interior reflection
                -- roundoff, or a proposal exactly on a bound, must retain velocity.
                p.velocity[j] = (proposed < d.lower[j] or proposed > d.upper[j]) and 0 or v
            end
            local f = d.evaluate(trial, info('pso_move', i))
            if f == nil then
                return
            end
            p.x, p.f = trial, f
            if f < p.best_f then
                p.best, p.best_f = U.copy(trial), f
            end
            if f < global_f then
                global, global_f = U.copy(trial), f
            end
            refresh()
        end
    end
end

local function annealing(self)
    local o = U.options(
        self,
        {
            seed = 1,
            temperature = 0.05,
            cooling = 0.98,
            temperature_min = 1e-6,
            proposal_scale = 2,
            min_proposal_scale = 0.05,
            reheat_after = 0,
            max_reheats = 2
        }
    )
    U.number(o, 'temperature', 1e-300, math.huge)
    U.number(o, 'temperature_min', 0, o.temperature)
    U.number(o, 'cooling', 1e-12, 1)
    U.number(o, 'proposal_scale', 1e-12, math.huge)
    U.number(o, 'min_proposal_scale', 0, o.proposal_scale)
    U.number(o, 'reheat_after', 0, math.huge, true)
    U.number(o, 'max_reheats', 0, math.huge, true)
    local random, normal = U.generator(o.seed)
    local d = U.domain(self, o)
    local x = U.copy(d.zero)
    local f = d.evaluate(
        x,
        {
            phase = 'initial',
            iteration = 0,
            temperature = o.temperature,
            search_scale = o.proposal_scale,
            reheat_id = 0
        }
    )
    if f == nil or #d.active == 0 then
        self._radius = 0;
        return
    end
    local best, stall, age, iteration, reheats = f, 0, 0, 0, 0
    local accepted_previous
    while d.available() do
        if o.reheat_after > 0 and stall >= o.reheat_after and reheats < o.max_reheats then
            age, stall, reheats = 0, 0, reheats + 1
        end
        local temperature = math.max(o.temperature_min, o.temperature * o.cooling ^ age)
        local scale = math.max(
            o.min_proposal_scale,
            o.proposal_scale * math.sqrt(temperature / o.temperature)
        )
        -- Annealing uses a temperature-dependent proposal scale, not a poll mesh.
        self._radius = scale * d.max_step / math.sqrt(#d.active)
        if self._radius < self._minradius then
            return
        end
        local trial = U.copy(x)
        for _, j in ipairs(d.active) do
            trial[j] = d.reflect(x[j] + scale / math.sqrt(#d.active) * normal(), j)
        end
        iteration = iteration + 1
        local value = d.evaluate(
            trial,
            {
                phase = 'annealing_trial',
                iteration = iteration,
                temperature = temperature,
                search_scale = scale,
                reheat_id = reheats,
                center_cost_before = f,
                accepted_previous = accepted_previous
            }
        )
        if value == nil then
            return
        end
        local accept = false
        if U.finite(value) then
            accept = value <= f or (
                    temperature > 0 and math.log(random()) < -(value - f) / temperature
                )
        end
        if accept then
            x, f = trial, value
        end
        accepted_previous = accept and 1 or 0
        if value < best then
            best, stall = value, 0
        else
            stall = stall + 1
        end
        age = age + 1
    end
end

function M.run(self, variant)
    if variant == 'pattern' then
        return pattern(self)
    elseif variant == 'pso' then
        return pso(self)
    elseif variant == 'annealing' then
        return annealing(self)
    end
    error('unknown derivative-free variant: ' .. tostring(variant))
end
return M
