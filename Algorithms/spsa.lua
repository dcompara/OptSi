-- SPSA with directional curvature and an Armijo line search.
-- Coordinates are scaled by step. Every objective request uses the same budget.
-- Optional stagnation recovery: smaller probes first, then bounded random restarts.
local unpack = unpack or table.unpack

local function copy(x)
    return { unpack(x) }
end

local function same(a, b)
    for i = 1, #a do
        if a[i] ~= b[i] then
            return false
        end
    end
    return true
end

local function new(o)
    local U = require 'optimizer_utils'
    assert(
        type(o.start) == 'table'
            and #o.start > 0
            and type(o.step) == 'table'
            and #o.start == #o.step,
        'SPSA requires matching start and step vectors'
    )
    local opt, count = {}, 0
    local best, best_value = nil, math.huge
    local c, budget = o.c_iteration or 0.2, o.maxcalls or math.huge
    local invalid = o.invalid_cost or math.huge
    local random = U.generator(o.seed or 1)
    assert(U.finite(c) and c > 0, 'SPSA c_iteration must be positive')
    assert(budget > 0 and budget == math.floor(budget), 'SPSA maxcalls must be a positive integer')
    local info, iteration, restarts, centre_value = {}, 0, 0, nil
    local last_progress, milestone, reduced = 0, nil, false
    local patience, tol = o.stall_calls or 30, o.improvement_tol or 1e-4
    local c_initial, c_min = c, math.min(c, o.c_min or 0.025)
    local scale, active = {}, 0
    for i = 1, #o.start do
        assert(U.finite(o.start[i]) and U.finite(o.step[i]), 'SPSA start/step must be finite')
        scale[i] = math.abs(o.step[i])
        if o.lower and o.upper and o.lower[i] == o.upper[i] then
            scale[i] = 0
        end
        if scale[i] > 0 then
            active = active + 1
        end
    end

    local function valid(f)
        return type(f) == 'number' and f == f and f > -math.huge and f < invalid
    end

    local function shifted(x, d, a)
        local p = {}
        for i = 1, #x do
            p[i] = x[i] + scale[i] * (a * d[i])
        end
        return p
    end

    local function sign()
        return random() > 0.5 and 1 or -1
    end

    local function roundoff(f0, fp, fm)
        return 16
            * 2.220446049250313e-16
            * math.max(
                math.abs(f0),
                valid(fp) and math.abs(fp) or 0,
                valid(fm) and math.abs(fm) or 0
            )
    end

    local function evaluate(x, phase)
        if count >= budget then
            return nil
        end
        info = {
            phase = phase,
            iteration = iteration,
            restart_id = restarts,
            probe_scale = c,
            center_cost_before = centre_value
        }
        local f = coroutine.yield(copy(x))
        -- Local count advances on feedback. The outer evaluator counts requests
        -- and can reject symmetric out-of-bound probes without calling the model.
        count = count + 1
        if valid(f) and f < best_value then
            best, best_value = copy(x), f
        end
        if best and (not milestone or milestone - best_value >= tol) then
            milestone, last_progress, reduced = best_value, count, false
        end
        return f
    end
    local co = coroutine.wrap(function()
        local x = copy(o.start)
        local f0 = evaluate(x, 'initial')
        if not valid(f0) then
            return
        end -- A valid initial objective value is required.
        local radius = 1
        while active > 0 and count < budget do
            centre_value = f0
            if o.restart and count - last_progress >= patience then
                -- Reuse a measured point without spending another evaluation.
                x, f0 = copy(best), best_value
                radius = 1
                centre_value = f0
                if not reduced or restarts >= (o.max_restarts or 3) then
                    c = math.max(c_min, 0.5 * c)
                    reduced = true
                elseif restarts < (o.max_restarts or 3) and budget - count >= 4 then
                    restarts = restarts + 1
                    local amplitude = math.min((o.restart_scale or 2) * 1.5 ^ (restarts - 1), 4)
                    local p = copy(best)
                    for i = 1, #p do
                        if scale[i] > 0 then
                            p[i] = p[i] + scale[i] * amplitude * sign() * (0.5 + 0.5 * random())
                            -- Reflect restart points into bounds. Probes keep their symmetry.
                            if o.lower and o.upper then
                                local lo, hi = o.lower[i], o.upper[i]
                                local width = hi - lo
                                if width > 0 then
                                    local q = (p[i] - lo) % (2 * width)
                                    p[i] = lo + (q <= width and q or 2 * width - q)
                                else
                                    p[i] = lo
                                end
                            end
                        end
                    end
                    c = c_initial
                    local f = evaluate(p, 'restart')
                    if f == nil then
                        break
                    end
                    -- Accept a valid worse centre deliberately; the best archive is retained.
                    if valid(f) then
                        x, f0 = p, f
                    end
                    reduced = false
                end
                last_progress = count
            end
            centre_value = f0
            iteration = iteration + 1
            local delta = {}
            for i = 1, #x do
                delta[i] = scale[i] > 0 and sign() or 0
            end
            local plus, minus = shifted(x, delta, c), shifted(x, delta, -c)
            local fp = evaluate(plus, 'probe_plus')
            if fp == nil then
                break
            end
            local fm = evaluate(minus, 'probe_minus')
            if fm == nil then
                break
            end
            local both = valid(fp) and valid(fm)
            local numerator, denominator
            if both then
                numerator, denominator = fp - fm, 2 * c
            elseif valid(fp) then
                numerator, denominator = fp - f0, c
            elseif valid(fm) then
                numerator, denominator = f0 - fm, c
            end
            local eps = roundoff(f0, fp, fm)
            if numerator and math.abs(numerator) > eps then
                local slope = numerator / denominator
                local norm = math.abs(slope) * math.sqrt(active)
                local alpha = radius / norm
                if both then
                    local second = (fp - f0) + (fm - f0)
                    if second > 2 * eps then
                        alpha = c * c / second
                    end -- Line-Newton gain, 1/K.
                end
                alpha = math.min(alpha, 2 / norm) -- Maximum scaled displacement norm = 2.
                local gradient = {}
                for i = 1, #x do
                    gradient[i] = slope * delta[i]
                end
                local accepted = false
                for trial = 1, 8 do
                    local p = shifted(x, gradient, -alpha)
                    if same(p, x) then
                        break
                    end
                    local f
                    if same(p, plus) then
                        f = fp
                    elseif same(p, minus) then
                        f = fm
                    else
                        f = evaluate(p, 'line_search')
                    end
                    if f == nil then
                        break
                    end
                    -- The directional decrease is alpha*slope^2, not alpha*||g||^2.
                    if valid(f) and f < f0 and f0 - f >= 1e-4 * alpha * slope * slope then
                        x, f0 = p, f
                        radius = math.min(2 * alpha * norm, 2)
                        accepted = true
                        break
                    end
                    alpha = 0.5 * alpha
                end
                if not accepted then
                    radius = alpha * norm
                end
            end -- An invalid/flat pair just requests another random direction.
        end
    end)
    local pending = co()

    function opt:running()
        return pending ~= nil
    end

    function opt:values()
        return unpack(pending or best or o.start)
    end

    function opt:result(f)
        pending = co(f)
    end

    function opt:ncalls()
        return count
    end

    function opt:best()
        return best and copy(best), best_value
    end

    function opt:info()
        return info
    end
    return opt
end

-- Adapter for Optimization_Algorithms.lua: same self:f / _start / _step interface.
-- SPSA stops by its request budget or an unusable initial value/all-fixed inputs;
-- minradius is not a stopping test in this adapter. Probes retain their symmetry.
local function run(self)
    -- Method defaults live here, like those of the other optimizers. A driver
    -- supplies only model-specific overrides (for example an invalid-cost cutoff).
    local options = require('optimizer_utils').options(
        self,
        {
            seed = 1,
            c_iteration = 0.2,
            invalid_cost = math.huge,
            restart = false,
            stall_calls = 30,
            improvement_tol = 1e-4,
            c_min = 0.025,
            restart_scale = 2,
            max_restarts = 3
        }
    )
    options.start, options.step = self._start, self._step
    options.maxcalls = self._maxcalls and (self._maxcalls - self:ncalls()) or math.huge
    self._resolved_options = options
    local opt = new(options)
    while opt:running() do
        self._request_info = opt:info()
        opt:result(self:f({ opt:values() }))
    end
    self._best_point, self._best_value = opt:best()
end
return { run = run }
