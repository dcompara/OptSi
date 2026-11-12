-- Lua 5.1: projected gradient descent (optionally BB-initialized Armijo), one inverse-BFGS minimizer,
-- and bounded scalar Brent minimization (NOT a zero finder).
-- Coordinates are normalized by abs(step), as in optimizer_utils.domain.
-- Brent uses safeguarded parabolic interpolation and golden-section steps.
-- Reference: R. P. Brent, Algorithms for Minimization Without Derivatives
-- (1973), chapter 5; https://www.netlib.org/fmm/fmin.f
local U = require 'optimizer_utils'
local M = {}
local abs, sqrt = math.abs, math.sqrt

local function dot(a, b)
    local s = 0;
    for i = 1, #a do
        s = s + a[i] * b[i]
    end;
    return s
end

local function identity(n)
    local H = {};
    for i = 1, n do
        H[i] = {};
        for j = 1, n do
            H[i][j] = i == j and 1 or 0
        end
    end
    return H
end

local function stop(self, status)
    self._status = status
end

local function gradient_search(self, bfgs)
    local o = U.options(
        self,
        {
            fd_step = 1e-5,
            difference = 'central',
            gradient_tolerance = 1e-7,
            initial_alpha = 1,
            armijo_c1 = 1e-4,
            backtrack = 0.5,
            min_alpha = 1e-14,
            curvature_tolerance = 1e-10,
            max_iterations = 1000,
            max_linesearch = 40,
            bb_variant = 'none'
        }
    )
    local fd_steps = U.fd_steps(o, #self._start)
    U.number(o, 'gradient_tolerance', 0, math.huge)
    U.number(o, 'initial_alpha', 1e-15, math.huge)
    U.number(o, 'armijo_c1', 1e-12, 0.49);
    U.number(o, 'backtrack', 1e-6, 0.999999)
    U.number(o, 'min_alpha', 1e-300, o.initial_alpha)
    U.number(o, 'curvature_tolerance', 0, 0.99)
    U.number(o, 'max_iterations', 1, math.huge, true)
    U.number(o, 'max_linesearch', 1, math.huge, true)
    assert(
        o.bb_variant == 'none'
            or o.bb_variant == 'long'
            or o.bb_variant == 'short'
            or o.bb_variant == 'alternate',
        'invalid optimizer option: bb_variant'
    )
    if bfgs then
        assert(o.bb_variant == 'none', 'bb_variant is supported only by Gradient_Descent')
    end
    local d = U.domain(self, o)
    local x = U.copy(d.zero)
    self._radius = d.max_step
    local f = d.evaluate(x, { phase = 'initial', iteration = 0 })
    if f == nil then
        return stop(self, 'budget')
    end
    if not U.finite(f) then
        return stop(self, 'nonfinite_initial_value')
    end
    if #d.active == 0 then
        self._radius = 0;
        return stop(self, 'fixed_variables')
    end
    local n, H = #x, identity(#x)

    -- Probe distances are normalized by abs(step); shortened/one-sided probes
    -- near bounds still spend the same global objective-request budget.
    local function gradient(z, fz, iteration)
        local g = U.copy(d.zero)
        for _, j in ipairs(d.active) do
            local center = d.physical(z)[j]
            local h = math.max(
                fd_steps[j] * math.max(1, abs(z[j])),
                8 * 2.220446049250313e-16 * abs(center) / d.scale[j]
            )
            local hp = math.min(h, d.upper[j] - z[j])
            local hm = math.min(h, z[j] - d.lower[j])
            local available_hp, available_hm = hp, hm
            -- Avoid an almost coincident probe when one bound is very near.
            if hp < h * 0.25 and hm >= h * 0.25 then
                hp = 0
            end
            if hm < h * 0.25 and hp >= h * 0.25 then
                hm = 0
            end
            local function probe(sign, distance)
                if distance <= 0 or z[j] + sign * distance == z[j] then
                    return false, 0
                end
                local p = U.copy(z);
                p[j] = z[j] + sign * distance
                distance = sign * (d.physical(p)[j] - center) / d.scale[j]
                if distance <= 0 then
                    return false, 0
                end
                return d.evaluate(p,
                    { phase = 'finite_difference', iteration = iteration, coordinate = j }), distance
            end
            local fp, fm
            if o.difference == 'central' then
                fp, hp = probe(1, hp)
                if fp == nil then return nil, 'budget' end
                fm, hm = probe(-1, hm)
                if fm == nil then return nil, 'budget' end
            elseif hp > 0 then
                fp, hp = probe(1, hp)
                if fp == nil then return nil, 'budget' end
                if not U.finite(fp) then
                    -- A failed forward probe still spends its request. Try the
                    -- other feasible side, even if the quarter-step rule hid it.
                    fm, hm = probe(-1, available_hm)
                    if fm == nil then return nil, 'budget' end
                end
            else
                fm, hm = probe(-1, hm)
                if fm == nil then return nil, 'budget' end
                if not U.finite(fm) then
                    fp, hp = probe(1, available_hp)
                    if fp == nil then
                        return nil, 'budget'
                    end
                end
            end
            if U.finite(fp) and U.finite(fm) then
                g[j] = (hm * (fp - fz) / hp + hp * (fz - fm) / hm) / (hp + hm)
            elseif U.finite(fp) then
                g[j] = (fp - fz) / hp
            elseif U.finite(fm) then
                g[j] = (fz - fm) / hm
            else
                return nil, 'finite_difference_failed'
            end
            if not U.finite(g[j]) then
                return nil, 'finite_difference_failed'
            end
        end
        return g
    end
    local g, status = gradient(x, f, 0)
    if not g then
        return stop(self, status)
    end
    local previous_s, previous_y

    local function initial_linesearch_alpha(iteration)
        if o.bb_variant == 'none' or not previous_s or not previous_y then
            return o.initial_alpha, 'none'
        end
        local ss, yy, sy =
            dot(previous_s, previous_s), dot(previous_y, previous_y), dot(previous_s, previous_y)
        local threshold = o.curvature_tolerance * sqrt(math.max(0, ss * yy))
        if
            not U.finite(ss)
            or not U.finite(yy)
            or not U.finite(sy)
            or ss <= 0
            or yy <= 0
            or sy <= threshold
        then
            return o.initial_alpha, 'fallback'
        end
        local variant = o.bb_variant
        if variant == 'alternate' then
            -- The first BB-eligible iteration uses the long (BB1) step.
            variant = (iteration % 2 == 0) and 'long' or 'short'
        end
        local alpha = variant == 'long' and ss / sy or sy / yy
        if not U.finite(alpha) or alpha <= 0 then
            return o.initial_alpha, 'fallback'
        end
        return math.max(o.min_alpha, alpha), variant
    end
    for iteration = 1, o.max_iterations do
        local pg = U.copy(g)
        for j = 1, n do
            if (x[j] <= d.lower[j] and g[j] > 0) or (x[j] >= d.upper[j] and g[j] < 0) then
                pg[j] = 0
            end
        end
        -- This is a projected-gradient test, distinct from the accepted-step test.
        if sqrt(dot(pg, pg)) <= o.gradient_tolerance then
            self._radius = 0;
            return stop(self, 'converged')
        end
        if not d.available() then
            return stop(self, 'budget')
        end
        local p = U.copy(d.zero)
        for _, i in ipairs(d.active) do
            if bfgs then
                for _, j in ipairs(d.active) do
                    p[i] = p[i] - H[i][j] * pg[j]
                end
            else
                p[i] = -pg[i]
            end
            if (x[i] <= d.lower[i] and p[i] < 0) or (x[i] >= d.upper[i] and p[i] > 0) then
                p[i] = 0
            end
        end
        if not U.finite(dot(g, p)) or dot(g, p) >= -1e-12 * sqrt(dot(pg, pg) * dot(p, p)) then
            H = identity(n);
            for j = 1, n do
                p[j] = -pg[j]
            end
        end
        local alpha, bb_used = initial_linesearch_alpha(iteration)
        local alpha_initial = alpha
        local trial, ft, s, accepted
        -- Every evaluated line-search trial counts, including rejected trials.
        for ls = 1, o.max_linesearch do
            trial, s = U.copy(x), U.copy(d.zero)
            local length2 = 0
            for _, j in ipairs(d.active) do
                trial[j] = d.clip(x[j] + alpha * p[j], j);
                s[j] = trial[j] - x[j]
                length2 = length2 + (s[j] * d.scale[j]) ^ 2
            end
            self._radius = sqrt(length2)
            local slope = dot(g, s)
            if slope < 0 and self._radius > 0 then
                ft = d.evaluate(
                    trial,
                    {
                        phase = 'armijo',
                        iteration = iteration,
                        alpha = alpha,
                        alpha_initial = alpha_initial,
                        bb_variant = bb_used,
                        center_cost_before = f,
                        directional_derivative = slope / alpha
                    }
                )
                if ft == nil then
                    return stop(self, 'budget')
                end
                if ft <= f + o.armijo_c1 * slope then
                    accepted = true;
                    break
                end
            end
            alpha = alpha * o.backtrack
            if alpha < o.min_alpha then
                break
            end
        end
        if not accepted then
            return stop(self, 'line_search_failed')
        end
        local oldg = g
        x, f = trial, ft
        if self._radius <= (self._minradius or 0) then
            return stop(self, 'step_tolerance')
        end
        g, status = gradient(x, f, iteration)
        if not g then
            return stop(self, status)
        end
        if not bfgs and o.bb_variant ~= 'none' then
            previous_s, previous_y = U.copy(s), {}
            for i = 1, n do
                previous_y[i] = g[i] - oldg[i]
            end
        end
        if bfgs then
            local y, Hy = {}, {}
            for i = 1, n do
                y[i] = g[i] - oldg[i]
            end
            local ys, ss, yy = dot(y, s), dot(s, s), dot(y, y)
            -- Armijo alone does not imply y^T s > 0. Skip unsafe updates.
            if U.finite(ys) and ys > 0 and ys > o.curvature_tolerance * sqrt(ss * yy) then
                for i = 1, n do
                    Hy[i] = 0;
                    for j = 1, n do
                        Hy[i] = Hy[i] + H[i][j] * y[j]
                    end
                end
                local factor = (1 + dot(y, Hy) / ys) / ys
                local valid = true
                for i = 1, n do
                    for j = i, n do
                        local value = H[i][j]
                            + factor * s[i] * s[j]
                            - (Hy[i] * s[j] + s[i] * Hy[j]) / ys
                        if not U.finite(value) then
                            valid = false
                        end
                        H[i][j], H[j][i] = value, value
                    end
                end
                if not valid then
                    H = identity(n)
                end
            end
        end
    end
    stop(self, 'iteration_limit')
end

-- Brent narrows a finite minimization interval; it does not test a signed root.
-- Its interval/step tolerance and the gradient methods' tolerances differ.
local function brent_minimum(self)
    local o = U.options(
        self,
        {
            relative_tolerance = 1.4901161193847656e-8,
            x_tolerance = self._minradius or 1e-7,
            max_iterations = 1000
        }
    )
    U.number(o, 'relative_tolerance', 2.220446049250313e-16, 0.1)
    U.number(o, 'x_tolerance', 0, math.huge)
    U.number(o, 'max_iterations', 1, math.huge, true)
    local domain = U.domain(self, o)
    assert(#domain.active == 1, 'Brent minimization requires exactly one active variable')
    local j = domain.active[1]
    local a, b = domain.lower[j], domain.upper[j]
    assert(U.finite(a) and U.finite(b), 'Brent minimization requires finite lower and upper bounds')

    local function evaluate(t, phase, iteration)
        local z = U.copy(domain.zero);
        z[j] = t
        return domain.evaluate(z, { phase = phase, iteration = iteration })
    end
    self._radius = (b - a) * domain.scale[j] / 2
    -- Include both closed-interval endpoints in the best-so-far result.
    local startf = evaluate(0, 'initial', 0)
    if startf == nil then
        return stop(self, 'budget')
    end
    if a ~= 0 and evaluate(a, 'brent_endpoint', 0) == nil then
        return stop(self, 'budget')
    end
    if b ~= 0 and evaluate(b, 'brent_endpoint', 0) == nil then
        return stop(self, 'budget')
    end
    local golden = (3 - sqrt(5)) / 2
    local x = (a < 0 and 0 < b) and 0 or a + golden * (b - a)
    local fx = x == 0 and startf or evaluate(x, 'brent_initial', 0)
    if fx == nil then
        return stop(self, 'budget')
    end
    local w, v, fw, fv = x, x, fx, fx
    local step, previous = 0, 0
    for iteration = 1, o.max_iterations do
        local middle = a + (b - a) / 2
        local tol = o.relative_tolerance * abs(x)
            + math.max(o.x_tolerance / domain.scale[j] / 3, 1e-15)
        if abs(x - middle) <= 2 * tol - (b - a) / 2 then
            return stop(self, 'converged')
        end
        local parabolic = false
        if abs(previous) > tol and U.finite(fx) and U.finite(fw) and U.finite(fv) then
            local r = (x - w) * (fx - fv)
            local q = (x - v) * (fx - fw)
            local p = (x - v) * q - (x - w) * r
            q = 2 * (q - r);
            if q > 0 then
                p = -p
            end;
            q = abs(q)
            local old = previous;
            previous = step
            if q > 0 and abs(p) < abs(0.5 * q * old) and p > q * (a - x) and p < q * (b - x) then
                step = p / q;
                parabolic = true
                local u = x + step
                if u - a < 2 * tol or b - u < 2 * tol then
                    step = middle >= x and tol or -tol
                end
            end
        end
        if not parabolic then
            previous = x >= middle and a - x or b - x;
            step = golden * previous
        end
        local u = x + (abs(step) >= tol and step or (step >= 0 and tol or -tol))
        u = math.max(a, math.min(b, u))
        if u == x then
            return stop(self, 'step_tolerance')
        end
        local fu = evaluate(u, parabolic and 'brent_parabolic' or 'brent_golden', iteration)
        if fu == nil then
            return stop(self, 'budget')
        end
        if fu <= fx then
            if u >= x then
                a = x
            else
                b = x
            end
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else
            if u < x then
                a = u
            else
                b = u
            end
            if fu <= fw or w == x then
                v, fv, w, fw = w, fw, u, fu
            elseif fu <= fv or v == x or v == w then
                v, fv = u, fu
            end
        end
        self._radius = (b - a) * domain.scale[j] / 2
    end
    stop(self, 'iteration_limit')
end

function M.run(self, variant)
    if variant == 'gradient' then
        return gradient_search(self, false)
    elseif variant == 'bfgs' then
        return gradient_search(self, true)
    elseif variant == 'brent' then
        return brent_minimum(self)
    end
    error('unknown gradient/scalar minimization method: ' .. tostring(variant))
end
return M
