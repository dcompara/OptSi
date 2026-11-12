-- Safeguarded, box-projected Levenberg--Marquardt least squares, Lua 5.1.
-- F = 1/2 sum((r_i / residual_scale_i)^2), m >= number of active parameters.
-- Independent of root_solvers: a least-squares minimum need not be a root.
-- Damped steps use column-pivoted Householder QR of an augmented system.
-- Bounds use an outward-gradient working set followed by box clipping; this
-- is not a complete trust-region reflective algorithm. Nonlinear physical
-- feasibility remains the caller's responsibility; no geometry is repaired.
local U = require 'optimizer_utils'
local QR = require 'lm_qr'
local M = {}
M.__index = M
local unpack = unpack or table.unpack
local abs, max, min, sqrt = math.abs, math.max, math.min, math.sqrt
local EPS = 2.220446049250313e-16

local function clone(t)
    if type(t) ~= 'table' then return t end
    local c = {}
    for k, v in pairs(t) do c[k] = clone(v) end
    return c
end

local function dense(t, n, name)
    assert(type(t) == 'table' and #t == n, name .. ' must be a dense vector')
    local count = 0
    for k in pairs(t) do
        assert(type(k) == 'number' and k == math.floor(k) and k >= 1 and k <= n,
            name .. ' must be a dense vector')
        count = count + 1
    end
    assert(count == n, name .. ' must be a dense vector')
end

local function norminf(v)
    local a = 0
    for i = 1, #v do a = max(a, abs(v[i])) end
    return a
end

local function norm(v)
    local a = norminf(v)
    if a == 0 then return 0 end
    local s = 0
    for i = 1, #v do s = s + (v[i] / a)^2 end
    return a * sqrt(s)
end

local function stop(self, status)
    -- Retain every valid measurement, including the last pending/probe result.
    if self._best_norm <= self._options.ftol then status = 'residual_tolerance' end
    self._status = status
end

-- nil: no request left. false: returned nonfinite residual/scaled cost.
-- Malformed vectors and callback exceptions abort, rather than becoming
-- ordinary physical failures. Every probe and rejected trial spends a request.
local function evaluate(self, z, info)
    if self._count >= self._maxcalls then return nil end
    local x = self._domain.physical(z)
    self._test_value = x
    self._count = self._count + 1
    self._info = info
    info.request_index = self._count
    self._evaluations = self._evaluations + 1
    local raw
    if self._func then raw = self._func(unpack(x)) else raw = coroutine.yield() end
    self._completed = self._completed + 1
    dense(raw, self._m, 'LM residual result')
    local f = {}
    for i = 1, self._m do
        assert(type(raw[i]) == 'number', 'LM residual components must be numbers')
        f[i] = raw[i] / self._residual_scale[i]
        if not U.finite(raw[i]) or not U.finite(f[i]) then
            self._invalid = self._invalid + 1
            return false
        end
    end
    local length = norm(f)
    local cost = (0.5 * length) * length
    if not U.finite(cost) then
        self._invalid = self._invalid + 1
        return false
    end
    if cost < self._best_cost or (cost == self._best_cost and norminf(f) < self._best_norm) then
        self._best_point, self._best_raw = U.copy(x), U.copy(raw)
        self._best_cost, self._best_norm = cost, norminf(f)
    end
    return f, cost
end

local function jacobian(self, z, f, iteration)
    local d, o, m = self._domain, self._options, self._m
    local J, one_sided, unresolved = {}, {}, {}
    for i = 1, m do J[i] = {} end
    for col, j in ipairs(d.active) do
        local center = d.physical(z)[j]
        local h = max(self._fd_steps[j] * max(1, abs(z[j])), 8 * EPS * abs(center) / d.scale[j])
        local hp, hm = min(h, d.upper[j] - z[j]), min(h, z[j] - d.lower[j])
        local available_hp, available_hm = hp, hm
        if hp < h * 0.25 and hm >= h * 0.25 then hp = 0 end
        if hm < h * 0.25 and hp >= h * 0.25 then hm = 0 end
        local function probe(sign, distance)
            if distance <= 0 or z[j] + sign * distance == z[j] then return false, 0 end
            local p = U.copy(z)
            p[j] = z[j] + sign * distance
            distance = sign * (d.physical(p)[j] - center) / d.scale[j]
            if distance <= 0 then return false, 0 end
            local value = evaluate(self, p, {phase='jacobian', iteration=iteration,
                coordinate=j, sign=sign, normalized_displacement=distance,
                difference=o.difference})
            if value == false then self._diagnostics.failed_probes = self._diagnostics.failed_probes + 1 end
            return value, distance
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
            if not fp then
                fm, hm = probe(-1, available_hm)
                if fm == nil then return nil, 'budget' end
            end
        else
            fm, hm = probe(-1, hm)
            if fm == nil then return nil, 'budget' end
            if not fm then
                fp, hp = probe(1, available_hp)
                if fp == nil then return nil, 'budget' end
            end
        end
        if not fp and not fm then
            self._diagnostics.failed_column = j
            return nil, 'jacobian_failed'
        end
        local largest = 0
        for i = 1, m do
            local v
            if fp and fm then
                v = (hm * (fp[i] - f[i]) / hp + hp * (f[i] - fm[i]) / hm) / (hp + hm)
            elseif fp then v = (fp[i] - f[i]) / hp
            else v = (f[i] - fm[i]) / hm end
            if not U.finite(v) then return nil, 'jacobian_failed' end
            J[i][col], largest = v, max(largest, abs(v))
        end
        if not (fp and fm) then one_sided[#one_sided + 1] = j end
        if largest == 0 then unresolved[#unresolved + 1] = j end
    end
    self._diagnostics.one_sided_columns = one_sided
    self._diagnostics.unresolved_columns = unresolved
    self._diagnostics.jacobian = clone(J)
    self._diagnostics.jacobian_builds = self._diagnostics.jacobian_builds + 1
    return J
end

local function solve_model(self, J, f, z)
    local m, n, d = self._m, #self._domain.active, self._domain
    local jscale, fscale = 0, max(1, norminf(f))
    for i = 1, m do
        for j = 1, n do jscale = max(jscale, abs(J[i][j])) end
    end
    jscale = max(jscale, 1e-150)
    local A, zero, column_norm, gradient, free = {}, {}, {}, {}, {}
    for i = 1, m do
        A[i], zero[i] = {}, 0
        for j = 1, n do A[i][j] = J[i][j] / jscale end
    end
    local _, rank_info = QR.solve(A, zero, self._options.rank_tolerance)
    if not rank_info then return nil end
    self._diagnostics.rank = rank_info.rank
    self._diagnostics.qr_diagonal_ratio = rank_info.qr_diagonal_ratio
    local physical, metric, weak, bound_active = d.physical(z), 0, {}, {}
    local fnorm = norm(f) / fscale
    for col, j in ipairs(d.active) do
        local v, g = {}, 0
        for i = 1, m do
            v[i] = A[i][col]
            g = g + A[i][col] * (f[i] / fscale)
        end
        column_norm[col], gradient[col] = norm(v), g
        if column_norm[col] <= self._options.rank_tolerance then weak[#weak + 1] = j end
        local blocked = (physical[j] <= self._lower[j] and g >= 0)
            or (physical[j] >= self._upper[j] and g <= 0)
        if blocked then bound_active[#bound_active + 1] = j end
        if not blocked then
            free[#free + 1] = col
            if column_norm[col] > 0 and fnorm > 0 then
                metric = max(metric, abs(g) / (column_norm[col] * fnorm))
            end
        end
    end
    self._diagnostics.projected_gradient_metric = metric
    self._diagnostics.weak_columns = weak
    self._diagnostics.bound_active_columns = bound_active
    return {matrix=A, scale=jscale, column_norm=column_norm, free=free, metric=metric}
end

local function damped_step(self, model, f, mu)
    local m, k = self._m, #model.free
    if k == 0 then return {} end
    local A, b = {}, {}
    for i = 1, m + k do
        A[i] = {}
        b[i] = i <= m and -f[i] / model.scale or 0
        for j, col in ipairs(model.free) do
            A[i][j] = i <= m and model.matrix[i][col] or 0
        end
    end
    for j, col in ipairs(model.free) do
        A[m+j][j] = sqrt(mu) * max(1e-6, model.column_norm[col])
    end
    local reduced, info = QR.solve(A, b, 0)
    if not reduced or info.rank < k then return nil end
    local p = {}
    for j = 1, #self._domain.active do p[j] = 0 end
    for j, col in ipairs(model.free) do p[col] = reduced[j] end
    return p
end

local function execute(self)
    self._status = 'running'
    local d, o = self._domain, self._options
    local z = U.copy(d.zero)
    local f, cost = evaluate(self, z, {phase='initial', iteration=0})
    if f == nil then return stop(self, 'budget') end
    if not f then return stop(self, 'nonfinite_initial_value') end
    if norminf(f) <= o.ftol then return stop(self, 'residual_tolerance') end
    if #d.active == 0 then return stop(self, 'fixed_parameters') end
    local mu, nu = o.initial_damping, 2
    for iteration = 1, o.max_iterations do
        self._diagnostics.iterations = iteration
        self._diagnostics.iterate = {x=d.physical(z), scaled_residual=U.copy(f), cost=cost}
        if self._count >= self._maxcalls then return stop(self, 'budget') end
        local J, reason = jacobian(self, z, f, iteration)
        if not J then return stop(self, reason) end
        if self._best_norm <= o.ftol then return stop(self, 'residual_tolerance') end
        local model = solve_model(self, J, f, z)
        if not model then return stop(self, 'numerical_failure') end
        if model.metric <= o.gtol then
            return stop(self, #self._diagnostics.unresolved_columns > 0
                and 'unresolved_jacobian' or 'least_squares_stationary')
        end
        local accepted = false
        for attempt = 1, o.max_rejections do
            self._diagnostics.damping = mu
            local p = damped_step(self, model, f, mu)
            if not p then self._diagnostics.linear_failures = self._diagnostics.linear_failures + 1 end
            local newz, s, predicted = U.copy(z), {}, 0
            if p then
                for col, j in ipairs(d.active) do newz[j] = d.clip(z[j] + p[col], j) end
                local oldx, newx = d.physical(z), d.physical(newz)
                for col, j in ipairs(d.active) do
                    s[col] = (newx[j] - oldx[j]) / d.scale[j]
                    newz[j] = (newx[j] - self._start[j]) / d.scale[j]
                end
                self._radius = norminf(s)
                if self._radius <= o.xtol * (1 + norminf(z)) then
                    return stop(self, 'step_tolerance')
                end
                for i = 1, self._m do
                    local jp = 0
                    for col = 1, #s do jp = jp + J[i][col] * s[col] end
                    predicted = predicted - jp * (f[i] + 0.5 * jp)
                end
            end
            local history = {iteration=iteration, attempt=attempt, damping=mu,
                predicted_reduction=predicted, accepted=false, evaluated=false}
            self._diagnostics.steps[#self._diagnostics.steps + 1] = history
            if p and U.finite(predicted) and predicted > 0 then
                local newf, newcost = evaluate(self, newz,
                    {phase='lm_trial', iteration=iteration, attempt=attempt, damping=mu})
                if newf == nil then return stop(self, 'budget') end
                history.evaluated = true
                if newf then
                    local actual = cost - newcost
                    local rho = actual / predicted
                    history.actual_reduction, history.ratio = actual, rho
                    if actual > 0 and rho > o.acceptance_ratio then
                        history.accepted, accepted = true, true
                        self._diagnostics.accepted_steps = self._diagnostics.accepted_steps + 1
                        local oldcost = cost
                        z, f, cost = newz, newf, newcost
                        self._diagnostics.iterate = {x=d.physical(z), scaled_residual=U.copy(f), cost=cost}
                        mu = min(o.max_damping, max(o.min_damping, mu * max(1/3, 1 - (2*rho - 1)^3)))
                        nu = 2
                        if norminf(f) <= o.ftol then return stop(self, 'residual_tolerance') end
                        if actual <= o.cost_tol * oldcost and rho >= 0.25 then
                            return stop(self, 'cost_tolerance')
                        end
                        break
                    end
                end
            end
            self._diagnostics.rejected_steps = self._diagnostics.rejected_steps + 1
            if self._count >= self._maxcalls then return stop(self, 'budget') end
            if mu >= o.max_damping then return stop(self, 'damping_limit') end
            mu, nu = min(o.max_damping, mu * nu), min(nu * 2, 1e12)
        end
        if not accepted then return stop(self, 'no_progress') end
    end
    stop(self, 'iteration_limit')
end

local function resume(self, value)
    local ok, message = coroutine.resume(self._co, value)
    if not ok then self._status = 'error'; error(message, 0) end
end

function M.new(t)
    assert(type(t) == 'table', 'LM constructor table required')
    local fields = {algorithm=true, start=true, step=true, maxcalls=true,
        options=true, func=true, mode=true}
    for key in pairs(t) do assert(fields[key], 'unknown LM constructor field: '..tostring(key)) end
    assert(t.algorithm == nil or t.algorithm == 'LM', 'least_squares implements LM only')
    assert(t.mode == nil or t.mode == 'least_squares', 'LM requires mode="least_squares" or omitted mode')
    assert(t.func == nil or type(t.func) == 'function', 'func must be a function')
    assert(type(t.start) == 'table' and #t.start > 0, 'LM requires a nonempty start')
    dense(t.start, #t.start, 'start')
    if t.step then dense(t.step, #t.start, 'step') end
    assert(t.options == nil or type(t.options) == 'table', 'options must be a table')
    local o = {difference='central', fd_step=1e-5, ftol=1e-8, gtol=1e-8,
        xtol=1e-10, cost_tol=1e-12, initial_damping=1e-3, min_damping=1e-15,
        max_damping=1e15, acceptance_ratio=1e-4, rank_tolerance=1e-12,
        max_iterations=200, max_rejections=20}
    local allowed = {lower=true, upper=true, residual_scale=true}
    for key in pairs(o) do allowed[key] = true end
    for key, value in pairs(t.options or {}) do
        assert(allowed[key], 'unknown LM option: '..tostring(key))
        o[key] = clone(value)
    end
    for _, key in ipairs{'ftol','gtol','xtol','cost_tol'} do U.number(o, key, 0, 1e100) end
    for _, key in ipairs{'initial_damping','min_damping','max_damping'} do U.number(o, key, 1e-300, 1e300) end
    assert(o.min_damping <= o.initial_damping and o.initial_damping <= o.max_damping, 'invalid damping interval')
    U.number(o, 'acceptance_ratio', 0, 0.49)
    U.number(o, 'rank_tolerance', 0, 0.1)
    U.number(o, 'max_iterations', 1, 1e9, true)
    U.number(o, 'max_rejections', 1, 1e9, true)
    local self = setmetatable({_algorithm='LM', _start=U.copy(t.start), _step={},
        _options=o, _status='ready', _count=0, _evaluations=0, _completed=0,
        _invalid=0, _best_cost=math.huge, _best_norm=math.huge, _radius=math.huge,
        _func=t.func, _diagnostics={accepted_steps=0, rejected_steps=0,
            failed_probes=0, linear_failures=0, jacobian_builds=0, steps={}, unresolved_columns={}}}, M)
    self._maxcalls = t.maxcalls or 200
    assert(U.finite(self._maxcalls) and self._maxcalls >= 0 and self._maxcalls == math.floor(self._maxcalls),
        'maxcalls must be a nonnegative integer')
    for j = 1, #self._start do self._step[j] = t.step and t.step[j] or 1 end
    self._fd_steps = U.fd_steps(o, #self._start)
    for _, key in ipairs{'lower','upper'} do
        if o[key] then dense(o[key], #self._start, key) end
    end
    self._domain = U.domain(self, o)
    self._lower, self._upper = {}, {}
    for j = 1, #self._start do
        self._lower[j], self._upper[j] = o.lower and o.lower[j] or -math.huge, o.upper and o.upper[j] or math.huge
        if self._domain.scale[j] > 0 then
            assert((not U.finite(self._lower[j]) or U.finite(self._domain.lower[j]))
                and (not U.finite(self._upper[j]) or U.finite(self._domain.upper[j])),
                'normalized LM bounds must be representable')
        end
    end
    assert(type(o.residual_scale) == 'table' and #o.residual_scale > 0,
        'LM requires explicit positive residual_scale entries')
    self._m = #o.residual_scale
    dense(o.residual_scale, self._m, 'residual_scale')
    assert(self._m >= #self._domain.active, 'LM requires m >= n_active')
    for i = 1, self._m do
        assert(U.finite(o.residual_scale[i]) and o.residual_scale[i] > 0, 'invalid residual_scale')
    end
    self._residual_scale = U.copy(o.residual_scale)
    if not self._func then
        self._co = coroutine.create(function() execute(self) end)
        resume(self)
    end
    return self
end

function M:running() return self._co ~= nil and coroutine.status(self._co) == 'suspended' end
function M:values() assert(self:running(), 'no pending LM evaluation'); return unpack(self._test_value) end
function M:result(value) assert(self:running(), 'no pending LM evaluation'); resume(self, value) end
function M:ncalls() return self._count end
function M:nevaluations() return self._evaluations end
function M:ncompleted() return self._completed end
function M:ninvalid() return self._invalid end
function M:nrejected() return 0 end -- No out-of-box requests are dispatched.
function M:status() return self._status end
function M:radius() return self._radius end
function M:info() return clone(self._info or {}) end
function M:options() return clone(self._options) end
function M:diagnostics() return clone(self._diagnostics) end
function M:residual_norm() return self._best_norm end
function M:residuals_satisfied() return self._best_norm <= self._options.ftol end
function M:best()
    return self._best_point and U.copy(self._best_point), self._best_raw and U.copy(self._best_raw), self._best_cost
end
function M:run()
    assert(self._func and not self._co, 'use values()/result() without func')
    assert(self._status == 'ready', 'create a new LM optimizer to run again')
    local ok, message = pcall(execute, self)
    if not ok then self._status = 'error'; error(message, 0) end
    return self:best()
end
return setmetatable(M, {__call=function(_, t) return M.new(t) end})
