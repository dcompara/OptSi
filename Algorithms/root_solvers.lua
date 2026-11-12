-- Signed equations F(x)=0, deliberately separate from scalar minimization.
-- Lua 5.1, no native or matrix dependencies.
--
-- Coroutine/SIMION use:
--   local r=require('root_solvers'){
--     algorithm='Newton',start={1,1},step={1,1},maxcalls=100,
--     options={ftol=1e-8,residual_scale={1,1}}}
--   while r:running() do local x,y=r:values();r:result({x+y-3,x-y-1}) end
--   local x,F,norm=r:best();print(r:status())
-- Direct use: supply func=function(x,y) return {F1,F2} end, then r:solve().
-- solve() returns best_point, signed_residuals, status. M.new(args) also constructs a solver.
-- Scalar return values are accepted only when there is ONE variable.
-- Newton/Broyden require n signed conditions for n variables, positive step
-- scales and no fixed variables. Bounds are optional. residual_scale contains
-- positive characteristic units; ftol applies to max(abs(F_i)/scale_i).
-- xtol is a normalized step tolerance, not a guarantee that a root was found.
-- Brent requires one variable and bracket={a,b} (or finite lower/upper bounds).
-- A root bracket needs continuous F and opposite endpoint signs (or a zero).
-- status='bracket_tolerance' certifies a small sign bracket, not a small F.
-- Broyden uses the good rank-one Jacobian update, with a finite-difference
-- refresh on a singular model or a failed line search. See
-- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.html
-- Brent: R. P. Brent (1973), chapter 4; https://www.netlib.org/fmm/zeroin.f
local U = require 'optimizer_utils'
local M = {};
M.__index = M
local unpack = unpack or table.unpack
local abs, sqrt = math.abs, math.sqrt

local function copy(x)
    return U.copy(x)
end

local function norm2(x)
    local s = 0;
    for i = 1, #x do
        s = s + x[i] ^ 2
    end;
    return s
end

local function norminf(x)
    local s = 0;
    for i = 1, #x do
        s = math.max(s, abs(x[i]))
    end;
    return s
end

local function stop(self, status)
    self._status = status
end

local function physical(self, z)
    local x = {};
    for j = 1, self._n do
        x[j] = math.max(
            self._lower[j],
            math.min(self._upper[j], self._start[j] + z[j] * self._step[j])
        )
    end;
    return x
end

-- nil means the request budget is exhausted; false means nonfinite residuals.
-- Best stores raw signed residuals, ranked by max(abs(F_i)/residual_scale_i).
local function evaluate(self, z, phase, iteration)
    if self._count >= self._maxcalls then
        return nil
    end
    local x = physical(self, z)
    self._test_value = x;
    self._info = { phase = phase, iteration = iteration or 0 }
    self._count = self._count + 1
    local raw
    if self._func then
        raw = self._func(unpack(x))
    else
        raw = coroutine.yield()
    end
    if self._n == 1 and type(raw) == 'number' then
        raw = { raw }
    end
    assert(
        type(raw) == 'table' and #raw == self._n,
        'root solver requires a table of n signed conditions for n variables; a scalar cost is not a system'
    )
    local scaled = {}
    for i = 1, self._n do
        if not U.finite(raw[i]) then
            return false
        end
        scaled[i] = raw[i] / self._residual_scale[i]
        if not U.finite(scaled[i]) then
            return false
        end
    end
    local measure = norminf(scaled)
    if measure < self._best_norm then
        self._best_point, self._best_residual, self._best_norm = copy(x), copy(raw), measure
    end
    return scaled
end

-- Gaussian elimination with scaled partial pivoting. Solves J p = rhs.
local function linear_solve(J, rhs, tolerance)
    local A, b, scale = {}, {}, {}
    local n = #rhs
    for i = 1, n do
        A[i], b[i], scale[i] = copy(J[i]), rhs[i], 0
        for j = 1, n do
            scale[i] = math.max(scale[i], abs(A[i][j]))
        end
        if scale[i] == 0 or not U.finite(scale[i]) then
            return nil
        end
    end
    for k = 1, n do
        local pivot, ratio = k, 0
        for i = k, n do
            local r = abs(A[i][k]) / scale[i];
            if r > ratio then
                pivot, ratio = i, r
            end
        end
        if ratio <= tolerance then
            return nil
        end
        A[k], A[pivot] = A[pivot], A[k];
        b[k], b[pivot] = b[pivot], b[k]
        scale[k], scale[pivot] = scale[pivot], scale[k]
        for i = k + 1, n do
            local factor = A[i][k] / A[k][k]
            A[i][k] = 0
            for j = k + 1, n do
                A[i][j] = A[i][j] - factor * A[k][j]
            end
            b[i] = b[i] - factor * b[k]
        end
    end
    local x = {}
    for i = n, 1, -1 do
        local r = b[i];
        for j = i + 1, n do
            r = r - A[i][j] * x[j]
        end
        x[i] = r / A[i][i];
        if not U.finite(x[i]) then
            return nil
        end
    end
    return x
end

local function jacobian(self, z, F, iteration)
    local J = {};
    for i = 1, self._n do
        J[i] = {}
    end
    for j = 1, self._n do
        local center = physical(self, z)[j]
        local h = math.max(
            self._fd_steps[j] * math.max(1, abs(z[j])),
            8 * 2.220446049250313e-16 * abs(center) / self._step[j]
        )
        local hp = math.min(h, self._zu[j] - z[j])
        local hm = math.min(h, z[j] - self._zl[j])
        local available_hp, available_hm = hp, hm
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
            local p = copy(z);
            p[j] = z[j] + sign * distance
            distance = sign * (physical(self, p)[j] - center) / self._step[j]
            if distance <= 0 then
                return false, 0
            end
            return evaluate(self, p, 'jacobian', iteration), distance
        end
        local fp, fm
        if self._options.difference == 'central' then
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
                if fp == nil then
                    return nil, 'budget'
                end
            end
        end
        if not fp and not fm then
            return nil, 'jacobian_failed'
        end
        for i = 1, self._n do
            if fp and fm then
                J[i][j] = (hm * (fp[i] - F[i]) / hp + hp * (F[i] - fm[i]) / hm) / (hp + hm)
            elseif fp then
                J[i][j] = (fp[i] - F[i]) / hp
            else
                J[i][j] = (F[i] - fm[i]) / hm
            end
            if not U.finite(J[i][j]) then
                return nil, 'jacobian_failed'
            end
        end
    end
    return J
end

local function system_solve(self)
    local o, n = self._options, self._n
    local z = {};
    for i = 1, n do
        z[i] = 0
    end
    local F = evaluate(self, z, 'initial', 0)
    if F == nil then
        return stop(self, 'budget')
    end
    if not F then
        return stop(self, 'nonfinite_initial_value')
    end
    local J, fresh, status
    for iteration = 1, o.max_iterations do
        if norminf(F) <= o.ftol then
            return stop(self, 'converged')
        end
        if self._count >= self._maxcalls then
            return stop(self, 'budget')
        end
        if self._algorithm == 'Newton' or not J then
            J, status = jacobian(self, z, F, iteration);
            fresh = true
            if not J then
                return stop(self, status)
            end
        end
        local accepted, newz, newF, s
        -- A stale Broyden model may be refreshed once at the same point.
        for attempt = 1, 2 do
            local rhs = {};
            for i = 1, n do
                rhs[i] = -F[i]
            end
            local p = linear_solve(J, rhs, o.pivot_tolerance)
            if p then
                local alpha = 1
                for j = 1, n do
                    if p[j] > 0 then
                        alpha = math.min(alpha, (self._zu[j] - z[j]) / p[j])
                    elseif p[j] < 0 then
                        alpha = math.min(alpha, (self._zl[j] - z[j]) / p[j])
                    end
                end
                local merit = norm2(F)
                for ls = 1, o.max_linesearch do
                    if alpha < o.min_alpha then
                        break
                    end
                    newz, s = {}, {}
                    for j = 1, n do
                        newz[j] = math.max(self._zl[j], math.min(self._zu[j], z[j] + alpha * p[j]))
                        s[j] = newz[j] - z[j]
                    end
                    if norminf(s) == 0 then
                        break
                    end
                    newF = evaluate(self, newz, 'root_line_search', iteration)
                    if newF == nil then
                        return stop(self, 'budget')
                    end
                    -- Merit is used only to safeguard a coupled root step.
                    if
                        newF
                        and (
                            norminf(newF) <= o.ftol
                            or norm2(newF) <= (1 - 2 * o.armijo_c1 * alpha) * merit
                        )
                    then
                        accepted = true;
                        break
                    end
                    alpha = alpha * o.backtrack
                end
            end
            if accepted then
                break
            end
            if fresh or self._algorithm == 'Newton' then
                return stop(self, p and 'line_search_failed' or 'singular_jacobian')
            end
            J, status = jacobian(self, z, F, iteration);
            fresh = true
            if not J then
                return stop(self, status)
            end
        end
        if not accepted then
            return stop(self, 'line_search_failed')
        end
        self._radius = norminf(s)
        if norminf(newF) <= o.ftol then
            return stop(self, 'converged')
        end
        if self._radius <= o.xtol then
            return stop(self, 'step_tolerance')
        end
        if self._algorithm == 'Broyden' then
            local ss = norm2(s)
            if ss > 0 then
                for i = 1, n do
                    local js = 0;
                    for j = 1, n do
                        js = js + J[i][j] * s[j]
                    end
                    local correction = (newF[i] - F[i] - js) / ss
                    for j = 1, n do
                        J[i][j] = J[i][j] + correction * s[j]
                    end
                end
                fresh = false
            else
                J = nil
            end
        end
        z, F = newz, newF
    end
    stop(self, 'iteration_limit')
end

local function brent_root(self)
    local o = self._options
    local a = (self._bracket[1] - self._start[1]) / self._step[1]
    local b = (self._bracket[2] - self._start[1]) / self._step[1]
    local fa = evaluate(self, { a }, 'bracket_endpoint', 0)
    if fa == nil then
        return stop(self, 'budget')
    end
    if not fa then
        return stop(self, 'nonfinite_bracket_value')
    end
    fa = fa[1];
    if abs(fa) <= o.ftol then
        return stop(self, 'converged')
    end
    local fb = evaluate(self, { b }, 'bracket_endpoint', 0)
    if fb == nil then
        return stop(self, 'budget')
    end
    if not fb then
        return stop(self, 'nonfinite_bracket_value')
    end
    fb = fb[1];
    if abs(fb) <= o.ftol then
        return stop(self, 'converged')
    end
    if (fa > 0) == (fb > 0) then
        return stop(self, 'invalid_bracket')
    end
    local c, fc, step, previous = a, fa, b - a, b - a
    for iteration = 1, o.max_iterations do
        if (fb > 0) == (fc > 0) then
            c, fc, step, previous = a, fa, b - a, b - a
        end
        if abs(fc) < abs(fb) then
            -- Sequential Brent rotation: c receives the OLD b, not old a.
            local oldb, oldfb = b, fb;
            a, fa = b, fb;
            b, fb = c, fc;
            c, fc = oldb, oldfb
        end
        local tol = 2 * 2.220446049250313e-16 * abs(b) + o.xtol / 2
        local middle = (c - b) / 2
        self._radius = abs(middle)
        self._last_bracket =
            { physical(self, { math.min(b, c) })[1], physical(self, { math.max(b, c) })[1] }
        if abs(fb) <= o.ftol then
            return stop(self, 'converged')
        end
        if abs(middle) <= tol then
            return stop(self, 'bracket_tolerance')
        end
        local interpolation = false
        if abs(previous) >= tol and abs(fa) > abs(fb) then
            local s = fb / fa
            local p, q
            if a == c then
                p, q = 2 * middle * s, 1 - s
            else
                q = fa / fc;
                local r = fb / fc
                p = s * (2 * middle * q * (q - r) - (b - a) * (r - 1));
                q = (q - 1) * (r - 1) * (s - 1)
            end
            if p > 0 then
                q = -q
            end;
            p = abs(p)
            if
                q ~= 0
                and U.finite(p)
                and U.finite(q)
                and 2 * p < 3 * middle * q - abs(tol * q)
                and p < abs(previous * q / 2)
            then
                previous, step = step, p / q;
                interpolation = true
            end
        end
        if not interpolation then
            step, previous = middle, middle
        end
        a, fa = b, fb
        b = b + (abs(step) > tol and step or (middle > 0 and tol or -tol))
        local f = evaluate(
            self,
            { b },
            interpolation and 'brent_interpolation' or 'brent_bisection',
            iteration
        )
        if f == nil then
            return stop(self, 'budget')
        end
        if not f then
            return stop(self, 'nonfinite_value')
        end
        fb = f[1]
    end
    stop(self, 'iteration_limit')
end

local function run(self)
    self._status = 'running'
    if self._algorithm == 'Brent' then
        brent_root(self)
    else
        system_solve(self)
    end
end

local function resume(self, value)
    local ok, message = coroutine.resume(self._co, value)
    if not ok then
        self._status = 'error';
        error(message, 0)
    end
end

function M.new(args)
    assert(type(args) == 'table', 'root solver options table required')
    local algorithm = args.algorithm or 'Newton'
    assert(
        algorithm == 'Newton' or algorithm == 'Broyden' or algorithm == 'Brent',
        'unknown root solver: ' .. tostring(algorithm)
    )
    local self = setmetatable(
        {
            _algorithm = algorithm,
            _count = 0,
            _best_norm = math.huge,
            _status = 'ready',
            _radius = math.huge,
            _func = args.func
        },
        M
    )
    assert(not args.func or type(args.func) == 'function', 'func must be a function')
    local o = {
        ftol = 1e-8,
        xtol = 1e-10,
        fd_step = 1e-5,
        pivot_tolerance = 1e-12,
        max_iterations = 100,
        max_linesearch = 30,
        min_alpha = 1e-12,
        armijo_c1 = 1e-4,
        backtrack = 0.5
    }
    for k, v in pairs(args.options or {}) do
        o[k] = v
    end
    self._options = o
    for _, key in ipairs { 'ftol', 'xtol' } do
        U.number(o, key, 0, math.huge)
    end
    U.number(o, 'pivot_tolerance', 0, 0.1)
    U.number(o, 'max_iterations', 1, math.huge, true);
    U.number(o, 'max_linesearch', 1, math.huge, true)
    U.number(o, 'min_alpha', 1e-300, 1);
    U.number(o, 'armijo_c1', 1e-12, 0.49)
    U.number(o, 'backtrack', 1e-6, 0.999999)
    self._maxcalls = args.maxcalls or 200
    assert(
        U.finite(self._maxcalls)
            and self._maxcalls >= 0
            and self._maxcalls == math.floor(self._maxcalls),
        'maxcalls must be a nonnegative integer'
    )
    local bracket = args.bracket or o.bracket
    if algorithm == 'Brent' and not bracket and o.lower and o.upper then
        bracket = { o.lower[1], o.upper[1] }
    end
    if algorithm == 'Brent' then
        assert(
            type(bracket) == 'table'
                and #bracket == 2
                and U.finite(bracket[1])
                and U.finite(bracket[2])
                and bracket[1] < bracket[2],
            'Brent root finder requires a finite increasing bracket={a,b}'
        )
        self._bracket = copy(bracket)
    end
    self._start = args.start and copy(args.start) or (
            bracket and { bracket[1] / 2 + bracket[2] / 2 }
        )
    assert(self._start and #self._start > 0, 'start must be a nonempty coordinate table')
    self._n = #self._start
    if algorithm == 'Brent' then
        U.number(o, 'fd_step', 1e-12, 1)
    else
        self._fd_steps = U.fd_steps(o, self._n)
    end
    if algorithm == 'Brent' then
        assert(self._n == 1, 'Brent root finder requires one variable')
    end
    self._step, self._lower, self._upper, self._zl, self._zu, self._residual_scale =
        {}, {}, {}, {}, {}, {}
    for _, key in ipairs { 'lower', 'upper', 'residual_scale' } do
        if o[key] then
            assert(type(o[key]) == 'table' and #o[key] == self._n, key .. ' must have n entries')
        end
    end
    if args.step then
        assert(#args.step == self._n, 'step must have n entries')
    end
    for j = 1, self._n do
        local x = self._start[j]
        local scale = args.step and abs(args.step[j]) or math.max(1, abs(x))
        local lower = o.lower and o.lower[j] or -math.huge
        local upper = o.upper and o.upper[j] or math.huge
        local residual_scale = o.residual_scale and o.residual_scale[j] or 1
        assert(
            U.finite(x) and U.finite(scale) and scale > 0,
            'root coordinates must be finite and step scales positive'
        )
        assert(
            type(lower) == 'number'
                and type(upper) == 'number'
                and lower < upper
                and x >= lower
                and x <= upper,
            'invalid root bounds or start outside bounds'
        )
        assert(
            U.finite(residual_scale) and residual_scale > 0,
            'residual_scale must contain positive finite numbers'
        )
        self._step[j], self._lower[j], self._upper[j], self._residual_scale[j] =
            scale, lower, upper, residual_scale
        self._zl[j], self._zu[j] = (lower - x) / scale, (upper - x) / scale
    end
    if bracket then
        assert(
            bracket[1] >= self._lower[1] and bracket[2] <= self._upper[1],
            'bracket outside bounds'
        )
    end
    if not self._func then
        self._co = coroutine.create(function()
            run(self)
        end);
        resume(self)
    end
    return self
end

function M:running()
    return self._co ~= nil and coroutine.status(self._co) == 'suspended'
end

function M:values()
    assert(self._test_value, 'no pending evaluation');
    return unpack(self._test_value)
end

function M:result(value)
    assert(self:running(), 'root solver terminated');
    resume(self, value)
end

function M:ncalls()
    return self._count
end

function M:radius()
    return self._radius
end

function M:status()
    return self._status
end

function M:info()
    return self._info
end

function M:options()
    local o = {};
    for k, v in pairs(self._options) do
        o[k] = v
    end;
    return o
end

-- Return copied best coordinates, raw signed residuals and the scaled infinity norm.
function M:best()
    return self._best_point and copy(self._best_point),
        self._best_residual and copy(self._best_residual),
        self._best_norm
end

function M:bracket()
    return self._last_bracket and copy(self._last_bracket)
end

-- Unlike best(), the third solve() return is a status string. A small step or
-- sign bracket need not satisfy ftol; inspect both residuals and status.
function M:solve()
    assert(self._func, 'solve() requires func; otherwise use values()/result()')
    assert(self._status == 'ready', 'root solver already used')
    run(self)
    return self._best_point and copy(self._best_point),
        self._best_residual and copy(self._best_residual),
        self._status
end
setmetatable(M, {
    __call = function(_, args)
        return M.new(args)
    end
})
return M
