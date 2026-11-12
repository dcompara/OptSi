-- Common constructor for scalar/Pareto search, signed roots and least squares.
-- Lua 5.1 (SIMION 8.1); EHVI and LM delegate to their own public objects.
-- Root names return the unchanged root_solvers object, including solve()/best().
-- Construct with M{algorithm, start, step, maxcalls, minradius, options, func}.
-- With func: scalar/Pareto/LM use run(), roots solve(); otherwise values()/result().
-- Algorithm-specific settings belong exclusively in options={...}.
-- NSGA-II/MOPSO are vector-only public names for the existing GA/PSO engines.
local U = require 'optimizer_utils'
local unpack = unpack or table.unpack
local M = {};
M.__index = M
local STOP = {}
local methods = {
    ['Nelder-Mead'] = true,
    SPSA = true,
    JADE = true,
    Pattern_Search = true,
    PSO = true,
    GA = true,
    Simulated_Annealing = true,
    Gradient_Descent = true,
    BFGS = true,
    Brent_Minimize = true,
    BOBYQA = true,
    ['CMA-ES'] = true,
    BADS = true,
    ParEGO = true,
    EGO = true,
    EHVI = true
}
local roots = { Newton = true, Broyden = true, Brent = true }
local vector_aliases = { ['NSGA-II'] = 'GA', MOPSO = 'PSO' }
local constructor_fields = {
    algorithm = true,
    start = true,
    step = true,
    maxcalls = true,
    minradius = true,
    mode = true,
    func = true,
    options = true
}

local function clone(x)
    if type(x) ~= 'table' then
        return x
    end
    local y = {};
    for k, v in pairs(x) do
        y[k] = clone(v)
    end;
    return y
end

local function positive(v, name, integer)
    assert(
        U.finite(v) and v > 0 and (not integer or v == math.floor(v)),
        name .. ' must be finite and positive'
    )
    return v
end

local function resume(self, ...)
    local ok, err = coroutine.resume(self._co, ...)
    if not ok then
        if err == STOP then
            self._status = 'budget'
        else
            self._status = 'error';
            error(err, 0)
        end
    end
end

-- Requires SIMION's installed simionx modules; no substitute simplex is supplied.
-- The common evaluator still enforces bounds and the total request budget.
local function native_simplex(self)
    -- Fixed coordinates are removed; the search itself is SIMION's native simplex.
    local d = U.domain(self, self._options)
    self._resolved_options = clone(self._options)
    if #d.active == 0 then
        self:f(self._start);
        self._radius = 0;
        return
    end
    local start, step = {}, {}
    for i, j in ipairs(d.active) do
        start[i] = self._start[j];
        step[i] = self._step[j]
    end
    local opt = (require 'simionx.SimplexOptimizer') {
        start = start,
        step = step,
        maxcalls = self._maxcalls,
        minradius = self._minradius
    }
    while opt:running() do
        local reduced = { opt:values() };
        local x = U.copy(self._start)
        for i, j in ipairs(d.active) do
            x[j] = reduced[i]
        end
        self._request_info = { phase = 'native_simplex' }
        opt:result(self:f(x));
        self._radius = opt:radius()
    end
end

local function dispatch(self)
    -- Common-object Pareto modes share the evaluation/archiving contract.
    -- GA+vector means NSGA-II; PSO+vector means MOPSO. ParEGO has its own search
    -- engine because it fits a surrogate, not because its physical API differs.
    local a = self._algorithm
    if self._mode == 'multiobjective' and (a == 'GA' or a == 'PSO') then
        require('multiobjective').run(self, a)
    elseif a == 'Nelder-Mead' then
        native_simplex(self)
    elseif a == 'SPSA' then
        require('spsa').run(self)
    elseif a == 'GA' then
        require('evolutionary').run(self, 'ga')
    elseif a == 'JADE' then
        require('evolutionary').run(self, 'jade')
    elseif a == 'Pattern_Search' then
        require('derivative_free').run(self, 'pattern')
    elseif a == 'PSO' then
        require('derivative_free').run(self, 'pso')
    elseif a == 'Simulated_Annealing' then
        require('derivative_free').run(self, 'annealing')
    elseif a == 'Gradient_Descent' then
        require('gradient_methods').run(self, 'gradient')
    elseif a == 'BFGS' then
        require('gradient_methods').run(self, 'bfgs')
    elseif a == 'Brent_Minimize' then
        require('gradient_methods').run(self, 'brent')
    elseif a == 'CMA-ES' then
        require('cmaes').run(self)
    elseif a == 'ParEGO' then
        require('parego').run(self)
    elseif a == 'EGO' then
        require('ego').run(self)
    elseif a == 'BOBYQA' then
        require('bobyqa').run(self)
    elseif a == 'BADS' then
        require('bads').run(self)
    end
end

local function execute(self)
    dispatch(self)
    if self._status == 'running' then
        self._status = self._stop_reason
            or (self._callcount >= self._maxcalls and 'budget' or 'stopped')
    end
end

local function new(t)
    assert(type(t) == 'table', 'optimizer settings must be a table')
    -- Least squares has signed vector feedback but minimizes a measured cost.
    -- Keep its API and validation separate from scalar/Pareto and exact roots.
    if t.algorithm == 'LM' then return require('least_squares')(t) end
    -- EHVI delegates to its validated two-objective object.
    if t.algorithm == 'EHVI' then return require('ehvi')(t) end
    if roots[t.algorithm] then
        assert(t.mode == nil or t.mode == 'root',
            t.algorithm .. ' solves signed equations; omit mode or use mode="root"')
        return require('root_solvers')(t)
    end
    for key in pairs(t) do
        assert(
            constructor_fields[key],
            'unknown constructor field: ' .. tostring(key) .. '; method settings belong in options'
        )
    end
    local requested_name = t.algorithm or 'Nelder-Mead'
    local vector_alias = vector_aliases[requested_name]
    local name = vector_alias or requested_name
    assert(methods[name], 'Unknown minimizer: ' .. tostring(name))
    assert(type(t.start) == 'table' and #t.start > 0, 'start must be a nonempty vector')
    assert(t.func == nil or type(t.func) == 'function', 'func must be a function')
    local start, step = U.copy(t.start), {}
    if t.step then
        assert(type(t.step) == 'table' and #t.step == #start, 'step length must match start')
    end
    for j = 1, #start do
        step[j] = t.step and t.step[j] or 1
        assert(U.finite(start[j]) and U.finite(step[j]), 'start and step must be finite')
    end
    assert(t.options == nil or type(t.options) == 'table', 'options must be a table')
    local mode = t.mode or ((vector_alias or name == 'ParEGO') and 'multiobjective' or 'scalar')
    assert(mode == 'scalar' or mode == 'multiobjective', 'mode must be scalar or multiobjective')
    assert(
        not vector_alias or mode == 'multiobjective',
        requested_name .. ' requires mode="multiobjective"; mode="scalar" is not supported'
    )
    assert(name ~= 'ParEGO' or mode == 'multiobjective', 'ParEGO requires mode="multiobjective"')
    assert(
        mode ~= 'multiobjective' or name == 'ParEGO' or name == 'GA' or name == 'PSO',
        'objective-vector mode is supported by GA, PSO and ParEGO; other methods need a scalar cost'
    )
    local self = setmetatable({
        _algorithm = name,
        _mode = mode,
        _start = start,
        _step = step,
        _func = t.func,
        _maxcalls = positive(t.maxcalls or 250, 'maxcalls', true),
        _minradius = positive(t.minradius or 1e-7, 'minradius'),
        _options = clone(t.options or {}),
        _callcount = 0,
        _evaluations = 0,
        _rejected_calls = 0,
        _best_value = math.huge,
        _request_info = {},
        _test_value = {},
        _status = 'ready'
    }, M)
    U.domain(self, self._options) -- Validate before requesting an objective evaluation.
    if not self._func then
        self._func = function(...)
            return coroutine.yield()
        end
        self._co = coroutine.create(execute);
        self._status = 'running';
        resume(self, self)
    end
    return self
end

function M:f(x)
    -- Every method passes through this gate: it enforces the global request
    -- budget and records the measured result before the search can stop.
    if self._callcount >= self._maxcalls then
        error(STOP, 0)
    end
    assert(type(x) == 'table' and #x == #self._start, 'candidate has wrong dimension')
    self._callcount = self._callcount + 1
    self._request_info = self._request_info or {}
    self._request_info.request_index = self._callcount
    self._test_value = U.copy(x)
    for j = 1, #x do
        assert(U.finite(x[j]), 'candidate must be finite')
        local lo, hi = self._options.lower, self._options.upper
        if (lo and x[j] < lo[j]) or (hi and x[j] > hi[j]) then
            assert(self._mode == 'scalar', 'multiobjective candidate outside bounds')
            self._rejected_calls = self._rejected_calls + 1
            return math.huge -- Count the request without evaluating an infeasible point.
        end
    end
    self._evaluations = self._evaluations + 1
    local value = self._func(unpack(x))
    if self._mode == 'multiobjective' then
        assert(
            type(value) == 'table' and #value >= 2,
            'multiobjective result must be a vector (at least 2)'
        )
        assert(
            not self._objective_count or self._objective_count == #value,
            'objective dimension changed'
        )
        self._objective_count = #value
        for j = 1, #value do
            assert(U.finite(value[j]), 'objectives must be finite')
        end
        -- Record every completed measurement, including the last one in the budget.
        -- Search populations may be bounded; the public archive is never truncated.
        self._pareto = self._pareto or {}
        require('multiobjective').update_front(self._pareto, { x = x, objectives = value })
        return U.copy(value)
    end
    assert(type(value) == 'number', 'scalar minimization requires one numeric cost')
    if not U.finite(value) then
        return math.huge
    end
    if value < self._best_value then
        self._best_point, self._best_value = U.copy(x), value
    end
    return value
end

function M:run()
    assert(not self._co, 'use values()/result() for an optimizer without func')
    assert(self._status == 'ready', 'create a new optimizer to run again')
    self._status = 'running'
    local ok, err = pcall(execute, self) -- Direct callback mode: no yield across pcall in Lua 5.1.
    if not ok then
        if err == STOP then
            self._status = 'budget'
        else
            self._status = 'error';
            error(err, 0)
        end
    end
    if self._mode == 'multiobjective' then
        return self:front()
    end
    return self:best()
end

function M:running()
    return self._co ~= nil and coroutine.status(self._co) == 'suspended'
end

function M:values()
    return unpack(self._test_value)
end

-- Resume exactly the outstanding candidate. The next request is generated
-- only after its scalar cost or complete objective vector has been supplied.
function M:result(value)
    assert(self:running(), 'optimizer is not waiting for a result');
    resume(self, value)
end

-- Counts include an outstanding request before feedback arrives. Evaluations are
-- requests dispatched to the objective, not a count of physical flights or Refines.
-- Bound rejections spend requests without dispatching the objective.
function M:ncalls()
    return self._callcount
end

function M:nevaluations()
    return self._evaluations
end

function M:nrejected()
    return self._rejected_calls
end

function M:radius()
    return self._radius
end

-- A budget or generic stopped status is not a certificate of a minimum.
function M:status()
    return self._status
end

-- Scalar result: copied best parameter vector and its measured cost. An
-- exploratory probe can be best even when it was not an accepted search iterate.
function M:best()
    assert(self._mode == 'scalar', 'a Pareto front has no single best point; use front()')
    return self._best_point and U.copy(self._best_point), self._best_value
end

-- Vector result: copied entries {x=..., objectives=...}, with all costs minimized.
function M:front()
    return clone(self._pareto or {})
end

function M:info()
    return clone(self._request_info)
end

function M:options()
    return clone(self._resolved_options or self._options)
end
M.new = new
return setmetatable(M, {
    __call = function(_, t)
        return new(t)
    end
})
