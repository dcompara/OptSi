-- Small shared helpers for bounded, derivative-free searches. Lua 5.1 syntax.
local M = {}
local unpack = unpack or table.unpack

function M.copy(x)
    return { unpack(x) }
end

function M.finite(x)
    return type(x) == 'number' and x == x and x > -math.huge and x < math.huge
end

function M.options(self, defaults)
    local o = {}
    for k, v in pairs(defaults) do
        o[k] = v
    end
    for k, v in pairs(self._options or {}) do
        o[k] = v
    end
    self._resolved_options = o
    return o
end

function M.number(o, key, low, high, integer)
    local v = o[key]
    assert(
        M.finite(v) and v >= low and v <= high and (not integer or v == math.floor(v)),
        'invalid optimizer option: ' .. key
    )
    return v
end

-- FD entries use the full parameter order, including fixed coordinates.
-- Return a private vector without changing the public scalar/vector option.
function M.fd_steps(o, n)
    if o.difference == nil then
        o.difference = 'central'
    end
    assert(o.difference == 'forward' or o.difference == 'central',
        'invalid optimizer option: difference')
    local steps = {}
    if type(o.fd_step) == 'table' then
        local count = 0
        for j, value in pairs(o.fd_step) do
            assert(type(j) == 'number' and j == math.floor(j) and j >= 1 and j <= n,
                'fd_step must have one entry per parameter')
            assert(M.finite(value) and value >= 1e-12 and value <= 1,
                'invalid optimizer option: fd_step')
            steps[j], count = value, count + 1
        end
        assert(count == n, 'fd_step must have one entry per parameter')
    else
        local value = M.number(o, 'fd_step', 1e-12, 1)
        for j = 1, n do
            steps[j] = value
        end
    end
    return steps
end

-- Private Park-Miller stream using Schrage arithmetic; Lua's global RNG is untouched.
function M.generator(seed)
    assert(M.finite(seed) and seed == math.floor(seed), 'seed must be an integer')
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

function M.domain(self, o)
    -- Search coordinates are dimensionless multiples of each parameter's step.
    -- The callback always receives the original physical units (for example V/mm).
    local start, step = self._start, self._step
    local d = { active = {}, scale = {}, lower = {}, upper = {}, zero = {}, max_step = 0 }
    for _, key in ipairs { 'lower', 'upper' } do
        if o[key] then
            assert(#o[key] == #start, key .. ' bounds have the wrong length')
        end
    end
    for j = 1, #start do
        local a, b = o.lower and o.lower[j] or -math.huge, o.upper and o.upper[j] or math.huge
        assert(M.finite(start[j]) and M.finite(step[j]), 'start and step must be finite')
        assert(
            type(a) == 'number' and type(b) == 'number' and a <= start[j] and start[j] <= b,
            'invalid bounds or start outside bounds'
        )
        local scale = math.abs(step[j])
        if scale > 0 and a < b then
            d.active[#d.active + 1] = j
            d.lower[j], d.upper[j] = (a - start[j]) / scale, (b - start[j]) / scale
        else
            scale = 0;
            d.lower[j], d.upper[j] = 0, 0
        end
        d.scale[j], d.zero[j] = scale, 0
        d.max_step = math.max(d.max_step, scale)
    end

    function d.physical(z)
        local x = {}
        for j = 1, #start do
            x[j] = start[j] + d.scale[j] * z[j]
            if o.lower then
                x[j] = math.max(o.lower[j], x[j])
            end
            if o.upper then
                x[j] = math.min(o.upper[j], x[j])
            end
        end
        return x
    end

    function d.clip(value, j)
        return math.max(d.lower[j], math.min(d.upper[j], value))
    end

    -- Called for active coordinates only: fixed coordinates have zero width.
    -- Two finite bounds reflect periodically; one-sided bounds reflect a crossing.
    -- Missing bounds remain infinite in these step-normalized coordinates.
    function d.reflect(value, j)
        local a, b = d.lower[j], d.upper[j]
        if a > -math.huge and b < math.huge then
            local width = b - a
            local offset = (value - a) % (2 * width)
            return a + (offset <= width and offset or 2 * width - offset)
        elseif value < a then
            return 2 * a - value
        elseif value > b then
            return 2 * b - value
        end
        return value
    end

    function d.available()
        return self:ncalls() < (self._maxcalls or math.huge)
    end

    function d.evaluate(z, info)
        if not d.available() then
            return nil
        end
        self._request_info = info
        local f = self:f(d.physical(z))
        return M.finite(f) and f or math.huge
    end
    return d
end
return M
