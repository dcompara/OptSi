-- Full-covariance (mu/mu_w,lambda)-CMA-ES, positive recombination weights.
-- Hansen, The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772,
-- Appendix C (rank-one + rank-mu update, cumulative step-size adaptation).
-- https://arxiv.org/abs/1604.00772
-- Coordinates are scaled by abs(step); zero steps fix a coordinate.
-- Optional bounds use reflection; the repaired displacement is used in CMA.
-- No external linear algebra package or change to Lua's global RNG is needed.
local U = require 'optimizer_utils'
local M = {}

local function identity(n)
    local a = {}
    for i = 1, n do
        a[i] = {}
        for j = 1, n do
            a[i][j] = i == j and 1 or 0
        end
    end
    return a
end

-- Symmetric Jacobi eigensolver. Columns of b are orthonormal eigenvectors.
-- Keeping the full eigensystem is necessary for the symmetric C^(-1/2).
-- Return nil for a nonpositive/nonfinite eigensystem or exhausted Jacobi sweeps;
-- the caller reports the numerical limit rather than continuing with that model.
local function eigen(c)
    local n = #c
    local a, b = {}, identity(n)
    for i = 1, n do
        a[i] = U.copy(c[i])
    end
    for iteration = 1, math.max(1, 80 * n * n) do
        local p, q, off, scale = 1, 1, 0, 0
        for i = 1, n do
            scale = math.max(scale, math.abs(a[i][i]))
            for j = i + 1, n do
                if math.abs(a[i][j]) > off then
                    p, q, off = i, j, math.abs(a[i][j])
                end
            end
        end
        if off <= 1e-14 * math.max(scale, 1e-300) then
            local axes = {}
            for i = 1, n do
                if not U.finite(a[i][i]) or a[i][i] <= 0 then
                    return nil
                end
                axes[i] = math.sqrt(a[i][i])
            end
            return b, axes
        end
        local tau = (a[q][q] - a[p][p]) / (2 * a[p][q])
        local sign = tau >= 0 and 1 or -1
        local t = sign / (math.abs(tau) + math.sqrt(1 + tau * tau))
        local cosine = 1 / math.sqrt(1 + t * t)
        local sine = t * cosine
        local apq = a[p][q]
        a[p][p], a[q][q] = a[p][p] - t * apq, a[q][q] + t * apq
        a[p][q], a[q][p] = 0, 0
        for k = 1, n do
            if k ~= p and k ~= q then
                local u, v = a[k][p], a[k][q]
                a[k][p], a[p][k] = cosine * u - sine * v, cosine * u - sine * v
                a[k][q], a[q][k] = sine * u + cosine * v, sine * u + cosine * v
            end
            local u, v = b[k][p], b[k][q]
            b[k][p], b[k][q] = cosine * u - sine * v, sine * u + cosine * v
        end
    end
    return nil
end

function M.run(self)
    local o = U.options(self, { seed = 1, sigma = 1, max_iterations = 100000 })
    local d = U.domain(self, o)
    local n = #d.active
    if o.population == nil then
        o.population = 4 + math.floor(3 * math.log(math.max(1, n)))
    end
    local lambda = U.number(o, 'population', 2, math.huge, true)
    local sigma = U.number(o, 'sigma', 1e-15, math.huge)
    U.number(o, 'max_iterations', 1, math.huge, true)
    local _, normal = U.generator(o.seed)
    self._radius = sigma * d.max_step
    if not d.available() then
        return
    end
    d.evaluate(d.zero, { phase = 'initial', iteration = 0, search_scale = sigma })
    if n == 0 then
        self._radius = 0;
        self._stop_reason = 'fixed_variables';
        return
    end

    local mu = math.floor(lambda / 2)
    local weights, sum, sumsq = {}, 0, 0
    for k = 1, mu do
        weights[k] = math.log(mu + 0.5) - math.log(k);
        sum = sum + weights[k]
    end
    for k = 1, mu do
        weights[k] = weights[k] / sum;
        sumsq = sumsq + weights[k] ^ 2
    end
    local mueff = 1 / sumsq
    local cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    local cs = (mueff + 2) / (n + mueff + 5)
    local c1 = 2 / ((n + 1.3) ^ 2 + mueff)
    local cmu = math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ^ 2 + mueff))
    local damping = 1 + 2 * math.max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    local chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    local mean, pc, ps = {}, {}, {}
    for i = 1, n do
        mean[i], pc[i], ps[i] = 0, 0, 0
    end
    local covariance = identity(n)
    local generation = 0
    while d.available() and generation < o.max_iterations do
        local b, axes = eigen(covariance)
        if not b then
            self._stop_reason = 'covariance_numerical_limit';
            break
        end
        local smallest, largest = math.huge, 0
        for i = 1, n do
            smallest = math.min(smallest, axes[i]);
            largest = math.max(largest, axes[i])
        end
        if largest / smallest > 1e7 then
            self._stop_reason = 'covariance_condition_limit';
            break
        end
        local radius = 0
        for i, j in ipairs(d.active) do
            radius = math.max(radius, sigma * d.scale[j] * math.sqrt(covariance[i][i]))
        end
        self._radius = radius
        if radius < (self._minradius or 1e-6) then
            self._stop_reason = 'radius';
            break
        end
        generation = generation + 1
        local population = {}
        for k = 1, lambda do
            if not d.available() then
                break
            end
            local z, y, x, full = {}, {}, {}, U.copy(d.zero)
            for i = 1, n do
                z[i] = normal()
            end
            for i, j in ipairs(d.active) do
                local delta = 0
                for h = 1, n do
                    delta = delta + b[i][h] * axes[h] * z[h]
                end
                x[i] = d.reflect(mean[i] + sigma * delta, j)
                if not U.finite(x[i]) then
                    self._stop_reason = 'numerical_limit';
                    return
                end
                y[i] = (x[i] - mean[i]) / sigma
                full[j] = x[i]
            end
            local value = d.evaluate(
                full,
                {
                    phase = 'cma_population',
                    iteration = generation,
                    member = k,
                    population = lambda,
                    search_scale = sigma
                }
            )
            population[#population + 1] = { x = x, y = y, f = value, index = k }
        end
        -- An incomplete last generation still contributes measured best points,
        -- but does not train a covariance matrix with unbalanced weights.
        if #population < lambda then
            break
        end
        table.sort(population, function(a, c)
            return a.f < c.f or (a.f == c.f and a.index < c.index)
        end)
        local shift = {}
        for i = 1, n do
            shift[i] = 0
            for k = 1, mu do
                shift[i] = shift[i] + weights[k] * population[k].y[i]
            end
            mean[i] = mean[i] + sigma * shift[i]
        end
        local whitened = {}
        for i = 1, n do
            whitened[i] = 0
        end
        for k = 1, n do
            local projection = 0
            for i = 1, n do
                projection = projection + b[i][k] * shift[i]
            end
            for i = 1, n do
                whitened[i] = whitened[i] + b[i][k] * projection / axes[k]
            end
        end
        local psnorm = 0
        for i = 1, n do
            ps[i] = (1 - cs) * ps[i] + math.sqrt(cs * (2 - cs) * mueff) * whitened[i]
            psnorm = psnorm + ps[i] * ps[i]
        end
        psnorm = math.sqrt(psnorm)
        local hsig = psnorm / math.sqrt(1 - (1 - cs) ^ (2 * generation)) / chi < 1.4 + 2 / (n + 1) and 1
            or 0
        for i = 1, n do
            pc[i] = (1 - cc) * pc[i] + hsig * math.sqrt(cc * (2 - cc) * mueff) * shift[i]
        end
        for i = 1, n do
            for j = i, n do
                local rankmu = 0
                for k = 1, mu do
                    rankmu = rankmu + weights[k] * population[k].y[i] * population[k].y[j]
                end
                local value = (1 - c1 - cmu + c1 * (1 - hsig) * cc * (2 - cc)) * covariance[i][j]
                    + c1 * pc[i] * pc[j]
                    + cmu * rankmu
                covariance[i][j], covariance[j][i] = value, value
            end
        end
        sigma = sigma * math.exp(math.min(1, (cs / damping) * (psnorm / chi - 1)))
        if not U.finite(sigma) then
            self._stop_reason = 'numerical_limit';
            break
        end
        -- Preserve adaptation state for offline inspection; it is not a new request.
        self._cma_state = {
            mean = U.copy(mean),
            covariance = covariance,
            sigma = sigma,
            pc = U.copy(pc),
            ps = U.copy(ps),
            generation = generation,
            active = U.copy(d.active)
        }
    end
    if not self._stop_reason then
        self._stop_reason = d.available() and 'max_iterations' or 'budget'
    end
end

return M
