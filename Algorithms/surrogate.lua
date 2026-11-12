-- Small deterministic ordinary-kriging model for expensive, low-budget runs.
-- Squared-exponential ARD correlation, constant mean estimated by GLS,
-- concentrated maximum likelihood, Cholesky solves and latent uncertainty.
-- Rasmussen & Williams (2006), Gaussian Processes for Machine Learning, ch.2.
-- https://gaussianprocess.org/gpml/chapters/RW2.pdf
local U = require 'optimizer_utils'
local M = {}

local function dot(a, b)
    local value = 0
    for i = 1, #a do
        value = value + a[i] * b[i]
    end
    return value
end

local function kernel(a, b, length)
    local value = 0
    for j = 1, #a do
        value = value + ((a[j] - b[j]) / length[j]) ^ 2
    end
    return math.exp(-0.5 * value)
end

local function cholesky(a)
    local l = {}
    for i = 1, #a do
        l[i] = {}
        for j = 1, i do
            local v = a[i][j]
            for k = 1, j - 1 do
                v = v - l[i][k] * l[j][k]
            end
            if i == j then
                if not U.finite(v) or v <= 0 then
                    return nil
                end
                l[i][j] = math.sqrt(v)
            else
                l[i][j] = v / l[j][j]
            end
        end
    end
    return l
end

local function forward_solve(l, b)
    local y = {}
    for i = 1, #l do
        local value = b[i]
        for j = 1, i - 1 do
            value = value - l[i][j] * y[j]
        end
        y[i] = value / l[i][i]
    end
    return y
end

local function solve(l, b)
    local y = forward_solve(l, b)
    local x = {}
    for i = #l, 1, -1 do
        local value = y[i]
        for j = i + 1, #l do
            value = value - l[j][i] * x[j]
        end
        x[i] = value / l[i][i]
    end
    return x
end

-- ParEGO supplies normalized parameter points and its current scalarized costs.
-- The nugget regularizes the GLS/Cholesky fit; an unsuccessful fit returns nil.
function M.fit(points, values, options)
    local o = options or {}
    local n = #points
    assert(n == #values and n > 0, 'GP requires matching, nonempty points and values')
    local dimension = #points[1]
    local one = {}
    for i = 1, n do
        one[i] = 1
        assert(U.finite(values[i]) and #points[i] == dimension, 'invalid GP training data')
        for j = 1, dimension do
            assert(U.finite(points[i][j]), 'invalid GP training coordinate')
        end
    end
    local nugget = o.nugget or 1e-9
    assert(U.finite(nugget) and nugget > 0 and nugget < 1, 'invalid GP nugget')

    local function build(length)
        local correlation = {}
        for i = 1, n do
            correlation[i] = {}
            for j = 1, i do
                local r = kernel(points[i], points[j], length)
                correlation[i][j] = r
                if j < i then
                    correlation[j][i] = r
                end
            end
            correlation[i][i] = 1 + nugget
        end
        local l = cholesky(correlation)
        if not l then
            return nil
        end
        local invone, invvalues = solve(l, one), solve(l, values)
        local denominator = dot(one, invone)
        if denominator <= 0 then
            return nil
        end
        local average = dot(one, invvalues) / denominator
        local residual = {}
        for i = 1, n do
            residual[i] = values[i] - average
        end
        local alpha = solve(l, residual)
        local variance = math.max(dot(residual, alpha) / n, 1e-20)
        local likelihood = n * math.log(variance)
        for i = 1, n do
            likelihood = likelihood + 2 * math.log(l[i][i])
        end
        return {
            length = U.copy(length),
            l = l,
            mean = average,
            variance = variance,
            alpha = alpha,
            invone = invone,
            denominator = denominator,
            likelihood = likelihood
        }
    end
    local best

    local function consider(length)
        local candidate = build(length)
        if candidate and (not best or candidate.likelihood < best.likelihood) then
            best = candidate;
            return true
        end
        return false
    end
    -- Fixed lengths reuse supplied hyperparameters while refitting to current data.
    local fixed_length = o.fixed_length == true
    assert(
        o.fixed_length == nil or type(o.fixed_length) == 'boolean',
        'invalid GP fixed_length flag'
    )
    if o.length then
        assert(#o.length == dimension, 'GP length scale dimension mismatch')
        for j = 1, dimension do
            assert(U.finite(o.length[j]) and o.length[j] > 0, 'invalid GP length scale')
        end
        consider(o.length)
    elseif fixed_length then
        error('GP fixed_length requires supplied length scales')
    end
    if not fixed_length then
        for _, scale in ipairs { 0.1, 0.3, 1 } do
            local length = {}
            for j = 1, dimension do
                length[j] = scale
            end
            consider(length)
        end
    end
    if not best then
        return nil
    end
    local sweeps = o.sweeps or 2
    assert(U.finite(sweeps) and sweeps >= 0 and sweeps == math.floor(sweeps), 'invalid GP sweeps')
    if not fixed_length then
        for sweep = 1, sweeps do
            local changed = false
            for j = 1, dimension do
                local center = best.length[j]
                for _, factor in ipairs { 0.5, 2 } do
                    local length = U.copy(best.length)
                    length[j] = math.max(0.025, math.min(4, center * factor))
                    if consider(length) then
                        changed = true
                    end
                end
            end
            if not changed then
                break
            end
        end
    end

    -- Return the scalar mean and latent ordinary-kriging variance used by EI.
    function best:predict(point)
        local r = {}
        for i = 1, n do
            r[i] = kernel(point, points[i], self.length)
        end
        local mean = self.mean + dot(r, self.alpha)
        -- r^T K^{-1} r = ||L^{-1} r||^2, so prediction variance only
        -- needs the forward triangular solve rather than a full solve.
        local projected = forward_solve(self.l, r)
        local correction = 1 - dot(r, self.invone)
        local variance = self.variance
            * math.max(0, 1 - dot(projected, projected) + correction ^ 2 / self.denominator)
        return mean, variance
    end
    return best
end

-- Normal CDF approximation, maximum absolute error about 7.5e-8.
local function normal_cdf(x)
    local a = math.abs(x)
    local t = 1 / (1 + 0.2316419 * a)
    local tail = math.exp(-0.5 * a * a)
        / math.sqrt(2 * math.pi)
        * t
        * (
            0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
    return x >= 0 and 1 - tail or tail
end

function M.expected_improvement(mean, variance, best)
    if variance <= 1e-30 then
        return math.max(0, best - mean)
    end
    local sd = math.sqrt(variance)
    local z = (best - mean) / sd
    return math.max(
        0,
        (best - mean) * normal_cdf(z) + sd * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    )
end

-- Stable scalar-minimization logEI: log(sigma) + log(phi(z)+z*Phi(z)),
-- z=(best-mean)/sigma. Ament et al. (NeurIPS 2023), sec. 3 / Appendix A:
-- https://arxiv.org/abs/2310.20708 . This addition does not alter ParEGO's EI.
-- For z<=-1 use Laplace's continued fraction for the normal Mills ratio R(a):
-- R(a)=1/(a+1/(a+2/(a+3/...))), a=-z. If b=1/(a+2/(a+3/...)), then
-- 1-a*R(a)=b/(a+b). Evaluating this positive ratio in log space avoids both
-- cancellation and the underflow of phi(a). No erfc/DLL dependency is needed.
local LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
local function mills_remainder(a)
    local depth = math.max(24, math.ceil(400 / a / a))
    local tail = 0
    for k = depth, 2, -1 do
        tail = k / (a + tail)
    end
    return 1 / (a + tail)
end

local function log_positive_difference(a, b)
    local delta = a - b
    if delta == math.huge then
        local scale = math.max(math.abs(a), math.abs(b))
        return math.log(scale) + math.log(a / scale - b / scale)
    end
    return math.log(delta)
end

function M.log_expected_improvement(mean, variance, best)
    assert(U.finite(mean) and U.finite(best) and U.finite(variance) and variance >= 0,
        'logEI requires finite mean, incumbent and nonnegative variance')
    local improvement = best - mean
    if variance == 0 then
        return improvement > 0 and log_positive_difference(best, mean) or -math.huge
    end
    local log_sd = 0.5 * math.log(variance)
    local z = improvement / math.sqrt(variance)
    if z >= 8 then
        -- Phi(z) rounds to 1; log(improvement) also avoids overflow in z.
        return log_positive_difference(best, mean)
    elseif z <= -1 then
        local a = -z
        local b = mills_remainder(a)
        return log_sd - 0.5 * a * a - LOG_SQRT_2PI + math.log(b) - math.log(a + b)
    end
    local cdf
    if z >= 1 then
        local b = mills_remainder(z)
        cdf = 1 - math.exp(-0.5 * z * z - LOG_SQRT_2PI) / (z + b)
    else
        -- Integral of exp(-t^2/2) from 0 to z, convergent on [-1,1].
        local term, integral = z, z
        for k = 1, 20 do
            term = term * (-0.5 * z * z) / k
            integral = integral + term / (2 * k + 1)
        end
        cdf = 0.5 + math.exp(-LOG_SQRT_2PI) * integral
    end
    return log_sd + math.log(math.exp(-0.5 * z * z - LOG_SQRT_2PI) + z * cdf)
end

return M
