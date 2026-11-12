-- Local ARD rational-quadratic GP for deterministic BADS.
-- Empirical MAP priors: Acerbi & Ma (2017), Supplement A.3 / Table S1.
-- https://arxiv.org/html/1705.04405v2
-- The constant mean is profiled analytically with its Gaussian prior;
-- positive hyperparameters are fitted in log space by bounded pattern search.
local U = require 'optimizer_utils'
local M = {}

local function dot(a, b)
    local v = 0;
    for i = 1, #a do
        v = v + a[i] * b[i]
    end;
    return v
end

function M.cholesky(a)
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

local function solve(l, b)
    local y, x = {}, {}
    for i = 1, #l do
        local v = b[i];
        for j = 1, i - 1 do
            v = v - l[i][j] * y[j]
        end;
        y[i] = v / l[i][i]
    end
    for i = #l, 1, -1 do
        local v = y[i];
        for j = i + 1, #l do
            v = v - l[j][i] * x[j]
        end;
        x[i] = v / l[i][i]
    end
    return x
end

local function quantile(sorted, p)
    local index = 1 + (#sorted - 1) * p
    local low = math.floor(index)
    return sorted[low] + (index - low) * (sorted[math.min(#sorted, low + 1)] - sorted[low])
end

local function correlation(a, b, length, alpha)
    local r = 0
    for j = 1, #a do
        r = r + ((a[j] - b[j]) / length[j]) ^ 2
    end
    return (1 + r / (2 * alpha)) ^( -alpha)
end

-- Internal BADS coordinates and scaled scalar costs enter the rational-quadratic
-- MAP fit. Positive hyperparameters use log coordinates; failed fits return nil.
function M.fit(points, values, o)
    local count, n = #points, #points[1]
    local sorted = U.copy(values);
    table.sort(sorted)
    local offset = quantile(sorted, 0.5)
    local meanprior = quantile(sorted, 0.9) - offset
    local meansd = math.max(math.abs(meanprior) / 5, 1e-6)
    local y, one, avg = {}, {}, 0
    for k = 1, count do
        y[k], one[k] = values[k] - offset, 1;
        avg = avg + y[k] / count
    end
    local sd = 0
    for k = 1, count do
        sd = sd + (y[k] - avg) ^ 2
    end
    sd = math.max(1e-3, math.sqrt(sd / math.max(1, count - 1)))
    local rmin, rmax = math.huge, 0
    for i = 1, count do
        for j = 1, i - 1 do
            local r = 0
            for h = 1, n do
                r = r + (points[i][h] - points[j][h]) ^ 2
            end
            r = math.sqrt(r)
            if r > 0 then
                rmin, rmax = math.min(rmin, r), math.max(rmax, r)
            end
        end
    end
    if rmax == 0 then
        rmin, rmax = math.max(o.poll, 1e-6), math.max(2 * o.poll, 2e-6)
    end
    local logcenter = (math.log(rmin) + math.log(rmax)) / 2
    local logspread = math.max(0.5, (math.log(rmax) - math.log(rmin)) / 2)
    local priors, prior_sd, lower, upper = {}, {}, {}, {}
    for j = 1, n do
        priors[j], prior_sd[j] = logcenter, logspread
        lower[j], upper[j] = math.log(o.min_poll or 1e-6), math.log(o.ranges[j])
    end
    priors[n + 1], prior_sd[n + 1] = math.log(sd), 2
    lower[n + 1], upper[n + 1] = math.log(1e-3), math.log(1e9)
    priors[n + 2], prior_sd[n + 2] = 1, 1
    lower[n + 2], upper[n + 2] = -5, 5
    priors[n + 3], prior_sd[n + 3] = math.log(math.sqrt(1e-3 * o.poll)), 1
    lower[n + 3], upper[n + 3] = math.log(4e-4), math.log(150)

    local function build(theta)
        local length = {}
        for j = 1, n do
            length[j] = math.exp(theta[j])
        end
        local signal, shape, noise =
            math.exp(2 * theta[n + 1]), math.exp(theta[n + 2]), math.exp(2 * theta[n + 3])
        local a = {}
        for i = 1, count do
            a[i] = {}
            for j = 1, i do
                local v = signal * correlation(points[i], points[j], length, shape)
                a[i][j] = v
                if j < i then
                    a[j][i] = v
                end
            end
            a[i][i] = a[i][i] + noise + 1e-12 * signal
        end
        local l = M.cholesky(a)
        if not l then
            return nil
        end
        local iy, ione = solve(l, y), solve(l, one)
        local precision = 1 / (meansd * meansd)
        local m = (dot(one, iy) + meanprior * precision) / (dot(one, ione) + precision)
        local residual = {}
        for k = 1, count do
            residual[k] = y[k] - m
        end
        local coefficients = solve(l, residual)
        local score = 0.5 * dot(residual, coefficients) + 0.5 * ((m - meanprior) / meansd) ^ 2
        for k = 1, count do
            score = score + math.log(l[k][k])
        end
        for j = 1, #theta do
            score = score + 0.5 * ((theta[j] - priors[j]) / prior_sd[j]) ^ 2
        end
        if not U.finite(score) then
            return nil
        end
        return {
            theta = U.copy(theta),
            length = length,
            shape = shape,
            signal = signal,
            noise = noise,
            mean = m + offset,
            l = l,
            coefficients = coefficients,
            score = score,
            points = points,
            values = values
        }
    end
    local best

    local function consider(theta)
        for j = 1, #theta do
            theta[j] = math.max(lower[j], math.min(upper[j], theta[j]))
        end
        local model = build(theta)
        if model and (not best or model.score < best.score) then
            best = model;
            return true
        end
        return false
    end
    if o.previous then
        consider(U.copy(o.previous.theta))
    end
    if o.refit or not best then
        consider(U.copy(priors))
    end
    if not best then
        return nil
    end
    if o.refit then
        local mesh = 0.8
        for _ = 1, (o.sweeps or 3) do
            local changed = false
            for j = 1, n + 3 do
                local center = best.theta[j]
                for _, sign in ipairs { -1, 1 } do
                    local theta = U.copy(best.theta);
                    theta[j] = center + sign * mesh
                    if consider(theta) then
                        changed = true
                    end
                end
            end
            if not changed then
                mesh = mesh * 0.5
            end
        end
    end

    -- Return scalar mean and latent variance. The fitted diagonal noise term
    -- regularizes training; it is not an extra measured objective component.
    function best:predict(x)
        local k = {}
        for i = 1, count do
            k[i] = self.signal * correlation(x, points[i], self.length, self.shape)
        end
        local mean = self.mean + dot(k, self.coefficients)
        -- Only a forward Cholesky solve is needed for k' K^-1 k.
        local z, correction = {}, 0
        for i = 1, count do
            local v = k[i]
            for j = 1, i - 1 do
                v = v - self.l[i][j] * z[j]
            end
            z[i] = v / self.l[i][i];
            correction = correction + z[i] * z[i]
        end
        return mean, math.max(0, self.signal - correction)
    end
    return best
end
return M
