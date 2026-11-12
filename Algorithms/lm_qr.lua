-- Householder QR with recomputed column pivoting; pure Lua 5.1.
-- Internal LM helper. No normal equations or explicit matrix inverse.
local U = require 'optimizer_utils'
local M = {}
local abs, sqrt = math.abs, math.sqrt

local function norm_column(A, j, first)
    local scale, sum = 0, 1
    for i = first, #A do
        local v = abs(A[i][j])
        if v > 0 then
            if v > scale then
                sum, scale = 1 + sum * (scale / v)^2, v
            else
                sum = sum + (v / scale)^2
            end
        end
    end
    return scale == 0 and 0 or scale * sqrt(sum)
end

-- Rank-truncated basic solution if undamped; LM's positive damping normally
-- makes its augmented system full rank. qr_diagonal_ratio is not a condition
-- number. Inputs are copied and never changed.
function M.solve(matrix, rhs, tolerance)
    local m, n = #matrix, #matrix[1]
    assert(m >= n and #rhs == m, 'QR requires m >= n and a matching RHS')
    local A, b, permutation = {}, U.copy(rhs), {}
    for i = 1, m do
        assert(#matrix[i] == n, 'ragged QR matrix')
        A[i] = U.copy(matrix[i])
        for j = 1, n do
            if not U.finite(A[i][j]) then return nil end
        end
        if not U.finite(b[i]) then return nil end
    end
    for j = 1, n do permutation[j] = j end
    local rank, largest, smallest = 0, 0, math.huge
    for k = 1, n do
        local pivot, length = k, 0
        for j = k, n do
            local v = norm_column(A, j, k)
            if v > length then pivot, length = j, v end
        end
        if k == 1 then largest = length end
        if not U.finite(length) then return nil end
        if length == 0 or length <= (tolerance or 1e-12) * largest then break end
        for i = 1, m do A[i][k], A[i][pivot] = A[i][pivot], A[i][k] end
        permutation[k], permutation[pivot] = permutation[pivot], permutation[k]
        -- Normalize before forming the reflector to avoid x1 +/- norm overflow.
        local u = A[k][k] / length
        local sign = u >= 0 and 1 or -1
        local denominator, tau = u + sign, 1 + abs(u)
        local v = {[k] = 1}
        for i = k + 1, m do v[i] = (A[i][k] / length) / denominator end
        for j = k + 1, n do
            local dot = A[k][j]
            for i = k + 1, m do dot = dot + v[i] * A[i][j] end
            dot = tau * dot
            A[k][j] = A[k][j] - dot
            for i = k + 1, m do A[i][j] = A[i][j] - v[i] * dot end
        end
        local dot = b[k]
        for i = k + 1, m do dot = dot + v[i] * b[i] end
        dot = tau * dot
        b[k] = b[k] - dot
        for i = k + 1, m do b[i] = b[i] - v[i] * dot end
        A[k][k] = -sign * length
        for i = k + 1, m do A[i][k] = 0 end
        rank, smallest = k, math.min(smallest, length)
    end
    local x, reduced = {}, {}
    for j = 1, n do reduced[j] = 0 end
    for j = rank, 1, -1 do
        local value = b[j]
        for k = j + 1, rank do value = value - A[j][k] * reduced[k] end
        reduced[j] = value / A[j][j]
        if not U.finite(reduced[j]) then return nil end
    end
    for j = 1, n do x[permutation[j]] = reduced[j] end
    return x, {rank = rank, qr_diagonal_ratio = rank == 0 and 0 or smallest / largest,
        permutation = permutation}
end
return M
