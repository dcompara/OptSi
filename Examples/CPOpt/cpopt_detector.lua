-- Native trajectory measurements for the CPOpt teaching example.
-- Account for every launched ion; missing reference hits invalidate optical scores.
-- Accounting for the virtual detector in the CPOpt workbench (mm, microseconds).
-- Independent of SIMION globals so crossing and loss handling can be tested.
-- Each initialized ion receives exactly one final outcome. A native stop may
-- mean an electrode collision or leaving the domain; do not call all of them hits.
local M = {};
M.__index = M

function M.new(c)
    assert(c.detector_radius > 0 and c.max_tof_us > 0, 'invalid detector settings')
    return setmetatable(
        {
            x = c.detector_x,
            radius2 = c.detector_radius ^ 2,
            max_tof = c.max_tof_us,
            tolerance = 1e-9,
            total = 0,
            hits = {},
            reference_hit = false,
            previous = {},
            outcome = {},
            counts = {
                detected = 0,
                detector_miss = 0,
                timed_out = 0,
                returned_upstream = 0,
                stopped_elsewhere = 0
            }
        },
        M
    )
end

function M:initialize(id, x, y, z)
    assert(not self.previous[id], 'ion initialized more than once')
    assert(x < self.x, 'CPOpt particles must start upstream of the detector')
    self.total = self.total + 1
    -- initialize can run for all ions before flight, even in ungrouped mode.
    self.previous[id] = { x, y, z }
end

function M:finish(id, reason)
    assert(self.previous[id], 'ion was not initialized inside the PA')
    if not self.outcome[id] then
        self.outcome[id] = reason
        self.counts[reason] = self.counts[reason] + 1
    end
end

function M:advance(id, x, y, z, vx, tof, native_stop)
    if self.outcome[id] then
        return false
    end
    local p = assert(self.previous[id], 'ion was not initialized inside the PA')
    local dx = x - p[1]
    if p[1] < self.x and x >= self.x - self.tolerance and dx > 0 then
        -- Interpolate the final short segment to the plane. Clamp tiny roundoff
        -- undershoots instead of extrapolating beyond the simulated segment.
        local fraction = math.max(0, math.min(1, (self.x - p[1]) / dx))
        local hy, hz = p[2] + fraction * (y - p[2]), p[3] + fraction * (z - p[3])
        if hy * hy + hz * hz <= self.radius2 then
            self.hits[#self.hits + 1] = { hy, hz }
            if id == 1 then
                self.reference_hit = true
            end
            self:finish(id, 'detected')
        else
            self:finish(id, 'detector_miss')
        end
        -- Even a native stop downstream of the plane can include a valid
        -- crossing first. An upstream electrode collision never reaches here.
        return true
    elseif native_stop then
        self:finish(id, 'stopped_elsewhere')
    elseif tof >= self.max_tof then
        self:finish(id, 'timed_out');
        return true
    elseif x <= 0 and vx < 0 then
        self:finish(id, 'returned_upstream');
        return true
    end
    self.previous[id] = { x, y, z }
    return false
end

function M:terminate(id)
    -- Catches native termination even if there was no final other_actions call.
    self:finish(id, 'stopped_elsewhere')
end

function M:check_complete()
    local ended = 0
    for _, n in pairs(self.counts) do
        ended = ended + n
    end
    assert(ended == self.total, 'some initialized ions have no final outcome')
    assert(self.counts.detected == #self.hits, 'detector count mismatch')
end

return M
