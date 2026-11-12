-- Independent ordinary-kriging objective models + bounded log-EHVI maximization.
-- Model calculations only; physical evaluations belong to the outer object.
local U=require 'optimizer_utils'
local GP=require 'surrogate'
local EHVI=require 'ehvi_math'
local M={}
function M.proposal(self,random,normal,iteration)
    local o,n=self._options,#self._active
    local began=os.clock();local scaled,reference=self:_scaled();local models={}
    for j=1,2 do
        local y={};for i,v in ipairs(scaled)do y[i]=v[j]end
        models[j]=GP.fit(self._points,y,{nugget=o.gp_nugget,sweeps=o.gp_sweeps})
        if not models[j]then
            self._stats.model_failures=self._stats.model_failures+1
            self._stats.model_seconds=self._stats.model_seconds+os.clock()-began
            return nil
        end
    end
    local boxes=EHVI.partition(scaled,reference)
    self._stats.model_seconds=self._stats.model_seconds+os.clock()-began
    began=os.clock()
    local function score(z)
        if not self:_novel(z)then return -math.huge end
        self._stats.surrogate_evaluations=self._stats.surrogate_evaluations+1
        local mean,variance={},{}
        for j=1,2 do mean[j],variance[j]=models[j]:predict(z)end
        return EHVI.log_from_boxes(boxes,mean,variance)
    end
    local candidates={}
    local function consider(z)
        if self:_novel(z)then candidates[#candidates+1]={x=z,value=score(z),index=#candidates+1}end
    end
    -- Fixed bounded global random pool, plus local perturbations of observed
    -- points and axis bounds. No model value is stored as an observation.
    for k=1,o.candidates do
        local z={};local base=self._points[math.floor(random()*#self._points)+1]
        for j=1,n do
            if n==1 and k%2==1 then z[j]=(k-1)/(o.candidates-1)
            elseif k%2==1 then local t=(base[j]+.2*normal())%2;z[j]=t<=1 and t or 2-t
            else z[j]=random()end
        end
        consider(z)
    end
    for j=1,n do
        for _,bound in ipairs{0,1}do local z=U.copy(self._points[#self._points]);z[j]=bound;consider(z)end
    end
    table.sort(candidates,function(a,b)return a.value>b.value or(a.value==b.value and a.index<b.index)end)
    local chosen,best=nil,-math.huge
    for k=1,math.min(o.acquisition_starts,#candidates)do
        local center,value,mesh=U.copy(candidates[k].x),candidates[k].value,.2
        for _=1,o.acquisition_iterations do
            local improved=false
            for j=1,n do
                for _,sign in ipairs{-1,1}do
                    local z=U.copy(center);z[j]=math.max(0,math.min(1,z[j]+sign*mesh))
                    local v=score(z)
                    if v>value then center,value,improved=z,v,true end
                end
            end
            if not improved then mesh=mesh/2 end
        end
        if not chosen or value>best then chosen,best=center,value end
    end
    self._stats.acquisition_seconds=self._stats.acquisition_seconds+os.clock()-began
    return chosen,{phase='lua_analytical_logehvi',iteration=iteration,training_points=#scaled,
        log_ehvi=best,boxes=#boxes}
end
return M
