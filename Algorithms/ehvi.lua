-- Public EHVI: sequential independent two-objective GPs.
-- Analytical EHVI in log space, not BoTorch smoothed MC qLogEHVI.
-- Validated research evaluation/archive contract; public constructor adapter only.
local U=require 'optimizer_utils'
local Pareto=require 'multiobjective'
local Proposal=require 'ehvi_proposal'
local unpack=unpack or table.unpack
local M={};M.__index=M
local function copy(x)
    if type(x)~='table'then return x end
    local y={};for k,v in pairs(x)do y[k]=copy(v)end;return y
end
local function vector(x,n,name)
    assert(type(x)=='table'and #x==n,name..' dimension mismatch')
    local count=0
    for k,v in pairs(x)do
        assert(type(k)=='number'and k==math.floor(k)and k>=1 and k<=n and U.finite(v),'invalid '..name)
        count=count+1
    end
    assert(count==n,name..' must be dense')
end
local function key(x)
    local a={};for j,v in ipairs(x)do a[j]=string.format('%.17g',v)end;return table.concat(a,':')
end
local function shuffle(a,r)for k=#a,2,-1 do local j=math.floor(r()*k)+1;a[k],a[j]=a[j],a[k]end end
local function squared(a,b)local s=0;for j=1,#a do s=s+(a[j]-b[j])^2 end;return s end
function M:_physical(z)
    local x=U.copy(self._start)
    for i,j in ipairs(self._active)do
        x[j]=math.max(self._options.lower[j],math.min(self._options.upper[j],self._lo[i]+self._span[i]*z[i]))
    end
    return x
end
function M:_normalized(x)
    vector(x,#self._start,'design point')
    local z={}
    for j,v in ipairs(x)do
        assert(v>=self._options.lower[j]and v<=self._options.upper[j],'design point outside bounds')
        if self._domain.scale[j]==0 then assert(v==self._start[j],'design changes a fixed coordinate')end
    end
    for i,j in ipairs(self._active)do z[i]=(x[j]-self._lo[i])/self._span[i]end
    return z
end
function M:_novel(z)
    if self._seen[key(self:_physical(z))]then return false end
    for _,x in ipairs(self._attempted)do if squared(z,x)<1e-14 then return false end end
    return true
end
function M:_evaluate(z,info,x)
    if self._calls>=self._maxcalls then return false end
    x=x or self:_physical(z)
    self._calls=self._calls+1;self._test=U.copy(x)
    info.request_index=self._calls;self._info=info
    local value=self._func(unpack(x)) -- exceptions abort, never physical losses
    self._attempted[#self._attempted+1]=U.copy(z);self._seen[key(x)]=true
    if type(value)=='table'and value.valid==false then
        assert(value.reason==nil or type(value.reason)=='string','invalid physical reason')
        self._stats.invalid=self._stats.invalid+1;return true
    end
    assert(type(value)=='table'and #value==self._m,'EHVI expects an objective vector')
    local finite=true;local count=0
    for j,v in pairs(value)do
        assert(type(j)=='number'and j==math.floor(j)and j>=1 and j<=self._m and type(v)=='number','invalid objective vector')
        count=count+1;finite=finite and U.finite(v)
    end
    assert(count==self._m,'objective vector must be dense')
    if not finite then self._stats.invalid=self._stats.invalid+1;return true end
    self._points[#self._points+1]=U.copy(z);self._values[#self._values+1]=U.copy(value)
    Pareto.update_front(self._front,{x=x,objectives=value})
    return true
end
-- Positive affine normalization for fitting/selection only. Raw observations
-- and the public front are never logged or otherwise transformed in place.
function M:_scaled()
    local values,ref={},{}
    for i=1,#self._values do values[i]={}end
    if self._options.objective_scale then
        for j=1,self._m do
            local a,s=self._options.objective_offset[j],self._options.objective_scale[j]
            local function transform(v)
                local out=(v-a)/s
                if not U.finite(out)then out=v/s-a/s end
                assert(U.finite(out),'nonfinite fixed objective normalization');return out
            end
            for i,y in ipairs(self._values)do values[i][j]=transform(y[j])end
            ref[j]=transform(self._options.reference[j])
        end
        return values,ref
    end
    for j=1,self._m do
        local lo,hi,mag=math.huge,-math.huge,math.abs(self._options.reference[j])
        for _,y in ipairs(self._values)do lo=math.min(lo,y[j]);hi=math.max(hi,y[j]);mag=math.max(mag,math.abs(y[j]))end
        if mag==0 then mag=1 end
        local offset=lo/mag;local width=hi/mag-offset
        if width==0 then width=1 end
        for i,y in ipairs(self._values)do values[i][j]=(y[j]/mag-offset)/width end
        ref[j]=(self._options.reference[j]/mag-offset)/width
        assert(U.finite(ref[j]),'unrepresentable reference normalization')
    end
    return values,ref
end
M._proposal=Proposal.proposal
function M:_execute()
    local o,n=self._options,#self._active
    local random,normal=U.generator(o.seed)
    self:_evaluate(self:_normalized(self._start),{phase='initial',iteration=0},U.copy(self._start))
    if n==0 then self._status='fixed_variables';return end
    if o.initial_design then
        for i=2,#o.initial_design do
            if self._calls>=self._maxcalls then break end
            local x=o.initial_design[i]
            self:_evaluate(self:_normalized(x),{phase='initial_design',iteration=0},U.copy(x))
        end
    else
        local count=math.min(o.initial_points-1,self._maxcalls-self._calls)
        local design={};for k=1,count do design[k]={}end
        for j=1,n do
            local order={};for k=1,count do order[k]=k end;shuffle(order,random)
            for k=1,count do design[k][j]=(order[k]-random())/count end
        end
        for _,z in ipairs(design)do self:_evaluate(z,{phase='initial_design',iteration=0})end
    end
    local iteration=0
    while self._calls<self._maxcalls do
        iteration=iteration+1
        local z,info
        if #self._points>=2 then z,info=self:_proposal(random,normal,iteration)end
        if not z then
            for attempt=1,1000 do
                local trial={};for j=1,n do trial[j]=random()end
                if self:_novel(trial)then z=trial;break end
            end
            info={phase='ehvi_space_filling',iteration=iteration}
        end
        if not z then self._status='design_resolution';return end
        self:_evaluate(z,info)
    end
    self._status='budget'
end
local function resume(self,...)
    local ok,err=coroutine.resume(self._co,...)
    if not ok then self._status='error';error(err,0)end
end
local function new(t)
    assert(type(t)=='table','EHVI settings must be a table')
    local fields={algorithm=true,start=true,step=true,maxcalls=true,minradius=true,mode=true,func=true,options=true}
    for k in pairs(t)do assert(fields[k],'unknown EHVI field: '..tostring(k))end
    assert(t.algorithm==nil or t.algorithm=='EHVI','module algorithm must be EHVI')
    assert(t.mode==nil or t.mode=='multiobjective','EHVI requires multiobjective mode')
    assert(type(t.start)=='table'and #t.start>0,'start must be nonempty')
    vector(t.start,#t.start,'start')
    assert(t.options==nil or type(t.options)=='table','options must be a table')
    local supported={lower=true,upper=true,seed=true,reference=true,n_objectives=true,
        initial_points=true,initial_design=true,objective_offset=true,objective_scale=true,
        candidates=true,acquisition_starts=true,acquisition_iterations=true,gp_sweeps=true,gp_nugget=true}
    for k in pairs(t.options or{})do assert(supported[k],'unknown EHVI option: '..tostring(k))end
    assert(t.func==nil or type(t.func)=='function','func must be a function')
    local budget=t.maxcalls or 250
    assert(U.finite(budget)and budget>0 and budget==math.floor(budget),'invalid maxcalls')
    local self=setmetatable({_start=U.copy(t.start),_step={},_options=copy(t.options or{}),_maxcalls=budget,
        _func=t.func,_calls=0,_front={},_points={},_values={},_attempted={},_seen={},_info={},_status='ready',
        _stats={invalid=0,model_failures=0,surrogate_evaluations=0,model_seconds=0,acquisition_seconds=0}},M)
    if t.step then vector(t.step,#t.start,'step')end
    for j=1,#t.start do self._step[j]=t.step and t.step[j]or 1 end
    local o=U.options(self,{seed=1,candidates=256,acquisition_starts=3,acquisition_iterations=12,gp_sweeps=2,gp_nugget=1e-6})
    self._options=o
    U.generator(o.seed)
    U.number(o,'candidates',8,100000,true);U.number(o,'acquisition_starts',1,100,true)
    U.number(o,'acquisition_iterations',1,100,true);U.number(o,'gp_sweeps',0,20,true);U.number(o,'gp_nugget',1e-12,.1)
    assert(type(o.reference)=='table'and#o.reference==2,'provide fixed physical reference for 2 objectives')
    self._m=o.n_objectives or #o.reference
    assert(self._m==#o.reference and self._m==2,'n_objectives/reference mismatch')
    vector(o.reference,self._m,'reference')
    if o.objective_scale~=nil or o.objective_offset~=nil then
        if o.objective_scale==nil then o.objective_scale={};for j=1,self._m do o.objective_scale[j]=1 end end
        if o.objective_offset==nil then o.objective_offset={};for j=1,self._m do o.objective_offset[j]=0 end end
        vector(o.objective_scale,self._m,'objective_scale');vector(o.objective_offset,self._m,'objective_offset')
        for _,s in ipairs(o.objective_scale)do assert(s>0,'objective_scale must be positive')end
    end
    self._domain=U.domain(self,o);self._active=self._domain.active;self._lo,self._span={},{}
    assert(type(o.lower)=='table'and type(o.upper)=='table','EHVI requires explicit finite bounds')
    vector(o.lower,#t.start,'lower');vector(o.upper,#t.start,'upper')
    for i,j in ipairs(self._active)do
        self._lo[i],self._span[i]=o.lower[j],o.upper[j]-o.lower[j]
        assert(U.finite(self._span[i])and self._span[i]>0,'invalid interval width')
    end
    if o.initial_points==nil then o.initial_points=2*#self._active+1 end
    U.number(o,'initial_points',1,1000000,true)
    if o.initial_design~=nil then
        assert(type(o.initial_design)=='table'and #o.initial_design>0,'initial_design must include start')
        local seen={}
        for _,x in ipairs(o.initial_design)do
            self:_normalized(x);local code=key(x);assert(not seen[code],'initial_design repeats a physical point');seen[code]=true
        end
        for j,v in ipairs(t.start)do assert(o.initial_design[1][j]==v,'initial_design must begin with start exactly once')end
    end
    if not self._func then
        self._func=function()return coroutine.yield()end
        self._co=coroutine.create(function()self:_execute()end);self._status='running';resume(self)
    end
    return self
end
function M:run()
    assert(not self._co and self._status=='ready','create a fresh direct EHVI object')
    self._status='running';local ok,err=pcall(self._execute,self)
    if not ok then self._status='error';error(err,0)end
    return self:front()
end
function M:running()return self._co~=nil and coroutine.status(self._co)=='suspended'end
function M:values()assert(self._test,'no pending evaluation');return unpack(self._test)end
function M:result(value)assert(self:running(),'not awaiting feedback');resume(self,value)end
function M:front()return copy(self._front)end
function M:best()error('EHVI has no scalar best; use front()',2)end
function M:status()return self._status end
function M:ncalls()return self._calls end
function M:nevaluations()return self._calls end
function M:nrejected()return 0 end
function M:info()return copy(self._info)end
function M:options()return copy(self._options)end
function M:stats()return copy(self._stats)end
M.new=new
return setmetatable(M,{__call=function(_,t)return new(t)end})
