-- Public analytical q=1 EHVI for independent minimized Gaussian costs.
-- Mathematical reference: Emmerich, Deutz and Klinkenberg (2011),
-- doi:10.1109/CEC.2011.5949880; see ../SCIENTIFIC_REFERENCES.md.
-- Integrate P(Y1<=t1)*P(Y2<=t2) over the nondominated staircase below ref.
-- This is log(analytical EHVI), NOT smoothed Monte Carlo qLogEHVI.
local U=require 'optimizer_utils'
local GP=require 'surrogate'
local M={}
local NEG=-math.huge
local LOG2PI=.5*math.log(2*math.pi)
local function log1p(x)
    if math.abs(x)>.01 then return math.log(1+x)end
    local term,total=x,x
    for k=2,24 do term=term*(-x);total=total+term/k end
    return total
end
local function expm1(x)
    if math.abs(x)>.01 then return math.exp(x)-1 end
    local term,total=x,x
    for k=2,18 do term=term*x/k;total=total+term end
    return total
end
local function add(a,b)
    if a==NEG then return b elseif b==NEG then return a end
    if a<b then a,b=b,a end
    return a+log1p(math.exp(b-a))
end
local function logcdf(z)
    if z<=-1 then
        local a=-z;local tail=0
        for k=math.max(24,math.ceil(400/a/a)),1,-1 do tail=k/(a+tail)end
        return -.5*a*a-LOG2PI-math.log(a+tail)
    elseif z>=1 then
        return log1p(-math.exp(logcdf(-z)))
    end
    local term,total=z,z
    for k=1,20 do term=term*(-.5*z*z)/k;total=total+term/(2*k+1)end
    return math.log(.5+math.exp(-LOG2PI)*total)
end
-- 16-point Gauss-Legendre, used only for nearly equal scalar-EI logs.
-- In that regime CDF varies little across the interval, or is nearly one.
local nodes={.09501250983763744,.2816035507792589,.4580167776572274,.6178762444026438,
    .755404408355003,.8656312023878318,.9445750230732326,.9894009349916499}
local weights={.1894506104550685,.1826034150449236,.1691565193950025,.1495959888165767,
    .1246289712555339,.09515851168249278,.06225352393864789,.027152459411754095}
local function interval(a,b,mu,variance)
    if a>=b then return NEG end
    local hi=GP.log_expected_improvement(mu,variance,b)
    if a==NEG then return hi end
    if variance==0 then
        local width=b-math.max(a,mu)
        return width>0 and math.log(width)or NEG
    end
    local lo=GP.log_expected_improvement(mu,variance,a)
    local delta=lo-hi
    if delta<-.0001 then return hi+math.log(-expm1(delta))end
    local half=(b-a)/2;local middle=a+half;local sd=math.sqrt(variance)
    assert(U.finite(half)and half>0,'EHVI interval width is unrepresentable; rescale objectives')
    local total=NEG
    for k,node in ipairs(nodes)do
        total=add(total,math.log(weights[k])+logcdf((middle-half*node-mu)/sd))
        total=add(total,math.log(weights[k])+logcdf((middle+half*node-mu)/sd))
    end
    return math.log(half)+total
end
local function vector(v,name)
    assert(type(v)=='table'and #v==2 and U.finite(v[1])and U.finite(v[2]),'invalid EHVI '..name)
    local count=0;for k in pairs(v)do assert(k==1 or k==2,'EHVI vectors must be dense');count=count+1 end
    assert(count==2,'EHVI vectors must be dense')
end
function M.partition(points,reference)
    vector(reference,'reference');local sorted={}
    for _,p in ipairs(points)do
        vector(p,'front point')
        if p[1]<reference[1]and p[2]<reference[2]then sorted[#sorted+1]={p[1],p[2]}end
    end
    table.sort(sorted,function(a,b)return a[1]<b[1]or(a[1]==b[1]and a[2]<b[2])end)
    local boxes,left,height={},NEG,reference[2]
    for _,p in ipairs(sorted)do
        if p[2]<height then
            if p[1]>left then boxes[#boxes+1]={left,p[1],height}end
            left,height=p[1],p[2]
        end
    end
    if left<reference[1]then boxes[#boxes+1]={left,reference[1],height}end
    return boxes
end
function M.log_from_boxes(boxes,mean,variance)
    vector(mean,'mean');vector(variance,'variance')
    assert(variance[1]>=0 and variance[2]>=0,'negative EHVI variance')
    local total=NEG
    for _,b in ipairs(boxes)do
        total=add(total,interval(b[1],b[2],mean[1],variance[1])+
            GP.log_expected_improvement(mean[2],variance[2],b[3]))
    end
    return total
end
function M.log_ehvi(points,reference,mean,variance)
    return M.log_from_boxes(M.partition(points,reference),mean,variance)
end
M.log_interval=interval -- numerical validation hook; no physical evaluation
return M
