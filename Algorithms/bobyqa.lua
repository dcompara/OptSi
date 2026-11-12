--[[
/* Copyright (c) 2009 M. J. D. Powell (mjdp@cam.ac.uk)
 * Modifications Copyright (c) 2010 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
 */


]]
-- Lua 5.1 translation of Powell's BOBYQA numerical routines as maintained in NLopt.
-- Upstream commit: 6e6593f131ba3a38bc9edbed0a357bc01526e54b
-- Source: https://github.com/stevengj/nlopt/blob/6e6593f131ba3a38bc9edbed0a357bc01526e54b/src/algs/bobyqa/bobyqa.c
-- C source SHA256: a1733948888e927a861197b9c8f4a84b561439fbe286d8020e6078e10135d95d
-- The complete numerical routines were translated mechanically before release;
-- pointer aliasing, integer arithmetic and control flow are preserved in Lua.
-- Development validation compared 370 evaluations in 12 representative cases
-- against the pinned C source: corresponding points and values agreed within
-- 1e-10 (maximum measured relative difference approximately 1e-31).
-- This is representative differential validation, not an exhaustive guarantee.
-- Runtime dependencies: Lua 5.1 and optimizer_utils.lua only. No C/Python backend.
-- See ALGORITHM_DETAILS_AND_VALIDATION.md for source and redistribution notices.


local U=require 'optimizer_utils'
local M={}
local function trunc(x) return x<0 and math.ceil(x) or math.floor(x) end
local function cmod(x,y) return x-trunc(x/y)*y end
local function truth(x) return x~=nil and x~=false and x~=0 end
-- Views preserve aliasing of Fortran work arrays and C addresses of scalars.
local pointer={}
pointer.__index=function(p,index)
    local offset=rawget(p,'offset')
    if type(offset)=='string' then
        assert(index==0,'scalar pointer arithmetic')
        return rawget(p,'data')[offset]
    end
    return rawget(p,'data')[offset+index]
end
pointer.__newindex=function(p,index,value)
    local offset=rawget(p,'offset')
    if type(offset)=='string' then
        assert(index==0,'scalar pointer arithmetic')
        rawget(p,'data')[offset]=value
    else rawget(p,'data')[offset+index]=value end
end
local function ptr(data,offset) return setmetatable({data=data,offset=offset or 0},pointer) end
local function add(p,offset)
    if offset==0 then return p end
    return ptr(rawget(p,'data'),rawget(p,'offset')+offset)
end
local function array(n)
    local data={}
    for i=0,n-1 do data[i]=0 end
    return ptr(data)
end
local NLOPT_SUCCESS=1
local NLOPT_MINF_MAX_REACHED=2
local NLOPT_FTOL_REACHED=3
local NLOPT_XTOL_REACHED=4
local NLOPT_MAXEVAL_REACHED=5
local NLOPT_MAXTIME_REACHED=6
local NLOPT_ROUNDOFF_LIMITED=-4
local NLOPT_FORCED_STOP=-5
local function nlopt_stop_forced(stop) return stop.forced or false end
local function nlopt_stop_evals(stop) return stop.nevals_p[0]>=stop.maxeval end
local function nlopt_stop_time(stop) return false end
local function nlopt_stop_ftol(stop,a,b) return false end
local function nlopt_isinf(x) return x==math.huge or x==-math.huge end


local update_, rescue_, altmov_, trsbox_, prelim_, bobyqb_

update_ = function(n, npt, bmat, zmat, ndim, vlag, beta, denom, knew, w)
  local v = {}
  v.n = n
  v.npt = npt
  v.bmat = bmat
  v.zmat = zmat
  v.ndim = ndim
  v.vlag = vlag
  v.beta = beta
  v.denom = denom
  v.knew = knew
  v.w = w
  local blocks, result = {}, nil
  blocks[1] = function()
    if truth((v.j <= v.i__2)) then return 13 else return nil end
  end
  blocks[2] = function()
    v.j = v.j + (1)
    return 1
  end
  blocks[3] = function()
    if truth((v.i__ <= v.i__1)) then return 7 else return 2 end
  end
  blocks[4] = function()
    v.i__ = v.i__ + (1)
    return 3
  end
  blocks[5] = function()
    v.bmat[(v.jp + ((v.i__ - v.npt[0]) * v.bmat_dim1))] = v.bmat[(v.i__ + (v.j * v.bmat_dim1))]
    return 4
  end
  blocks[7] = function()
    v.bmat[(v.i__ + (v.j * v.bmat_dim1))] = ((v.bmat[(v.i__ + (v.j * v.bmat_dim1))] + (v.tempa * v.vlag[v.i__])) + (v.tempb * v.w[v.i__]))
    if truth((v.i__ > v.npt[0])) then return 5 else return 4 end
  end
  blocks[13] = function()
    v.jp = (v.npt[0] + v.j)
    v.w[v.jp] = v.bmat[(v.knew[0] + (v.j * v.bmat_dim1))]
    v.tempa = (((v.alpha * v.vlag[v.jp]) - (v.tau * v.w[v.jp])) / v.denom[0])
    v.tempb = ((((-v.beta[0]) * v.w[v.jp]) - (v.tau * v.vlag[v.jp])) / v.denom[0])
    v.i__1 = v.jp
    v.i__ = 1
    return 3
  end
  blocks[15] = function()
    v.i__2 = v.n[0]
    v.j = 1
    return 1
  end
  blocks[16] = function()
    if truth((v.i__ <= v.i__2)) then return 18 else return 15 end
  end
  blocks[18] = function()
    v.zmat[(v.i__ + v.zmat_dim1)] = ((v.tempa * v.zmat[(v.i__ + v.zmat_dim1)]) - (v.tempb * v.vlag[v.i__]))
    v.i__ = v.i__ + (1)
    return 16
  end
  blocks[26] = function()
    v.alpha = v.w[v.knew[0]]
    v.tau = v.vlag[v.knew[0]]
    v.vlag[v.knew[0]] = (v.vlag[v.knew[0]] - v.one)
    v.temp = math.sqrt(v.denom[0])
    v.tempb = (v.zmat[(v.knew[0] + v.zmat_dim1)] / v.temp)
    v.tempa = (v.tau / v.temp)
    v.i__2 = v.npt[0]
    v.i__ = 1
    return 16
  end
  blocks[27] = function()
    if truth((v.i__ <= v.i__2)) then return 29 else return 26 end
  end
  blocks[29] = function()
    v.w[v.i__] = (v.zmat[(v.knew[0] + v.zmat_dim1)] * v.zmat[(v.i__ + v.zmat_dim1)])
    v.i__ = v.i__ + (1)
    return 27
  end
  blocks[31] = function()
    v.i__2 = v.npt[0]
    v.i__ = 1
    return 27
  end
  blocks[32] = function()
    if truth((v.j <= v.i__2)) then return 47 else return 31 end
  end
  blocks[34] = function()
    v.zmat[(v.knew[0] + (v.j * v.zmat_dim1))] = v.zero
    v.j = v.j + (1)
    return 32
  end
  blocks[35] = function()
    if truth((v.i__ <= v.i__1)) then return 39 else return 34 end
  end
  blocks[39] = function()
    v.temp = ((v.tempa * v.zmat[(v.i__ + v.zmat_dim1)]) + (v.tempb * v.zmat[(v.i__ + (v.j * v.zmat_dim1))]))
    v.zmat[(v.i__ + (v.j * v.zmat_dim1))] = ((v.tempa * v.zmat[(v.i__ + (v.j * v.zmat_dim1))]) - (v.tempb * v.zmat[(v.i__ + v.zmat_dim1)]))
    v.zmat[(v.i__ + v.zmat_dim1)] = v.temp
    v.i__ = v.i__ + (1)
    return 35
  end
  blocks[46] = function()
    v.d__1 = v.zmat[(v.knew[0] + v.zmat_dim1)]
    v.d__2 = v.zmat[(v.knew[0] + (v.j * v.zmat_dim1))]
    v.temp = math.sqrt(((v.d__1 * v.d__1) + (v.d__2 * v.d__2)))
    v.tempa = (v.zmat[(v.knew[0] + v.zmat_dim1)] / v.temp)
    v.tempb = (v.zmat[(v.knew[0] + (v.j * v.zmat_dim1))] / v.temp)
    v.i__1 = v.npt[0]
    v.i__ = 1
    return 35
  end
  blocks[47] = function()
    if truth(((function() v.d__1 = v.zmat[(v.knew[0] + (v.j * v.zmat_dim1))]; return math.abs(v.d__1) end)() > v.ztest)) then return 46 else return 34 end
  end
  blocks[50] = function()
    v.ztest = (v.ztest * 1e-20)
    v.i__2 = v.nptm
    v.j = 2
    return 32
  end
  blocks[51] = function()
    if truth((v.k <= v.i__1)) then return 58 else return 50 end
  end
  blocks[52] = function()
    v.k = v.k + (1)
    return 51
  end
  blocks[53] = function()
    if truth((v.j <= v.i__2)) then return 56 else return 52 end
  end
  blocks[56] = function()
    v.d__2 = v.ztest; v.d__3 = (function() v.d__1 = v.zmat[(v.k + (v.j * v.zmat_dim1))]; return math.abs(v.d__1) end)()
    v.ztest = math.max(v.d__2, v.d__3)
    v.j = v.j + (1)
    return 53
  end
  blocks[58] = function()
    v.i__2 = v.nptm
    v.j = 1
    return 53
  end
  blocks[94] = function()
    v.zmat_dim1 = v.npt[0]
    v.zmat_offset = (1 + v.zmat_dim1)
    v.zmat = add(v.zmat,(-v.zmat_offset))
    v.bmat_dim1 = v.ndim[0]
    v.bmat_offset = (1 + v.bmat_dim1)
    v.bmat = add(v.bmat,(-v.bmat_offset))
    v.vlag = add(v.vlag,-1)
    v.w = add(v.w,-1)
    v.one = 1.
    v.zero = 0.
    v.nptm = ((v.npt[0] - v.n[0]) - 1)
    v.ztest = v.zero
    v.i__1 = v.npt[0]
    v.k = 1
    return 51
  end
  local pc = 94
  while pc do pc = blocks[pc]() end
  return result
end

rescue_ = function(n, npt, xl, xu, stop, calfun, calfun_data, xbase, xpt, fval, xopt, gopt, hq, pq, bmat, zmat, ndim, sl, su, delta, kopt, vlag, ptsaux, ptsid, w)
  local v = {}
  v.n = n
  v.npt = npt
  v.xl = xl
  v.xu = xu
  v.stop = stop
  v.calfun = calfun
  v.calfun_data = calfun_data
  v.xbase = xbase
  v.xpt = xpt
  v.fval = fval
  v.xopt = xopt
  v.gopt = gopt
  v.hq = hq
  v.pq = pq
  v.bmat = bmat
  v.zmat = zmat
  v.ndim = ndim
  v.sl = sl
  v.su = su
  v.delta = delta
  v.kopt = kopt
  v.vlag = vlag
  v.ptsaux = ptsaux
  v.ptsid = ptsid
  v.w = w
  local blocks, result = {}, nil
  blocks[1] = function()
    result = NLOPT_SUCCESS; return nil
  end
  blocks[2] = function()
    if truth((v.kpt <= v.i__1)) then return 115 else return 1 end
  end
  blocks[4] = function()
    v.kpt = v.kpt + (1)
    return 2
  end
  blocks[5] = function()
    v.ptsid[v.kpt] = v.zero
    return 4
  end
  blocks[6] = function()
    if truth((v.k <= v.i__3)) then return 30 else return 5 end
  end
  blocks[7] = function()
    v.k = v.k + (1)
    return 6
  end
  blocks[8] = function()
    v.pq[v.k] = (v.pq[v.k] + v.temp)
    return 7
  end
  blocks[10] = function()
    v.d__1 = v.ptsaux[((v.iq * 2^(1)) + 2)]
    v.hq[v.ihq] = (v.hq[v.ihq] + (v.temp * (v.d__1 * v.d__1)))
    return 7
  end
  blocks[14] = function()
    v.d__1 = v.ptsaux[((v.iq * 2^(1)) + 1)]
    v.hq[v.ihq] = (v.hq[v.ihq] + (v.temp * (v.d__1 * v.d__1)))
    v.iw = (math.max(v.ihp, v.ihq) - (function() v.i__2 = (v.iq - v.ip); return math.abs(v.i__2) end)())
    v.hq[v.iw] = (v.hq[v.iw] + ((v.temp * v.ptsaux[((v.ip * 2^(1)) + 1)]) * v.ptsaux[((v.iq * 2^(1)) + 1)]))
    return 7
  end
  blocks[18] = function()
    v.ihp = trunc((((v.ip * v.ip) + v.ip))/(2))
    v.d__1 = v.ptsaux[((v.ip * 2^(1)) + 1)]
    v.hq[v.ihp] = (v.hq[v.ihp] + (v.temp * (v.d__1 * v.d__1)))
    if truth((v.iq > 0)) then return 14 else return 7 end
  end
  blocks[22] = function()
    v.ip = trunc(v.ptsid[v.k])
    v.iq = trunc(((v.np * v.ptsid[v.k]) - (v.ip * v.np)))
    v.ihq = trunc((((v.iq * v.iq) + v.iq))/(2))
    if truth((v.ip == 0)) then return 10 else return 18 end
  end
  blocks[24] = function()
    v.temp = (v.diff * v.sum)
    if truth((v.ptsid[v.k] == v.zero)) then return 8 else return 22 end
  end
  blocks[25] = function()
    if truth((v.j <= v.i__2)) then return 27 else return 24 end
  end
  blocks[27] = function()
    v.sum = (v.sum + (v.zmat[(v.k + (v.j * v.zmat_dim1))] * v.zmat[(v.kpt + (v.j * v.zmat_dim1))]))
    v.j = v.j + (1)
    return 25
  end
  blocks[30] = function()
    v.sum = v.zero
    v.i__2 = v.nptm
    v.j = 1
    return 25
  end
  blocks[32] = function()
    v.i__3 = v.npt[0]
    v.k = 1
    return 6
  end
  blocks[33] = function()
    if truth((v.i__ <= v.i__3)) then return 35 else return 32 end
  end
  blocks[35] = function()
    v.gopt[v.i__] = (v.gopt[v.i__] + (v.diff * v.bmat[(v.kpt + (v.i__ * v.bmat_dim1))]))
    v.i__ = v.i__ + (1)
    return 33
  end
  blocks[38] = function()
    v.diff = (v.f - v.vquad)
    v.i__3 = v.n[0]
    v.i__ = 1
    return 33
  end
  blocks[39] = function()
    result = NLOPT_FORCED_STOP; return nil
  end
  blocks[40] = function()
    result = NLOPT_MINF_MAX_REACHED; return nil
  end
  blocks[41] = function()
    result = NLOPT_MAXEVAL_REACHED; return nil
  end
  blocks[42] = function()
    result = NLOPT_MAXTIME_REACHED; return nil
  end
  blocks[43] = function()
    if truth(nlopt_stop_time(v.stop)) then return 42 else return 38 end
  end
  blocks[44] = function()
    if truth(nlopt_stop_evals(v.stop)) then return 41 else return 43 end
  end
  blocks[45] = function()
    if truth((v.f < v.stop.minf_max)) then return 40 else return 44 end
  end
  blocks[46] = function()
    if truth(nlopt_stop_forced(v.stop)) then return 39 else return 45 end
  end
  blocks[47] = function()
    v.kopt[0] = v.kpt
    return 46
  end
  blocks[51] = function()
    v.stop.nevals_p[0] = v.stop.nevals_p[0] + (1)
    v.f = v.calfun(v.n[0], add(v.w,1), v.calfun_data)
    v.fval[v.kpt] = v.f
    if truth((v.f < v.fval[v.kopt[0]])) then return 47 else return 46 end
  end
  blocks[52] = function()
    if truth((v.i__ <= v.i__3)) then return 60 else return 51 end
  end
  blocks[53] = function()
    v.i__ = v.i__ + (1)
    return 52
  end
  blocks[54] = function()
    v.w[v.i__] = v.xu[v.i__]
    return 53
  end
  blocks[55] = function()
    if truth((v.xpt[(v.kpt + (v.i__ * v.xpt_dim1))] == v.su[v.i__])) then return 54 else return 53 end
  end
  blocks[56] = function()
    v.w[v.i__] = v.xl[v.i__]
    return 55
  end
  blocks[60] = function()
    v.d__3 = v.xl[v.i__]; v.d__4 = (v.xbase[v.i__] + v.xpt[(v.kpt + (v.i__ * v.xpt_dim1))])
    v.d__1 = math.max(v.d__3, v.d__4); v.d__2 = v.xu[v.i__]
    v.w[v.i__] = math.min(v.d__1, v.d__2)
    if truth((v.xpt[(v.kpt + (v.i__ * v.xpt_dim1))] == v.sl[v.i__])) then return 56 else return 55 end
  end
  blocks[62] = function()
    v.i__3 = v.n[0]
    v.i__ = 1
    return 52
  end
  blocks[63] = function()
    if truth((v.k <= v.i__3)) then return 70 else return 62 end
  end
  blocks[65] = function()
    v.vquad = (v.vquad + (((v.half * v.pq[v.k]) * v.temp) * v.temp))
    v.k = v.k + (1)
    return 63
  end
  blocks[66] = function()
    v.temp = (v.temp + (v.xq * v.xpt[(v.k + (v.iq * v.xpt_dim1))]))
    return 65
  end
  blocks[67] = function()
    if truth((v.iq > 0)) then return 66 else return 65 end
  end
  blocks[68] = function()
    v.temp = (v.temp + (v.xp * v.xpt[(v.k + (v.ip * v.xpt_dim1))]))
    return 67
  end
  blocks[70] = function()
    v.temp = v.zero
    if truth((v.ip > 0)) then return 68 else return 67 end
  end
  blocks[72] = function()
    v.i__3 = v.npt[0]
    v.k = 1
    return 63
  end
  blocks[74] = function()
    v.iw = (math.max(v.ihp, v.ihq) - (function() v.i__3 = (v.ip - v.iq); return math.abs(v.i__3) end)())
    v.vquad = (v.vquad + ((v.xp * v.xq) * v.hq[v.iw]))
    return 72
  end
  blocks[77] = function()
    v.ihq = trunc(((v.iq + (v.iq * v.iq)))/(2))
    v.vquad = (v.vquad + (v.xq * (v.gopt[v.iq] + ((v.half * v.xq) * v.hq[v.ihq]))))
    if truth((v.ip > 0)) then return 74 else return 72 end
  end
  blocks[78] = function()
    if truth((v.iq > 0)) then return 77 else return 72 end
  end
  blocks[80] = function()
    v.ihp = trunc(((v.ip + (v.ip * v.ip)))/(2))
    v.vquad = (v.vquad + (v.xp * (v.gopt[v.ip] + ((v.half * v.xp) * v.hq[v.ihp]))))
    return 78
  end
  blocks[82] = function()
    v.vquad = v.fbase
    if truth((v.ip > 0)) then return 80 else return 78 end
  end
  blocks[83] = function()
    v.xpt[(v.kpt + (v.iq * v.xpt_dim1))] = v.xq
    return 82
  end
  blocks[84] = function()
    v.xq = v.ptsaux[((v.iq * 2^(1)) + 2)]
    return 83
  end
  blocks[86] = function()
    v.xq = v.ptsaux[((v.iq * 2^(1)) + 1)]
    if truth((v.ip == 0)) then return 84 else return 83 end
  end
  blocks[87] = function()
    if truth((v.iq > 0)) then return 86 else return 82 end
  end
  blocks[89] = function()
    v.xp = v.ptsaux[((v.ip * 2^(1)) + 1)]
    v.xpt[(v.kpt + (v.ip * v.xpt_dim1))] = v.xp
    return 87
  end
  blocks[93] = function()
    v.pq[v.kpt] = v.zero
    v.ip = trunc(v.ptsid[v.kpt])
    v.iq = trunc(((v.np * v.ptsid[v.kpt]) - (v.ip * v.np)))
    if truth((v.ip > 0)) then return 89 else return 87 end
  end
  blocks[94] = function()
    if truth((v.j <= v.i__2)) then return 104 else return 93 end
  end
  blocks[95] = function()
    v.j = v.j + (1)
    return 94
  end
  blocks[96] = function()
    if truth((v.i__ <= v.i__3)) then return 99 else return 95 end
  end
  blocks[99] = function()
    v.ih = v.ih + (1)
    v.hq[v.ih] = (v.hq[v.ih] + (v.temp * v.w[v.i__]))
    v.i__ = v.i__ + (1)
    return 96
  end
  blocks[104] = function()
    v.w[v.j] = v.xpt[(v.kpt + (v.j * v.xpt_dim1))]
    v.xpt[(v.kpt + (v.j * v.xpt_dim1))] = v.zero
    v.temp = (v.pq[v.kpt] * v.w[v.j])
    v.i__3 = v.j
    v.i__ = 1
    return 96
  end
  blocks[107] = function()
    v.ih = 0
    v.i__2 = v.n[0]
    v.j = 1
    return 94
  end
  blocks[108] = function()
    result = NLOPT_FORCED_STOP; return nil
  end
  blocks[109] = function()
    result = NLOPT_MAXEVAL_REACHED; return nil
  end
  blocks[110] = function()
    result = NLOPT_MAXTIME_REACHED; return nil
  end
  blocks[111] = function()
    if truth(nlopt_stop_time(v.stop)) then return 110 else return 107 end
  end
  blocks[112] = function()
    if truth(nlopt_stop_evals(v.stop)) then return 109 else return 111 end
  end
  blocks[113] = function()
    if truth(nlopt_stop_forced(v.stop)) then return 108 else return 112 end
  end
  blocks[114] = function()
    return 4
  end
  blocks[115] = function()
    if truth((v.ptsid[v.kpt] == v.zero)) then return 114 else return 113 end
  end
  blocks[118] = function()
    return 264
  end
  blocks[120] = function()
    v.w[(v.ndim[0] + v.knew)] = ((-v.w[(v.ndim[0] + v.knew)]) - v.winc)
    return 234
  end
  blocks[121] = function()
    if truth((v.denom <= (v.vlmxsq * .01))) then return 120 else return 118 end
  end
  blocks[122] = function()
    if truth((v.k <= v.i__1)) then return 139 else return 121 end
  end
  blocks[126] = function()
    v.d__3 = v.vlag[v.k]
    v.d__1 = v.vlmxsq; v.d__2 = (v.d__3 * v.d__3)
    v.vlmxsq = math.max(v.d__1, v.d__2)
    v.k = v.k + (1)
    return 122
  end
  blocks[128] = function()
    v.kold = v.k
    v.denom = v.den
    return 126
  end
  blocks[131] = function()
    v.d__1 = v.vlag[v.k]
    v.den = ((v.beta * v.hdiag) + (v.d__1 * v.d__1))
    if truth((v.den > v.denom)) then return 128 else return 126 end
  end
  blocks[132] = function()
    if truth((v.j <= v.i__2)) then return 135 else return 131 end
  end
  blocks[135] = function()
    v.d__1 = v.zmat[(v.k + (v.j * v.zmat_dim1))]
    v.hdiag = (v.hdiag + (v.d__1 * v.d__1))
    v.j = v.j + (1)
    return 132
  end
  blocks[138] = function()
    v.hdiag = v.zero
    v.i__2 = v.nptm
    v.j = 1
    return 132
  end
  blocks[139] = function()
    if truth((v.ptsid[v.k] ~= v.zero)) then return 138 else return 126 end
  end
  blocks[145] = function()
    v.beta = ((((v.half * v.distsq) * v.distsq) + v.beta) - v.bsum)
    v.vlag[v.kopt[0]] = (v.vlag[v.kopt[0]] + v.one)
    v.denom = v.zero
    v.vlmxsq = v.zero
    v.i__1 = v.npt[0]
    v.k = 1
    return 122
  end
  blocks[146] = function()
    if truth((v.j <= v.i__1)) then return 164 else return 145 end
  end
  blocks[151] = function()
    v.bsum = (v.bsum + (v.sum * v.w[v.jp]))
    v.vlag[v.jp] = v.sum
    v.d__1 = v.xpt[(v.knew + (v.j * v.xpt_dim1))]
    v.distsq = (v.distsq + (v.d__1 * v.d__1))
    v.j = v.j + (1)
    return 146
  end
  blocks[152] = function()
    if truth((v.ip <= v.i__2)) then return 154 else return 151 end
  end
  blocks[154] = function()
    v.sum = (v.sum + (v.bmat[(v.ip + (v.j * v.bmat_dim1))] * v.w[v.ip]))
    v.ip = v.ip + (1)
    return 152
  end
  blocks[158] = function()
    v.jp = (v.j + v.npt[0])
    v.bsum = (v.bsum + (v.sum * v.w[v.jp]))
    v.i__2 = v.ndim[0]
    v.ip = (v.npt[0] + 1)
    return 152
  end
  blocks[159] = function()
    if truth((v.k <= v.i__2)) then return 161 else return 158 end
  end
  blocks[161] = function()
    v.sum = (v.sum + (v.bmat[(v.k + (v.j * v.bmat_dim1))] * v.w[v.k]))
    v.k = v.k + (1)
    return 159
  end
  blocks[164] = function()
    v.sum = v.zero
    v.i__2 = v.npt[0]
    v.k = 1
    return 159
  end
  blocks[168] = function()
    v.bsum = v.zero
    v.distsq = v.zero
    v.i__1 = v.n[0]
    v.j = 1
    return 146
  end
  blocks[169] = function()
    if truth((v.j <= v.i__2)) then return 182 else return 168 end
  end
  blocks[170] = function()
    v.j = v.j + (1)
    return 169
  end
  blocks[171] = function()
    if truth((v.k <= v.i__1)) then return 173 else return 170 end
  end
  blocks[173] = function()
    v.vlag[v.k] = (v.vlag[v.k] + (v.sum * v.zmat[(v.k + (v.j * v.zmat_dim1))]))
    v.k = v.k + (1)
    return 171
  end
  blocks[176] = function()
    v.beta = (v.beta - (v.sum * v.sum))
    v.i__1 = v.npt[0]
    v.k = 1
    return 171
  end
  blocks[177] = function()
    if truth((v.k <= v.i__1)) then return 179 else return 176 end
  end
  blocks[179] = function()
    v.sum = (v.sum + (v.zmat[(v.k + (v.j * v.zmat_dim1))] * v.w[v.k]))
    v.k = v.k + (1)
    return 177
  end
  blocks[182] = function()
    v.sum = v.zero
    v.i__1 = v.npt[0]
    v.k = 1
    return 177
  end
  blocks[185] = function()
    v.beta = v.zero
    v.i__2 = v.nptm
    v.j = 1
    return 169
  end
  blocks[186] = function()
    if truth((v.k <= v.i__2)) then return 194 else return 185 end
  end
  blocks[188] = function()
    v.vlag[v.k] = v.sum
    v.k = v.k + (1)
    return 186
  end
  blocks[189] = function()
    if truth((v.j <= v.i__1)) then return 191 else return 188 end
  end
  blocks[191] = function()
    v.sum = (v.sum + (v.bmat[(v.k + (v.j * v.bmat_dim1))] * v.w[(v.npt[0] + v.j)]))
    v.j = v.j + (1)
    return 189
  end
  blocks[194] = function()
    v.sum = v.zero
    v.i__1 = v.n[0]
    v.j = 1
    return 189
  end
  blocks[196] = function()
    v.i__2 = v.npt[0]
    v.k = 1
    return 186
  end
  blocks[197] = function()
    if truth((v.k <= v.i__2)) then return 216 else return 196 end
  end
  blocks[199] = function()
    v.w[v.k] = ((v.half * v.sum) * v.sum)
    v.k = v.k + (1)
    return 197
  end
  blocks[200] = function()
    if truth((v.j <= v.i__1)) then return 202 else return 199 end
  end
  blocks[202] = function()
    v.sum = (v.sum + (v.w[(v.npt[0] + v.j)] * v.xpt[(v.k + (v.j * v.xpt_dim1))]))
    v.j = v.j + (1)
    return 200
  end
  blocks[204] = function()
    v.i__1 = v.n[0]
    v.j = 1
    return 200
  end
  blocks[205] = function()
    v.sum = (v.sum + (v.w[(v.npt[0] + v.iq)] * v.ptsaux[(v.iw + (v.iq * 2^(1)))]))
    return 199
  end
  blocks[206] = function()
    v.iw = 2
    return 205
  end
  blocks[208] = function()
    v.iw = 1
    if truth((v.ip == 0)) then return 206 else return 205 end
  end
  blocks[210] = function()
    v.iq = trunc(((v.np * v.ptsid[v.k]) - (v.ip * v.np)))
    if truth((v.iq > 0)) then return 208 else return 199 end
  end
  blocks[211] = function()
    v.sum = (v.w[(v.npt[0] + v.ip)] * v.ptsaux[((v.ip * 2^(1)) + 1)])
    return 210
  end
  blocks[213] = function()
    v.ip = trunc(v.ptsid[v.k])
    if truth((v.ip > 0)) then return 211 else return 210 end
  end
  blocks[214] = function()
    if truth((v.ptsid[v.k] == v.zero)) then return 204 else return 213 end
  end
  blocks[216] = function()
    v.sum = v.zero
    if truth((v.k == v.kopt[0])) then return 199 else return 214 end
  end
  blocks[218] = function()
    v.i__2 = v.npt[0]
    v.k = 1
    return 197
  end
  blocks[219] = function()
    if truth((v.j <= v.i__2)) then return 221 else return 218 end
  end
  blocks[221] = function()
    v.w[(v.npt[0] + v.j)] = v.xpt[(v.knew + (v.j * v.xpt_dim1))]
    v.j = v.j + (1)
    return 219
  end
  blocks[223] = function()
    v.i__2 = v.n[0]
    v.j = 1
    return 219
  end
  blocks[224] = function()
    v.i__1 = v.npt[0]
    v.kpt = 1
    return 2
  end
  blocks[225] = function()
    if truth((v.dsqmin == v.zero)) then return 224 else return 223 end
  end
  blocks[226] = function()
    if truth((v.k <= v.i__2)) then return 231 else return 225 end
  end
  blocks[227] = function()
    v.k = v.k + (1)
    return 226
  end
  blocks[229] = function()
    v.knew = v.k
    v.dsqmin = v.w[(v.ndim[0] + v.k)]
    return 227
  end
  blocks[230] = function()
    if truth((truth((v.dsqmin == v.zero)) or truth((v.w[(v.ndim[0] + v.k)] < v.dsqmin)))) then return 229 else return 227 end
  end
  blocks[231] = function()
    if truth((v.w[(v.ndim[0] + v.k)] > v.zero)) then return 230 else return 227 end
  end
  blocks[234] = function()
    v.dsqmin = v.zero
    v.i__2 = v.npt[0]
    v.k = 1
    return 226
  end
  blocks[235] = function()
    if truth((v.k <= v.i__2)) then return 237 else return 234 end
  end
  blocks[237] = function()
    v.w[(v.ndim[0] + v.k)] = (function() v.d__1 = v.w[(v.ndim[0] + v.k)]; return math.abs(v.d__1) end)()
    v.k = v.k + (1)
    return 235
  end
  blocks[239] = function()
    v.i__2 = v.npt[0]
    v.k = 1
    return 235
  end
  blocks[240] = function()
    return 1
  end
  blocks[245] = function()
    v.temp = v.vlag[v.kold]
    v.vlag[v.kold] = v.vlag[v.knew]
    v.vlag[v.knew] = v.temp
    update_(v.n, v.npt, add(v.bmat,v.bmat_offset), add(v.zmat,v.zmat_offset), v.ndim, add(v.vlag,1), ptr(v,"beta"), ptr(v,"denom"), ptr(v,"knew"), add(v.w,1))
    if truth((v.nrem == 0)) then return 240 else return 239 end
  end
  blocks[250] = function()
    v.ptsid[v.kold] = v.ptsid[v.knew]
    v.ptsid[v.knew] = v.zero
    v.w[(v.ndim[0] + v.knew)] = v.zero
    v.nrem = v.nrem + (-1)
    if truth((v.knew ~= v.kopt[0])) then return 245 else return 234 end
  end
  blocks[251] = function()
    if truth((v.j <= v.i__2)) then return 255 else return 250 end
  end
  blocks[255] = function()
    v.temp = v.zmat[(v.kold + (v.j * v.zmat_dim1))]
    v.zmat[(v.kold + (v.j * v.zmat_dim1))] = v.zmat[(v.knew + (v.j * v.zmat_dim1))]
    v.zmat[(v.knew + (v.j * v.zmat_dim1))] = v.temp
    v.j = v.j + (1)
    return 251
  end
  blocks[257] = function()
    v.i__2 = v.nptm
    v.j = 1
    return 251
  end
  blocks[258] = function()
    if truth((v.j <= v.i__2)) then return 262 else return 257 end
  end
  blocks[262] = function()
    v.temp = v.bmat[(v.kold + (v.j * v.bmat_dim1))]
    v.bmat[(v.kold + (v.j * v.bmat_dim1))] = v.bmat[(v.knew + (v.j * v.bmat_dim1))]
    v.bmat[(v.knew + (v.j * v.bmat_dim1))] = v.temp
    v.j = v.j + (1)
    return 258
  end
  blocks[264] = function()
    v.i__2 = v.n[0]
    v.j = 1
    return 258
  end
  blocks[267] = function()
    v.nrem = v.npt[0]
    v.kold = 1
    v.knew = v.kopt[0]
    return 264
  end
  blocks[268] = function()
    if truth((v.k <= v.i__2)) then return 280 else return 267 end
  end
  blocks[275] = function()
    v.ptsid[v.k] = ((v.ip + (v.iq / v.np)) + v.sfrac)
    v.temp = (v.one / (v.ptsaux[((v.ip * 2^(1)) + 1)] * v.ptsaux[((v.iq * 2^(1)) + 1)]))
    v.zmat[(((v.k - v.np) * v.zmat_dim1) + 1)] = v.temp
    v.zmat[((v.ip + 1) + ((v.k - v.np) * v.zmat_dim1))] = (-v.temp)
    v.zmat[((v.iq + 1) + ((v.k - v.np) * v.zmat_dim1))] = (-v.temp)
    v.zmat[(v.k + ((v.k - v.np) * v.zmat_dim1))] = v.temp
    v.k = v.k + (1)
    return 268
  end
  blocks[276] = function()
    v.iq = (v.iq - v.n[0])
    return 275
  end
  blocks[280] = function()
    v.iw = trunc((((v.k - v.np) - v.half) / v.n[0]))
    v.ip = ((v.k - v.np) - (v.iw * v.n[0]))
    v.iq = (v.ip + v.iw)
    if truth((v.iq > v.n[0])) then return 276 else return 275 end
  end
  blocks[282] = function()
    v.i__2 = v.npt[0]
    v.k = (v.np * 2^(1))
    return 268
  end
  blocks[283] = function()
    if truth((v.npt[0] >= (v.n[0] + v.np))) then return 282 else return 267 end
  end
  blocks[284] = function()
    if truth((v.j <= v.i__2)) then return 301 else return 283 end
  end
  blocks[285] = function()
    v.j = v.j + (1)
    return 284
  end
  blocks[293] = function()
    v.ptsid[v.jpn] = ((v.j / v.np) + v.sfrac)
    v.temp = (v.one / (v.ptsaux[((v.j * 2^(1)) + 1)] - v.ptsaux[((v.j * 2^(1)) + 2)]))
    v.bmat[(v.jp + (v.j * v.bmat_dim1))] = ((-v.temp) + (v.one / v.ptsaux[((v.j * 2^(1)) + 1)]))
    v.bmat[(v.jpn + (v.j * v.bmat_dim1))] = (v.temp + (v.one / v.ptsaux[((v.j * 2^(1)) + 2)]))
    v.bmat[((v.j * v.bmat_dim1) + 1)] = ((-v.bmat[(v.jp + (v.j * v.bmat_dim1))]) - v.bmat[(v.jpn + (v.j * v.bmat_dim1))])
    v.zmat[((v.j * v.zmat_dim1) + 1)] = (math.sqrt(2.) / (function() v.d__1 = (v.ptsaux[((v.j * 2^(1)) + 1)] * v.ptsaux[((v.j * 2^(1)) + 2)]); return math.abs(v.d__1) end)())
    v.zmat[(v.jp + (v.j * v.zmat_dim1))] = ((v.zmat[((v.j * v.zmat_dim1) + 1)] * v.ptsaux[((v.j * 2^(1)) + 2)]) * v.temp)
    v.zmat[(v.jpn + (v.j * v.zmat_dim1))] = (((-v.zmat[((v.j * v.zmat_dim1) + 1)]) * v.ptsaux[((v.j * 2^(1)) + 1)]) * v.temp)
    return 285
  end
  blocks[297] = function()
    v.bmat[((v.j * v.bmat_dim1) + 1)] = ((-v.one) / v.ptsaux[((v.j * 2^(1)) + 1)])
    v.bmat[(v.jp + (v.j * v.bmat_dim1))] = (v.one / v.ptsaux[((v.j * 2^(1)) + 1)])
    v.d__1 = v.ptsaux[((v.j * 2^(1)) + 1)]
    v.bmat[((v.j + v.npt[0]) + (v.j * v.bmat_dim1))] = ((-v.half) * (v.d__1 * v.d__1))
    return 285
  end
  blocks[301] = function()
    v.jp = (v.j + 1)
    v.jpn = (v.jp + v.n[0])
    v.ptsid[v.jp] = (v.j + v.sfrac)
    if truth((v.jpn <= v.npt[0])) then return 293 else return 297 end
  end
  blocks[305] = function()
    v.fbase = v.fval[v.kopt[0]]
    v.ptsid[1] = v.sfrac
    v.i__2 = v.n[0]
    v.j = 1
    return 284
  end
  blocks[306] = function()
    if truth((v.j <= v.i__1)) then return 326 else return 305 end
  end
  blocks[307] = function()
    v.j = v.j + (1)
    return 306
  end
  blocks[308] = function()
    if truth((v.i__ <= v.i__2)) then return 310 else return 307 end
  end
  blocks[310] = function()
    v.bmat[(v.i__ + (v.j * v.bmat_dim1))] = v.zero
    v.i__ = v.i__ + (1)
    return 308
  end
  blocks[312] = function()
    v.i__2 = v.ndim[0]
    v.i__ = 1
    return 308
  end
  blocks[313] = function()
    v.ptsaux[((v.j * 2^(1)) + 2)] = (v.half * v.ptsaux[((v.j * 2^(1)) + 1)])
    return 312
  end
  blocks[314] = function()
    if truth(((function() v.d__2 = v.ptsaux[((v.j * 2^(1)) + 2)]; return math.abs(v.d__2) end)() < (v.half * (function() v.d__1 = v.ptsaux[((v.j * 2^(1)) + 1)]; return math.abs(v.d__1) end)()))) then return 313 else return 312 end
  end
  blocks[317] = function()
    v.temp = v.ptsaux[((v.j * 2^(1)) + 1)]
    v.ptsaux[((v.j * 2^(1)) + 1)] = v.ptsaux[((v.j * 2^(1)) + 2)]
    v.ptsaux[((v.j * 2^(1)) + 2)] = v.temp
    return 314
  end
  blocks[326] = function()
    v.xbase[v.j] = (v.xbase[v.j] + v.xopt[v.j])
    v.sl[v.j] = (v.sl[v.j] - v.xopt[v.j])
    v.su[v.j] = (v.su[v.j] - v.xopt[v.j])
    v.xopt[v.j] = v.zero
    v.d__1 = v.delta[0]; v.d__2 = v.su[v.j]
    v.ptsaux[((v.j * 2^(1)) + 1)] = math.min(v.d__1, v.d__2)
    v.d__1 = (-v.delta[0]); v.d__2 = v.sl[v.j]
    v.ptsaux[((v.j * 2^(1)) + 2)] = math.max(v.d__1, v.d__2)
    if truth(((v.ptsaux[((v.j * 2^(1)) + 1)] + v.ptsaux[((v.j * 2^(1)) + 2)]) < v.zero)) then return 317 else return 314 end
  end
  blocks[328] = function()
    v.i__1 = v.n[0]
    v.j = 1
    return 306
  end
  blocks[329] = function()
    if truth((v.j <= v.i__2)) then return 342 else return 328 end
  end
  blocks[330] = function()
    v.j = v.j + (1)
    return 329
  end
  blocks[331] = function()
    if truth((v.i__ <= v.i__1)) then return 334 else return 330 end
  end
  blocks[334] = function()
    v.ih = v.ih + (1)
    v.hq[v.ih] = ((v.hq[v.ih] + (v.w[v.i__] * v.xopt[v.j])) + (v.w[v.j] * v.xopt[v.i__]))
    v.i__ = v.i__ + (1)
    return 331
  end
  blocks[336] = function()
    v.i__1 = v.j
    v.i__ = 1
    return 331
  end
  blocks[337] = function()
    if truth((v.k <= v.i__1)) then return 339 else return 336 end
  end
  blocks[339] = function()
    v.w[v.j] = (v.w[v.j] + (v.pq[v.k] * v.xpt[(v.k + (v.j * v.xpt_dim1))]))
    v.k = v.k + (1)
    return 337
  end
  blocks[342] = function()
    v.w[v.j] = ((v.half * v.sumpq) * v.xopt[v.j])
    v.i__1 = v.npt[0]
    v.k = 1
    return 337
  end
  blocks[345] = function()
    v.ih = 0
    v.i__2 = v.n[0]
    v.j = 1
    return 329
  end
  blocks[346] = function()
    if truth((v.k <= v.i__1)) then return 363 else return 345 end
  end
  blocks[347] = function()
    v.k = v.k + (1)
    return 346
  end
  blocks[348] = function()
    if truth((v.j <= v.i__2)) then return 350 else return 347 end
  end
  blocks[350] = function()
    v.zmat[(v.k + (v.j * v.zmat_dim1))] = v.zero
    v.j = v.j + (1)
    return 348
  end
  blocks[355] = function()
    v.sumpq = (v.sumpq + v.pq[v.k])
    v.w[(v.ndim[0] + v.k)] = v.distsq
    v.winc = math.max(v.winc, v.distsq)
    v.i__2 = v.nptm
    v.j = 1
    return 348
  end
  blocks[356] = function()
    if truth((v.j <= v.i__2)) then return 360 else return 355 end
  end
  blocks[360] = function()
    v.xpt[(v.k + (v.j * v.xpt_dim1))] = (v.xpt[(v.k + (v.j * v.xpt_dim1))] - v.xopt[v.j])
    v.d__1 = v.xpt[(v.k + (v.j * v.xpt_dim1))]
    v.distsq = (v.distsq + (v.d__1 * v.d__1))
    v.j = v.j + (1)
    return 356
  end
  blocks[363] = function()
    v.distsq = v.zero
    v.i__2 = v.n[0]
    v.j = 1
    return 356
  end
  blocks[448] = function()
    v.xp = 0.0
    v.xq = 0.0
    v.ihp = 0
    v.zmat_dim1 = v.npt[0]
    v.zmat_offset = (1 + v.zmat_dim1)
    v.zmat = add(v.zmat,(-v.zmat_offset))
    v.xpt_dim1 = v.npt[0]
    v.xpt_offset = (1 + v.xpt_dim1)
    v.xpt = add(v.xpt,(-v.xpt_offset))
    v.xl = add(v.xl,-1)
    v.xu = add(v.xu,-1)
    v.xbase = add(v.xbase,-1)
    v.fval = add(v.fval,-1)
    v.xopt = add(v.xopt,-1)
    v.gopt = add(v.gopt,-1)
    v.hq = add(v.hq,-1)
    v.pq = add(v.pq,-1)
    v.bmat_dim1 = v.ndim[0]
    v.bmat_offset = (1 + v.bmat_dim1)
    v.bmat = add(v.bmat,(-v.bmat_offset))
    v.sl = add(v.sl,-1)
    v.su = add(v.su,-1)
    v.vlag = add(v.vlag,-1)
    v.ptsaux = add(v.ptsaux,(-3))
    v.ptsid = add(v.ptsid,-1)
    v.w = add(v.w,-1)
    v.half = .5
    v.one = 1.
    v.zero = 0.
    v.np = (v.n[0] + 1)
    v.sfrac = (v.half / v.np)
    v.nptm = (v.npt[0] - v.np)
    v.sumpq = v.zero
    v.winc = v.zero
    v.i__1 = v.npt[0]
    v.k = 1
    return 346
  end
  local pc = 448
  while pc do pc = blocks[pc]() end
  return result
end

altmov_ = function(n, npt, xpt, xopt, bmat, zmat, ndim, sl, su, kopt, knew, adelt, xnew, xalt, alpha, cauchy, glag, hcol, w)
  local v = {}
  v.n = n
  v.npt = npt
  v.xpt = xpt
  v.xopt = xopt
  v.bmat = bmat
  v.zmat = zmat
  v.ndim = ndim
  v.sl = sl
  v.su = su
  v.kopt = kopt
  v.knew = knew
  v.adelt = adelt
  v.xnew = xnew
  v.xalt = xalt
  v.alpha = alpha
  v.cauchy = cauchy
  v.glag = glag
  v.hcol = hcol
  v.w = w
  local blocks, result = {}, nil
  blocks[1] = function()
    result = nil; return nil
  end
  blocks[2] = function()
    v.cauchy[0] = v.csave
    return 1
  end
  blocks[3] = function()
    if truth((v.i__ <= v.i__1)) then return 5 else return 2 end
  end
  blocks[5] = function()
    v.xalt[v.i__] = v.w[(v.n[0] + v.i__)]
    v.i__ = v.i__ + (1)
    return 3
  end
  blocks[7] = function()
    v.i__1 = v.n[0]
    v.i__ = 1
    return 3
  end
  blocks[8] = function()
    if truth((v.csave > v.cauchy[0])) then return 7 else return 1 end
  end
  blocks[11] = function()
    v.csave = v.cauchy[0]
    v.iflag = 1
    return 103
  end
  blocks[12] = function()
    if truth((v.i__ <= v.i__1)) then return 15 else return 11 end
  end
  blocks[15] = function()
    v.glag[v.i__] = (-v.glag[v.i__])
    v.w[(v.n[0] + v.i__)] = v.xalt[v.i__]
    v.i__ = v.i__ + (1)
    return 12
  end
  blocks[17] = function()
    v.i__1 = v.n[0]
    v.i__ = 1
    return 12
  end
  blocks[18] = function()
    if truth((v.iflag == 0)) then return 17 else return 8 end
  end
  blocks[20] = function()
    v.d__1 = ((v.half * v.gw) * v.scale)
    v.cauchy[0] = (v.d__1 * v.d__1)
    return 18
  end
  blocks[21] = function()
    if truth((v.i__ <= v.i__1)) then return 26 else return 20 end
  end
  blocks[26] = function()
    v.temp = (v.xopt[v.i__] + (v.scale * v.w[v.i__]))
    v.d__3 = v.su[v.i__]
    v.d__1 = v.sl[v.i__]; v.d__2 = math.min(v.d__3, v.temp)
    v.xalt[v.i__] = math.max(v.d__1, v.d__2)
    v.i__ = v.i__ + (1)
    return 21
  end
  blocks[29] = function()
    v.scale = ((-v.gw) / v.curv)
    v.i__1 = v.n[0]
    v.i__ = 1
    return 21
  end
  blocks[31] = function()
    v.d__1 = (v.gw + (v.half * v.curv))
    v.cauchy[0] = (v.d__1 * v.d__1)
    return 18
  end
  blocks[32] = function()
    if truth((truth((v.curv > (-v.gw))) and truth((v.curv < ((-v.const__) * v.gw))))) then return 29 else return 31 end
  end
  blocks[33] = function()
    v.curv = (-v.curv)
    return 32
  end
  blocks[34] = function()
    if truth((v.iflag == 1)) then return 33 else return 32 end
  end
  blocks[35] = function()
    if truth((v.k <= v.i__1)) then return 43 else return 34 end
  end
  blocks[37] = function()
    v.curv = (v.curv + ((v.hcol[v.k] * v.temp) * v.temp))
    v.k = v.k + (1)
    return 35
  end
  blocks[38] = function()
    if truth((v.j <= v.i__2)) then return 40 else return 37 end
  end
  blocks[40] = function()
    v.temp = (v.temp + (v.xpt[(v.k + (v.j * v.xpt_dim1))] * v.w[v.j]))
    v.j = v.j + (1)
    return 38
  end
  blocks[43] = function()
    v.temp = v.zero
    v.i__2 = v.n[0]
    v.j = 1
    return 38
  end
  blocks[46] = function()
    v.curv = v.zero
    v.i__1 = v.npt[0]
    v.k = 1
    return 35
  end
  blocks[47] = function()
    if truth((v.i__ <= v.i__1)) then return 59 else return 46 end
  end
  blocks[49] = function()
    v.gw = (v.gw + (v.glag[v.i__] * v.w[v.i__]))
    v.i__ = v.i__ + (1)
    return 47
  end
  blocks[53] = function()
    v.w[v.i__] = ((-v.step) * v.glag[v.i__])
    v.d__3 = v.su[v.i__]; v.d__4 = (v.xopt[v.i__] + v.w[v.i__])
    v.d__1 = v.sl[v.i__]; v.d__2 = math.min(v.d__3, v.d__4)
    v.xalt[v.i__] = math.max(v.d__1, v.d__2)
    return 49
  end
  blocks[54] = function()
    v.xalt[v.i__] = v.xopt[v.i__]
    return 49
  end
  blocks[55] = function()
    v.xalt[v.i__] = v.sl[v.i__]
    return 49
  end
  blocks[56] = function()
    v.xalt[v.i__] = v.su[v.i__]
    return 49
  end
  blocks[57] = function()
    if truth((v.glag[v.i__] > v.zero)) then return 55 else return 56 end
  end
  blocks[58] = function()
    if truth((v.w[v.i__] == v.zero)) then return 54 else return 57 end
  end
  blocks[59] = function()
    if truth((v.w[v.i__] == v.bigstp)) then return 53 else return 58 end
  end
  blocks[62] = function()
    v.gw = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 47
  end
  blocks[63] = function()
    return 85
  end
  blocks[64] = function()
    if truth((truth((v.wfixsq > v.wsqsav)) and truth((v.ggfree > v.zero)))) then return 63 else return 62 end
  end
  blocks[65] = function()
    if truth((v.i__ <= v.i__1)) then return 78 else return 64 end
  end
  blocks[66] = function()
    v.i__ = v.i__ + (1)
    return 65
  end
  blocks[69] = function()
    v.w[v.i__] = (v.sl[v.i__] - v.xopt[v.i__])
    v.d__1 = v.w[v.i__]
    v.wfixsq = (v.wfixsq + (v.d__1 * v.d__1))
    return 66
  end
  blocks[72] = function()
    v.w[v.i__] = (v.su[v.i__] - v.xopt[v.i__])
    v.d__1 = v.w[v.i__]
    v.wfixsq = (v.wfixsq + (v.d__1 * v.d__1))
    return 66
  end
  blocks[74] = function()
    v.d__1 = v.glag[v.i__]
    v.ggfree = (v.ggfree + (v.d__1 * v.d__1))
    return 66
  end
  blocks[75] = function()
    if truth((v.temp >= v.su[v.i__])) then return 72 else return 74 end
  end
  blocks[77] = function()
    v.temp = (v.xopt[v.i__] - (v.step * v.glag[v.i__]))
    if truth((v.temp <= v.sl[v.i__])) then return 69 else return 75 end
  end
  blocks[78] = function()
    if truth((v.w[v.i__] == v.bigstp)) then return 77 else return 66 end
  end
  blocks[83] = function()
    v.wsqsav = v.wfixsq
    v.step = math.sqrt((v.temp / v.ggfree))
    v.ggfree = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 65
  end
  blocks[85] = function()
    v.temp = ((v.adelt[0] * v.adelt[0]) - v.wfixsq)
    if truth((v.temp > v.zero)) then return 83 else return 62 end
  end
  blocks[87] = function()
    v.cauchy[0] = v.zero
    return 1
  end
  blocks[88] = function()
    if truth((v.ggfree == v.zero)) then return 87 else return 85 end
  end
  blocks[89] = function()
    if truth((v.i__ <= v.i__1)) then return 99 else return 88 end
  end
  blocks[90] = function()
    v.i__ = v.i__ + (1)
    return 89
  end
  blocks[93] = function()
    v.w[v.i__] = v.bigstp
    v.d__1 = v.glag[v.i__]
    v.ggfree = (v.ggfree + (v.d__1 * v.d__1))
    return 90
  end
  blocks[99] = function()
    v.w[v.i__] = v.zero
    v.d__1 = (v.xopt[v.i__] - v.sl[v.i__]); v.d__2 = v.glag[v.i__]
    v.tempa = math.min(v.d__1, v.d__2)
    v.d__1 = (v.xopt[v.i__] - v.su[v.i__]); v.d__2 = v.glag[v.i__]
    v.tempb = math.max(v.d__1, v.d__2)
    if truth((truth((v.tempa > v.zero)) or truth((v.tempb < v.zero)))) then return 93 else return 90 end
  end
  blocks[103] = function()
    v.wfixsq = v.zero
    v.ggfree = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 89
  end
  blocks[105] = function()
    v.bigstp = (v.adelt[0] + v.adelt[0])
    v.iflag = 0
    return 103
  end
  blocks[106] = function()
    v.xnew[v.ibdsav] = v.su[v.ibdsav]
    return 105
  end
  blocks[107] = function()
    if truth((v.ibdsav > 0)) then return 106 else return 105 end
  end
  blocks[108] = function()
    v.xnew[(-v.ibdsav)] = v.sl[(-v.ibdsav)]
    return 107
  end
  blocks[109] = function()
    if truth((v.ibdsav < 0)) then return 108 else return 107 end
  end
  blocks[110] = function()
    if truth((v.i__ <= v.i__1)) then return 115 else return 109 end
  end
  blocks[115] = function()
    v.temp = (v.xopt[v.i__] + (v.stpsav * (v.xpt[(v.ksav + (v.i__ * v.xpt_dim1))] - v.xopt[v.i__])))
    v.d__3 = v.su[v.i__]
    v.d__1 = v.sl[v.i__]; v.d__2 = math.min(v.d__3, v.temp)
    v.xnew[v.i__] = math.max(v.d__1, v.d__2)
    v.i__ = v.i__ + (1)
    return 110
  end
  blocks[117] = function()
    v.i__1 = v.n[0]
    v.i__ = 1
    return 110
  end
  blocks[118] = function()
    if truth((v.k <= v.i__1)) then return 197 else return 117 end
  end
  blocks[120] = function()
    v.k = v.k + (1)
    return 118
  end
  blocks[124] = function()
    v.presav = v.predsq
    v.ksav = v.k
    v.stpsav = v.step
    v.ibdsav = v.isbd
    return 120
  end
  blocks[127] = function()
    v.temp = ((v.step * (v.one - v.step)) * v.distsq)
    v.predsq = ((v.vlag * v.vlag) * ((v.vlag * v.vlag) + ((v.ha * v.temp) * v.temp)))
    if truth((v.predsq > v.presav)) then return 124 else return 120 end
  end
  blocks[130] = function()
    v.step = (v.tempd / v.diff)
    v.vlag = v.temp
    v.isbd = 0
    return 127
  end
  blocks[132] = function()
    v.temp = ((v.tempd * v.tempd) / v.diff)
    if truth((math.abs(v.temp) > math.abs(v.vlag))) then return 130 else return 127 end
  end
  blocks[136] = function()
    v.tempd = (v.half * v.dderiv)
    v.tempa = (v.tempd - (v.diff * v.slbd))
    v.tempb = (v.tempd - (v.diff * v.subd))
    if truth(((v.tempa * v.tempb) < v.zero)) then return 132 else return 127 end
  end
  blocks[139] = function()
    v.step = v.subd
    v.vlag = v.temp
    v.isbd = v.iubd
    return 136
  end
  blocks[145] = function()
    v.diff = (v.dderiv - v.one)
    v.step = v.slbd
    v.vlag = (v.slbd * (v.dderiv - (v.slbd * v.diff)))
    v.isbd = v.ilbd
    v.temp = (v.subd * (v.dderiv - (v.subd * v.diff)))
    if truth((math.abs(v.temp) > math.abs(v.vlag))) then return 139 else return 136 end
  end
  blocks[146] = function()
    v.vlag = (v.vlag * v.dderiv)
    return 127
  end
  blocks[149] = function()
    v.step = v.half
    v.vlag = .25
    v.isbd = 0
    return 146
  end
  blocks[150] = function()
    if truth((math.abs(v.vlag) < .25)) then return 149 else return 146 end
  end
  blocks[151] = function()
    if truth((v.subd > v.half)) then return 150 else return 146 end
  end
  blocks[154] = function()
    v.step = v.subd
    v.vlag = v.temp
    v.isbd = v.iubd
    return 151
  end
  blocks[159] = function()
    v.step = v.slbd
    v.vlag = (v.slbd * (v.one - v.slbd))
    v.isbd = v.ilbd
    v.temp = (v.subd * (v.one - v.subd))
    if truth((math.abs(v.temp) > math.abs(v.vlag))) then return 154 else return 151 end
  end
  blocks[160] = function()
    if truth((v.k == v.knew[0])) then return 145 else return 159 end
  end
  blocks[161] = function()
    if truth((v.i__ <= v.i__2)) then return 179 else return 160 end
  end
  blocks[162] = function()
    v.i__ = v.i__ + (1)
    return 161
  end
  blocks[165] = function()
    v.d__1 = v.sumin; v.d__2 = ((v.su[v.i__] - v.xopt[v.i__]) / v.temp)
    v.subd = math.max(v.d__1, v.d__2)
    v.iubd = v.i__
    return 162
  end
  blocks[166] = function()
    if truth(((v.subd * v.temp) > (v.su[v.i__] - v.xopt[v.i__]))) then return 165 else return 162 end
  end
  blocks[168] = function()
    v.slbd = ((v.sl[v.i__] - v.xopt[v.i__]) / v.temp)
    v.ilbd = (-v.i__)
    return 166
  end
  blocks[169] = function()
    if truth(((v.slbd * v.temp) < (v.sl[v.i__] - v.xopt[v.i__]))) then return 168 else return 166 end
  end
  blocks[172] = function()
    v.d__1 = v.sumin; v.d__2 = ((v.sl[v.i__] - v.xopt[v.i__]) / v.temp)
    v.subd = math.max(v.d__1, v.d__2)
    v.iubd = (-v.i__)
    return 162
  end
  blocks[173] = function()
    if truth(((v.subd * v.temp) < (v.sl[v.i__] - v.xopt[v.i__]))) then return 172 else return 162 end
  end
  blocks[175] = function()
    v.slbd = ((v.su[v.i__] - v.xopt[v.i__]) / v.temp)
    v.ilbd = v.i__
    return 173
  end
  blocks[176] = function()
    if truth(((v.slbd * v.temp) > (v.su[v.i__] - v.xopt[v.i__]))) then return 175 else return 173 end
  end
  blocks[177] = function()
    if truth((v.temp < v.zero)) then return 176 else return 162 end
  end
  blocks[179] = function()
    v.temp = (v.xpt[(v.k + (v.i__ * v.xpt_dim1))] - v.xopt[v.i__])
    if truth((v.temp > v.zero)) then return 169 else return 177 end
  end
  blocks[186] = function()
    v.subd = (v.adelt[0] / math.sqrt(v.distsq))
    v.slbd = (-v.subd)
    v.ilbd = 0
    v.iubd = 0
    v.sumin = math.min(v.one, v.subd)
    v.i__2 = v.n[0]
    v.i__ = 1
    return 161
  end
  blocks[187] = function()
    if truth((v.i__ <= v.i__2)) then return 191 else return 186 end
  end
  blocks[191] = function()
    v.temp = (v.xpt[(v.k + (v.i__ * v.xpt_dim1))] - v.xopt[v.i__])
    v.dderiv = (v.dderiv + (v.glag[v.i__] * v.temp))
    v.distsq = (v.distsq + (v.temp * v.temp))
    v.i__ = v.i__ + (1)
    return 187
  end
  blocks[195] = function()
    v.dderiv = v.zero
    v.distsq = v.zero
    v.i__2 = v.n[0]
    v.i__ = 1
    return 187
  end
  blocks[196] = function()
    return 120
  end
  blocks[197] = function()
    if truth((v.k == v.kopt[0])) then return 196 else return 195 end
  end
  blocks[200] = function()
    v.presav = v.zero
    v.i__1 = v.npt[0]
    v.k = 1
    return 118
  end
  blocks[201] = function()
    if truth((v.k <= v.i__2)) then return 214 else return 200 end
  end
  blocks[202] = function()
    v.k = v.k + (1)
    return 201
  end
  blocks[203] = function()
    if truth((v.i__ <= v.i__1)) then return 205 else return 202 end
  end
  blocks[205] = function()
    v.glag[v.i__] = (v.glag[v.i__] + (v.temp * v.xpt[(v.k + (v.i__ * v.xpt_dim1))]))
    v.i__ = v.i__ + (1)
    return 203
  end
  blocks[208] = function()
    v.temp = (v.hcol[v.k] * v.temp)
    v.i__1 = v.n[0]
    v.i__ = 1
    return 203
  end
  blocks[209] = function()
    if truth((v.j <= v.i__1)) then return 211 else return 208 end
  end
  blocks[211] = function()
    v.temp = (v.temp + (v.xpt[(v.k + (v.j * v.xpt_dim1))] * v.xopt[v.j]))
    v.j = v.j + (1)
    return 209
  end
  blocks[214] = function()
    v.temp = v.zero
    v.i__1 = v.n[0]
    v.j = 1
    return 209
  end
  blocks[216] = function()
    v.i__2 = v.npt[0]
    v.k = 1
    return 201
  end
  blocks[217] = function()
    if truth((v.i__ <= v.i__2)) then return 219 else return 216 end
  end
  blocks[219] = function()
    v.glag[v.i__] = v.bmat[(v.knew[0] + (v.i__ * v.bmat_dim1))]
    v.i__ = v.i__ + (1)
    return 217
  end
  blocks[223] = function()
    v.alpha[0] = v.hcol[v.knew[0]]
    v.ha = (v.half * v.alpha[0])
    v.i__2 = v.n[0]
    v.i__ = 1
    return 217
  end
  blocks[224] = function()
    if truth((v.j <= v.i__1)) then return 231 else return 223 end
  end
  blocks[225] = function()
    v.j = v.j + (1)
    return 224
  end
  blocks[226] = function()
    if truth((v.k <= v.i__2)) then return 228 else return 225 end
  end
  blocks[228] = function()
    v.hcol[v.k] = (v.hcol[v.k] + (v.temp * v.zmat[(v.k + (v.j * v.zmat_dim1))]))
    v.k = v.k + (1)
    return 226
  end
  blocks[231] = function()
    v.temp = v.zmat[(v.knew[0] + (v.j * v.zmat_dim1))]
    v.i__2 = v.npt[0]
    v.k = 1
    return 226
  end
  blocks[233] = function()
    v.i__1 = ((v.npt[0] - v.n[0]) - 1)
    v.j = 1
    return 224
  end
  blocks[234] = function()
    if truth((v.k <= v.i__1)) then return 236 else return 233 end
  end
  blocks[236] = function()
    v.hcol[v.k] = v.zero
    v.k = v.k + (1)
    return 234
  end
  blocks[308] = function()
    v.ksav = 0
    v.step = 0.0
    v.csave = 0.0
    v.ibdsav = 0
    v.stpsav = 0.0
    v.zmat_dim1 = v.npt[0]
    v.zmat_offset = (1 + v.zmat_dim1)
    v.zmat = add(v.zmat,(-v.zmat_offset))
    v.xpt_dim1 = v.npt[0]
    v.xpt_offset = (1 + v.xpt_dim1)
    v.xpt = add(v.xpt,(-v.xpt_offset))
    v.xopt = add(v.xopt,-1)
    v.bmat_dim1 = v.ndim[0]
    v.bmat_offset = (1 + v.bmat_dim1)
    v.bmat = add(v.bmat,(-v.bmat_offset))
    v.sl = add(v.sl,-1)
    v.su = add(v.su,-1)
    v.xnew = add(v.xnew,-1)
    v.xalt = add(v.xalt,-1)
    v.glag = add(v.glag,-1)
    v.hcol = add(v.hcol,-1)
    v.w = add(v.w,-1)
    v.half = .5
    v.one = 1.
    v.zero = 0.
    v.const__ = (v.one + math.sqrt(2.))
    v.i__1 = v.npt[0]
    v.k = 1
    return 234
  end
  local pc = 308
  while pc do pc = blocks[pc]() end
  return result
end

trsbox_ = function(n, npt, xpt, xopt, gopt, hq, pq, sl, su, delta, xnew, d__, gnew, xbdi, s, hs, hred, dsq, crvmin)
  local v = {}
  v.n = n
  v.npt = npt
  v.xpt = xpt
  v.xopt = xopt
  v.gopt = gopt
  v.hq = hq
  v.pq = pq
  v.sl = sl
  v.su = su
  v.delta = delta
  v.xnew = xnew
  v.d__ = d__
  v.gnew = gnew
  v.xbdi = xbdi
  v.s = s
  v.hs = hs
  v.hred = hred
  v.dsq = dsq
  v.crvmin = crvmin
  local blocks, result = {}, nil
  blocks[1] = function()
    return 168
  end
  blocks[2] = function()
    if truth((v.i__ <= v.i__2)) then return 4 else return 1 end
  end
  blocks[4] = function()
    v.hred[v.i__] = v.hs[v.i__]
    v.i__ = v.i__ + (1)
    return 2
  end
  blocks[6] = function()
    v.i__2 = v.n[0]
    v.i__ = 1
    return 2
  end
  blocks[7] = function()
    v.shs = v.zero
    v.dhs = v.zero
    v.dhd = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 108
  end
  blocks[8] = function()
    if truth((v.iterc > v.itcsav)) then return 7 else return 6 end
  end
  blocks[9] = function()
    v.resid = v.delsq
    v.ds = v.zero
    v.shs = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 251
  end
  blocks[10] = function()
    if truth((v.crvmin[0] ~= v.zero)) then return 9 else return 8 end
  end
  blocks[11] = function()
    if truth((v.k <= v.i__2)) then return 25 else return 10 end
  end
  blocks[12] = function()
    v.k = v.k + (1)
    return 11
  end
  blocks[13] = function()
    if truth((v.i__ <= v.i__1)) then return 15 else return 12 end
  end
  blocks[15] = function()
    v.hs[v.i__] = (v.hs[v.i__] + (v.temp * v.xpt[(v.k + (v.i__ * v.xpt_dim1))]))
    v.i__ = v.i__ + (1)
    return 13
  end
  blocks[18] = function()
    v.temp = (v.temp * v.pq[v.k])
    v.i__1 = v.n[0]
    v.i__ = 1
    return 13
  end
  blocks[19] = function()
    if truth((v.j <= v.i__1)) then return 21 else return 18 end
  end
  blocks[21] = function()
    v.temp = (v.temp + (v.xpt[(v.k + (v.j * v.xpt_dim1))] * v.s[v.j]))
    v.j = v.j + (1)
    return 19
  end
  blocks[24] = function()
    v.temp = v.zero
    v.i__1 = v.n[0]
    v.j = 1
    return 19
  end
  blocks[25] = function()
    if truth((v.pq[v.k] ~= v.zero)) then return 24 else return 12 end
  end
  blocks[27] = function()
    v.i__2 = v.npt[0]
    v.k = 1
    return 11
  end
  blocks[28] = function()
    if truth((v.j <= v.i__1)) then return 38 else return 27 end
  end
  blocks[29] = function()
    v.j = v.j + (1)
    return 28
  end
  blocks[30] = function()
    if truth((v.i__ <= v.i__2)) then return 35 else return 29 end
  end
  blocks[32] = function()
    v.hs[v.i__] = (v.hs[v.i__] + (v.hq[v.ih] * v.s[v.j]))
    v.i__ = v.i__ + (1)
    return 30
  end
  blocks[33] = function()
    v.hs[v.j] = (v.hs[v.j] + (v.hq[v.ih] * v.s[v.i__]))
    return 32
  end
  blocks[35] = function()
    v.ih = v.ih + (1)
    if truth((v.i__ < v.j)) then return 33 else return 32 end
  end
  blocks[38] = function()
    v.hs[v.j] = v.zero
    v.i__2 = v.j
    v.i__ = 1
    return 30
  end
  blocks[41] = function()
    v.ih = 0
    v.i__1 = v.n[0]
    v.j = 1
    return 28
  end
  blocks[42] = function()
    result = nil; return nil
  end
  blocks[43] = function()
    if truth((v.i__ <= v.i__1)) then return 54 else return 42 end
  end
  blocks[47] = function()
    v.d__[v.i__] = (v.xnew[v.i__] - v.xopt[v.i__])
    v.d__1 = v.d__[v.i__]
    v.dsq[0] = (v.dsq[0] + (v.d__1 * v.d__1))
    v.i__ = v.i__ + (1)
    return 43
  end
  blocks[48] = function()
    v.xnew[v.i__] = v.su[v.i__]
    return 47
  end
  blocks[49] = function()
    if truth((v.xbdi[v.i__] == v.one)) then return 48 else return 47 end
  end
  blocks[50] = function()
    v.xnew[v.i__] = v.sl[v.i__]
    return 49
  end
  blocks[54] = function()
    v.d__3 = (v.xopt[v.i__] + v.d__[v.i__]); v.d__4 = v.su[v.i__]
    v.d__1 = math.min(v.d__3, v.d__4); v.d__2 = v.sl[v.i__]
    v.xnew[v.i__] = math.max(v.d__1, v.d__2)
    if truth((v.xbdi[v.i__] == v.onemin)) then return 50 else return 49 end
  end
  blocks[57] = function()
    v.dsq[0] = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 43
  end
  blocks[58] = function()
    return 168
  end
  blocks[59] = function()
    if truth((v.sdec > (v.qred * .01))) then return 58 else return 57 end
  end
  blocks[62] = function()
    v.nact = v.nact + (1)
    v.xbdi[v.iact] = v.xsav
    return 187
  end
  blocks[64] = function()
    v.qred = (v.qred + v.sdec)
    if truth((truth((v.iact > 0)) and truth((v.isav == v.iu)))) then return 62 else return 59 end
  end
  blocks[65] = function()
    if truth((v.i__ <= v.i__1)) then return 73 else return 64 end
  end
  blocks[67] = function()
    v.hred[v.i__] = ((v.cth * v.hred[v.i__]) + (v.sth * v.hs[v.i__]))
    v.i__ = v.i__ + (1)
    return 65
  end
  blocks[71] = function()
    v.d__[v.i__] = ((v.cth * v.d__[v.i__]) + (v.sth * v.s[v.i__]))
    v.dredg = (v.dredg + (v.d__[v.i__] * v.gnew[v.i__]))
    v.d__1 = v.gnew[v.i__]
    v.gredsq = (v.gredsq + (v.d__1 * v.d__1))
    return 67
  end
  blocks[73] = function()
    v.gnew[v.i__] = ((v.gnew[v.i__] + ((v.cth - v.one) * v.hred[v.i__])) + (v.sth * v.hs[v.i__]))
    if truth((v.xbdi[v.i__] == v.zero)) then return 71 else return 67 end
  end
  blocks[77] = function()
    v.dredg = v.zero
    v.gredsq = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 65
  end
  blocks[78] = function()
    return 57
  end
  blocks[83] = function()
    v.cth = ((v.one - (v.angt * v.angt)) / (v.one + (v.angt * v.angt)))
    v.sth = ((v.angt + v.angt) / (v.one + (v.angt * v.angt)))
    v.temp = (v.shs + (v.angt * (((v.angt * v.dhd) - v.dhs) - v.dhs)))
    v.sdec = (v.sth * (((v.angt * v.dredg) - v.sredg) - ((v.half * v.sth) * v.temp)))
    if truth((v.sdec <= v.zero)) then return 78 else return 77 end
  end
  blocks[85] = function()
    v.temp = ((v.rdnext - v.rdprev) / (((v.redmax + v.redmax) - v.rdprev) - v.rdnext))
    v.angt = ((v.angbd * (v.isav + (v.half * v.temp))) / v.iu)
    return 83
  end
  blocks[86] = function()
    if truth((v.isav < v.iu)) then return 85 else return 83 end
  end
  blocks[87] = function()
    return 57
  end
  blocks[88] = function()
    if truth((v.isav == 0)) then return 87 else return 86 end
  end
  blocks[89] = function()
    if truth((v.i__ <= v.i__1)) then return 101 else return 88 end
  end
  blocks[91] = function()
    v.redsav = v.rednew
    v.i__ = v.i__ + (1)
    return 89
  end
  blocks[94] = function()
    v.redmax = v.rednew
    v.isav = v.i__
    v.rdprev = v.redsav
    return 91
  end
  blocks[95] = function()
    v.rdnext = v.rednew
    return 91
  end
  blocks[96] = function()
    if truth((v.i__ == (v.isav + 1))) then return 95 else return 91 end
  end
  blocks[101] = function()
    v.angt = ((v.angbd * v.i__) / v.iu)
    v.sth = ((v.angt + v.angt) / (v.one + (v.angt * v.angt)))
    v.temp = (v.shs + (v.angt * (((v.angt * v.dhd) - v.dhs) - v.dhs)))
    v.rednew = (v.sth * (((v.angt * v.dredg) - v.sredg) - ((v.half * v.sth) * v.temp)))
    if truth((v.rednew > v.redmax)) then return 94 else return 96 end
  end
  blocks[107] = function()
    v.redmax = v.zero
    v.isav = 0
    v.redsav = v.zero
    v.iu = trunc(((v.angbd * 17.) + 3.1))
    v.i__1 = v.iu
    v.i__ = 1
    return 89
  end
  blocks[108] = function()
    if truth((v.i__ <= v.i__1)) then return 113 else return 107 end
  end
  blocks[109] = function()
    v.i__ = v.i__ + (1)
    return 108
  end
  blocks[112] = function()
    v.shs = (v.shs + (v.s[v.i__] * v.hs[v.i__]))
    v.dhs = (v.dhs + (v.d__[v.i__] * v.hs[v.i__]))
    v.dhd = (v.dhd + (v.d__[v.i__] * v.hred[v.i__]))
    return 109
  end
  blocks[113] = function()
    if truth((v.xbdi[v.i__] == v.zero)) then return 112 else return 109 end
  end
  blocks[119] = function()
    return 41
  end
  blocks[120] = function()
    if truth((v.i__ <= v.i__1)) then return 151 else return 119 end
  end
  blocks[121] = function()
    v.i__ = v.i__ + (1)
    return 120
  end
  blocks[124] = function()
    v.angbd = (v.tempb / v.temp)
    v.iact = v.i__
    v.xsav = v.one
    return 121
  end
  blocks[126] = function()
    v.temp = (math.sqrt(v.temp) + v.s[v.i__])
    if truth(((v.angbd * v.temp) > v.tempb)) then return 124 else return 121 end
  end
  blocks[129] = function()
    v.d__1 = (v.su[v.i__] - v.xopt[v.i__])
    v.temp = (v.ssq - (v.d__1 * v.d__1))
    if truth((v.temp > v.zero)) then return 126 else return 121 end
  end
  blocks[132] = function()
    v.angbd = (v.tempa / v.temp)
    v.iact = v.i__
    v.xsav = v.onemin
    return 129
  end
  blocks[134] = function()
    v.temp = (math.sqrt(v.temp) - v.s[v.i__])
    if truth(((v.angbd * v.temp) > v.tempa)) then return 132 else return 129 end
  end
  blocks[140] = function()
    v.d__1 = v.d__[v.i__]
    v.d__2 = v.s[v.i__]
    v.ssq = ((v.d__1 * v.d__1) + (v.d__2 * v.d__2))
    v.d__1 = (v.xopt[v.i__] - v.sl[v.i__])
    v.temp = (v.ssq - (v.d__1 * v.d__1))
    if truth((v.temp > v.zero)) then return 134 else return 129 end
  end
  blocks[143] = function()
    v.nact = v.nact + (1)
    v.xbdi[v.i__] = v.onemin
    return 187
  end
  blocks[146] = function()
    v.nact = v.nact + (1)
    v.xbdi[v.i__] = v.one
    return 187
  end
  blocks[147] = function()
    if truth((v.tempb <= v.zero)) then return 146 else return 140 end
  end
  blocks[150] = function()
    v.tempa = ((v.xopt[v.i__] + v.d__[v.i__]) - v.sl[v.i__])
    v.tempb = ((v.su[v.i__] - v.xopt[v.i__]) - v.d__[v.i__])
    if truth((v.tempa <= v.zero)) then return 143 else return 147 end
  end
  blocks[151] = function()
    if truth((v.xbdi[v.i__] == v.zero)) then return 150 else return 121 end
  end
  blocks[156] = function()
    v.sredg = (-v.temp)
    v.angbd = v.one
    v.iact = 0
    v.i__1 = v.n[0]
    v.i__ = 1
    return 120
  end
  blocks[157] = function()
    if truth((v.i__ <= v.i__1)) then return 161 else return 156 end
  end
  blocks[158] = function()
    v.i__ = v.i__ + (1)
    return 157
  end
  blocks[159] = function()
    v.s[v.i__] = (((v.dredg * v.d__[v.i__]) - (v.dredsq * v.gnew[v.i__])) / v.temp)
    return 158
  end
  blocks[160] = function()
    v.s[v.i__] = v.zero
    return 158
  end
  blocks[161] = function()
    if truth((v.xbdi[v.i__] == v.zero)) then return 159 else return 160 end
  end
  blocks[164] = function()
    v.temp = math.sqrt(v.temp)
    v.i__1 = v.n[0]
    v.i__ = 1
    return 157
  end
  blocks[165] = function()
    return 57
  end
  blocks[168] = function()
    v.iterc = v.iterc + (1)
    v.temp = ((v.gredsq * v.dredsq) - (v.dredg * v.dredg))
    if truth((v.temp <= ((v.qred * 1e-4) * v.qred))) then return 165 else return 164 end
  end
  blocks[170] = function()
    v.itcsav = v.iterc
    return 41
  end
  blocks[171] = function()
    if truth((v.i__ <= v.i__1)) then return 180 else return 170 end
  end
  blocks[172] = function()
    v.i__ = v.i__ + (1)
    return 171
  end
  blocks[178] = function()
    v.d__1 = v.d__[v.i__]
    v.dredsq = (v.dredsq + (v.d__1 * v.d__1))
    v.dredg = (v.dredg + (v.d__[v.i__] * v.gnew[v.i__]))
    v.d__1 = v.gnew[v.i__]
    v.gredsq = (v.gredsq + (v.d__1 * v.d__1))
    v.s[v.i__] = v.d__[v.i__]
    return 172
  end
  blocks[179] = function()
    v.s[v.i__] = v.zero
    return 172
  end
  blocks[180] = function()
    if truth((v.xbdi[v.i__] == v.zero)) then return 178 else return 179 end
  end
  blocks[185] = function()
    v.dredsq = v.zero
    v.dredg = v.zero
    v.gredsq = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 171
  end
  blocks[186] = function()
    return 57
  end
  blocks[187] = function()
    if truth((v.nact >= (v.n[0] - 1))) then return 186 else return 185 end
  end
  blocks[188] = function()
    v.crvmin[0] = v.zero
    return 187
  end
  blocks[190] = function()
    v.beta = (v.gredsq / v.ggsav)
    return 282
  end
  blocks[191] = function()
    return 57
  end
  blocks[192] = function()
    if truth((v.sdec <= (v.qred * .01))) then return 191 else return 190 end
  end
  blocks[193] = function()
    return 57
  end
  blocks[194] = function()
    if truth((v.iterc == v.itermax)) then return 193 else return 192 end
  end
  blocks[195] = function()
    if truth((v.stplen < v.blen)) then return 194 else return 188 end
  end
  blocks[196] = function()
    return 283
  end
  blocks[197] = function()
    return 188
  end
  blocks[200] = function()
    v.d__1 = v.d__[v.iact]
    v.delsq = (v.delsq - (v.d__1 * v.d__1))
    if truth((v.delsq <= v.zero)) then return 197 else return 196 end
  end
  blocks[201] = function()
    v.xbdi[v.iact] = v.onemin
    return 200
  end
  blocks[204] = function()
    v.nact = v.nact + (1)
    v.xbdi[v.iact] = v.one
    if truth((v.s[v.iact] < v.zero)) then return 201 else return 200 end
  end
  blocks[205] = function()
    if truth((v.iact > 0)) then return 204 else return 195 end
  end
  blocks[208] = function()
    v.d__1 = (v.stplen * (v.ggsav - ((v.half * v.stplen) * v.shs)))
    v.sdec = math.max(v.d__1, v.zero)
    v.qred = (v.qred + v.sdec)
    return 205
  end
  blocks[209] = function()
    if truth((v.i__ <= v.i__1)) then return 215 else return 208 end
  end
  blocks[211] = function()
    v.d__[v.i__] = (v.d__[v.i__] + (v.stplen * v.s[v.i__]))
    v.i__ = v.i__ + (1)
    return 209
  end
  blocks[213] = function()
    v.d__1 = v.gnew[v.i__]
    v.gredsq = (v.gredsq + (v.d__1 * v.d__1))
    return 211
  end
  blocks[215] = function()
    v.gnew[v.i__] = (v.gnew[v.i__] + (v.stplen * v.hs[v.i__]))
    if truth((v.xbdi[v.i__] == v.zero)) then return 213 else return 211 end
  end
  blocks[219] = function()
    v.ggsav = v.gredsq
    v.gredsq = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 209
  end
  blocks[220] = function()
    v.crvmin[0] = v.temp
    return 219
  end
  blocks[222] = function()
    v.crvmin[0] = math.min(v.crvmin[0], v.temp)
    if truth((v.crvmin[0] == v.onemin)) then return 220 else return 219 end
  end
  blocks[225] = function()
    v.iterc = v.iterc + (1)
    v.temp = (v.shs / v.stepsq)
    if truth((truth((v.iact == 0)) and truth((v.temp > v.zero)))) then return 222 else return 219 end
  end
  blocks[227] = function()
    v.sdec = v.zero
    if truth((v.stplen > v.zero)) then return 225 else return 205 end
  end
  blocks[228] = function()
    if truth((v.i__ <= v.i__1)) then return 237 else return 227 end
  end
  blocks[229] = function()
    v.i__ = v.i__ + (1)
    return 228
  end
  blocks[231] = function()
    v.stplen = v.temp
    v.iact = v.i__
    return 229
  end
  blocks[232] = function()
    if truth((v.temp < v.stplen)) then return 231 else return 229 end
  end
  blocks[233] = function()
    v.temp = ((v.su[v.i__] - v.xsum) / v.s[v.i__])
    return 232
  end
  blocks[234] = function()
    v.temp = ((v.sl[v.i__] - v.xsum) / v.s[v.i__])
    return 232
  end
  blocks[236] = function()
    v.xsum = (v.xopt[v.i__] + v.d__[v.i__])
    if truth((v.s[v.i__] > v.zero)) then return 233 else return 234 end
  end
  blocks[237] = function()
    if truth((v.s[v.i__] ~= v.zero)) then return 236 else return 229 end
  end
  blocks[240] = function()
    v.iact = 0
    v.i__1 = v.n[0]
    v.i__ = 1
    return 228
  end
  blocks[242] = function()
    v.d__1 = v.blen; v.d__2 = (v.gredsq / v.shs)
    v.stplen = math.min(v.d__1, v.d__2)
    return 240
  end
  blocks[244] = function()
    v.stplen = v.blen
    if truth((v.shs > v.zero)) then return 242 else return 240 end
  end
  blocks[245] = function()
    v.blen = ((v.temp - v.ds) / v.stepsq)
    return 244
  end
  blocks[246] = function()
    v.blen = (v.resid / (v.temp + v.ds))
    return 244
  end
  blocks[248] = function()
    v.temp = math.sqrt(((v.stepsq * v.resid) + (v.ds * v.ds)))
    if truth((v.ds < v.zero)) then return 245 else return 246 end
  end
  blocks[249] = function()
    return 188
  end
  blocks[250] = function()
    if truth((v.resid <= v.zero)) then return 249 else return 248 end
  end
  blocks[251] = function()
    if truth((v.i__ <= v.i__1)) then return 257 else return 250 end
  end
  blocks[252] = function()
    v.i__ = v.i__ + (1)
    return 251
  end
  blocks[256] = function()
    v.d__1 = v.d__[v.i__]
    v.resid = (v.resid - (v.d__1 * v.d__1))
    v.ds = (v.ds + (v.s[v.i__] * v.d__[v.i__]))
    v.shs = (v.shs + (v.s[v.i__] * v.hs[v.i__]))
    return 252
  end
  blocks[257] = function()
    if truth((v.xbdi[v.i__] == v.zero)) then return 256 else return 252 end
  end
  blocks[263] = function()
    return 41
  end
  blocks[264] = function()
    return 57
  end
  blocks[265] = function()
    if truth(((v.gredsq * v.delsq) <= ((v.qred * 1e-4) * v.qred))) then return 264 else return 263 end
  end
  blocks[267] = function()
    v.gredsq = v.stepsq
    v.itermax = ((v.iterc + v.n[0]) - v.nact)
    return 265
  end
  blocks[268] = function()
    if truth((v.beta == v.zero)) then return 267 else return 265 end
  end
  blocks[269] = function()
    return 57
  end
  blocks[270] = function()
    if truth((v.stepsq == v.zero)) then return 269 else return 268 end
  end
  blocks[271] = function()
    if truth((v.i__ <= v.i__1)) then return 279 else return 270 end
  end
  blocks[274] = function()
    v.d__1 = v.s[v.i__]
    v.stepsq = (v.stepsq + (v.d__1 * v.d__1))
    v.i__ = v.i__ + (1)
    return 271
  end
  blocks[275] = function()
    v.s[v.i__] = v.zero
    return 274
  end
  blocks[276] = function()
    v.s[v.i__] = (-v.gnew[v.i__])
    return 274
  end
  blocks[277] = function()
    v.s[v.i__] = ((v.beta * v.s[v.i__]) - v.gnew[v.i__])
    return 274
  end
  blocks[278] = function()
    if truth((v.beta == v.zero)) then return 276 else return 277 end
  end
  blocks[279] = function()
    if truth((v.xbdi[v.i__] ~= v.zero)) then return 275 else return 278 end
  end
  blocks[282] = function()
    v.stepsq = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 271
  end
  blocks[283] = function()
    v.beta = v.zero
    return 282
  end
  blocks[286] = function()
    v.delsq = (v.delta[0] * v.delta[0])
    v.qred = v.zero
    v.crvmin[0] = v.onemin
    return 283
  end
  blocks[287] = function()
    if truth((v.i__ <= v.i__1)) then return 299 else return 286 end
  end
  blocks[290] = function()
    v.d__[v.i__] = v.zero
    v.gnew[v.i__] = v.gopt[v.i__]
    v.i__ = v.i__ + (1)
    return 287
  end
  blocks[291] = function()
    v.nact = v.nact + (1)
    return 290
  end
  blocks[292] = function()
    if truth((v.xbdi[v.i__] ~= v.zero)) then return 291 else return 290 end
  end
  blocks[293] = function()
    v.xbdi[v.i__] = v.onemin
    return 292
  end
  blocks[294] = function()
    if truth((v.gopt[v.i__] >= v.zero)) then return 293 else return 292 end
  end
  blocks[295] = function()
    v.xbdi[v.i__] = v.one
    return 292
  end
  blocks[296] = function()
    if truth((v.gopt[v.i__] <= v.zero)) then return 295 else return 292 end
  end
  blocks[297] = function()
    if truth((v.xopt[v.i__] >= v.su[v.i__])) then return 296 else return 292 end
  end
  blocks[299] = function()
    v.xbdi[v.i__] = v.zero
    if truth((v.xopt[v.i__] <= v.sl[v.i__])) then return 294 else return 297 end
  end
  blocks[378] = function()
    v.iact = 0
    v.xsav = 0.0
    v.angbd = 0.0
    v.dredg = 0.0
    v.sredg = 0.0
    v.ggsav = 0.0
    v.dredsq = 0.0
    v.gredsq = 0.0
    v.itcsav = 0
    v.rdnext = 0.0
    v.itermax = 0
    v.xpt_dim1 = v.npt[0]
    v.xpt_offset = (1 + v.xpt_dim1)
    v.xpt = add(v.xpt,(-v.xpt_offset))
    v.xopt = add(v.xopt,-1)
    v.gopt = add(v.gopt,-1)
    v.hq = add(v.hq,-1)
    v.pq = add(v.pq,-1)
    v.sl = add(v.sl,-1)
    v.su = add(v.su,-1)
    v.xnew = add(v.xnew,-1)
    v.d__ = add(v.d__,-1)
    v.gnew = add(v.gnew,-1)
    v.xbdi = add(v.xbdi,-1)
    v.s = add(v.s,-1)
    v.hs = add(v.hs,-1)
    v.hred = add(v.hred,-1)
    v.half = .5
    v.one = 1.
    v.onemin = (-1.)
    v.zero = 0.
    v.iterc = 0
    v.nact = 0
    v.i__1 = v.n[0]
    v.i__ = 1
    return 287
  end
  local pc = 378
  while pc do pc = blocks[pc]() end
  return result
end

prelim_ = function(n, npt, x, xl, xu, rhobeg, stop, calfun, calfun_data, xbase, xpt, fval, gopt, hq, pq, bmat, zmat, ndim, sl, su, kopt)
  local v = {}
  v.n = n
  v.npt = npt
  v.x = x
  v.xl = xl
  v.xu = xu
  v.rhobeg = rhobeg
  v.stop = stop
  v.calfun = calfun
  v.calfun_data = calfun_data
  v.xbase = xbase
  v.xpt = xpt
  v.fval = fval
  v.gopt = gopt
  v.hq = hq
  v.pq = pq
  v.bmat = bmat
  v.zmat = zmat
  v.ndim = ndim
  v.sl = sl
  v.su = su
  v.kopt = kopt
  local blocks, result = {}, nil
  blocks[1] = function()
    result = NLOPT_SUCCESS; return nil
  end
  blocks[2] = function()
    return 92
  end
  blocks[3] = function()
    if truth((v.nf < v.npt[0])) then return 2 else return 1 end
  end
  blocks[4] = function()
    result = NLOPT_FORCED_STOP; return nil
  end
  blocks[5] = function()
    result = NLOPT_MINF_MAX_REACHED; return nil
  end
  blocks[6] = function()
    result = NLOPT_MAXEVAL_REACHED; return nil
  end
  blocks[7] = function()
    result = NLOPT_MAXTIME_REACHED; return nil
  end
  blocks[8] = function()
    if truth(nlopt_stop_time(v.stop)) then return 7 else return 3 end
  end
  blocks[9] = function()
    if truth(nlopt_stop_evals(v.stop)) then return 6 else return 8 end
  end
  blocks[10] = function()
    if truth((v.f < v.stop.minf_max)) then return 5 else return 9 end
  end
  blocks[11] = function()
    if truth(nlopt_stop_forced(v.stop)) then return 4 else return 10 end
  end
  blocks[14] = function()
    v.bmat[((v.nfm * v.bmat_dim1) + 1)] = ((-v.one) / v.stepa)
    v.bmat[(v.nf + (v.nfm * v.bmat_dim1))] = (v.one / v.stepa)
    v.bmat[((v.npt[0] + v.nfm) + (v.nfm * v.bmat_dim1))] = ((-v.half) * v.rhosq)
    return 11
  end
  blocks[16] = function()
    v.gopt[v.nfm] = ((v.f - v.fbeg) / v.stepa)
    if truth((v.npt[0] < (v.nf + v.n[0]))) then return 14 else return 11 end
  end
  blocks[22] = function()
    v.bmat[((v.nfx * v.bmat_dim1) + 1)] = ((-(v.stepa + v.stepb)) / (v.stepa * v.stepb))
    v.bmat[(v.nf + (v.nfx * v.bmat_dim1))] = ((-v.half) / v.xpt[((v.nf - v.n[0]) + (v.nfx * v.xpt_dim1))])
    v.bmat[((v.nf - v.n[0]) + (v.nfx * v.bmat_dim1))] = ((-v.bmat[((v.nfx * v.bmat_dim1) + 1)]) - v.bmat[(v.nf + (v.nfx * v.bmat_dim1))])
    v.zmat[((v.nfx * v.zmat_dim1) + 1)] = (math.sqrt(v.two) / (v.stepa * v.stepb))
    v.zmat[(v.nf + (v.nfx * v.zmat_dim1))] = (math.sqrt(v.half) / v.rhosq)
    v.zmat[((v.nf - v.n[0]) + (v.nfx * v.zmat_dim1))] = ((-v.zmat[((v.nfx * v.zmat_dim1) + 1)]) - v.zmat[(v.nf + (v.nfx * v.zmat_dim1))])
    return 11
  end
  blocks[24] = function()
    v.xpt[((v.nf - v.n[0]) + (v.nfx * v.xpt_dim1))] = v.stepb
    v.xpt[(v.nf + (v.nfx * v.xpt_dim1))] = v.stepa
    return 22
  end
  blocks[25] = function()
    v.kopt[0] = (v.nf - v.n[0])
    return 24
  end
  blocks[28] = function()
    v.fval[v.nf] = v.fval[(v.nf - v.n[0])]
    v.fval[(v.nf - v.n[0])] = v.f
    if truth((v.kopt[0] == v.nf)) then return 25 else return 24 end
  end
  blocks[29] = function()
    if truth((v.f < v.fval[(v.nf - v.n[0])])) then return 28 else return 22 end
  end
  blocks[35] = function()
    v.ih = trunc(((v.nfx * (v.nfx + 1)))/(2))
    v.temp = ((v.f - v.fbeg) / v.stepb)
    v.diff = (v.stepb - v.stepa)
    v.hq[v.ih] = ((v.two * (v.temp - v.gopt[v.nfx])) / v.diff)
    v.gopt[v.nfx] = (((v.gopt[v.nfx] * v.stepb) - (v.temp * v.stepa)) / v.diff)
    if truth(((v.stepa * v.stepb) < v.zero)) then return 29 else return 22 end
  end
  blocks[36] = function()
    if truth((v.nf >= (v.n[0] + 2))) then return 35 else return 11 end
  end
  blocks[37] = function()
    if truth((truth((v.nf >= 2)) and truth((v.nf <= (v.n[0] + 1))))) then return 16 else return 36 end
  end
  blocks[44] = function()
    v.ih = (trunc(((v.ipt * (v.ipt - 1)))/(2)) + v.jpt)
    v.zmat[((v.nfx * v.zmat_dim1) + 1)] = v.recip
    v.zmat[(v.nf + (v.nfx * v.zmat_dim1))] = v.recip
    v.zmat[((v.ipt + 1) + (v.nfx * v.zmat_dim1))] = (-v.recip)
    v.zmat[((v.jpt + 1) + (v.nfx * v.zmat_dim1))] = (-v.recip)
    v.temp = (v.xpt[(v.nf + (v.ipt * v.xpt_dim1))] * v.xpt[(v.nf + (v.jpt * v.xpt_dim1))])
    v.hq[v.ih] = ((((v.fbeg - v.fval[(v.ipt + 1)]) - v.fval[(v.jpt + 1)]) + v.f) / v.temp)
    return 11
  end
  blocks[45] = function()
    if truth((v.nf <= ((v.n[0] * 2^(1)) + 1))) then return 37 else return 44 end
  end
  blocks[47] = function()
    v.fbeg = v.f
    v.kopt[0] = 1
    return 45
  end
  blocks[48] = function()
    v.kopt[0] = v.nf
    return 45
  end
  blocks[49] = function()
    if truth((v.f < v.fval[v.kopt[0]])) then return 48 else return 45 end
  end
  blocks[53] = function()
    v.stop.nevals_p[0] = v.stop.nevals_p[0] + (1)
    v.f = v.calfun(v.n[0], add(v.x,1), v.calfun_data)
    v.fval[v.nf] = v.f
    if truth((v.nf == 1)) then return 47 else return 49 end
  end
  blocks[54] = function()
    if truth((v.j <= v.i__1)) then return 62 else return 53 end
  end
  blocks[55] = function()
    v.j = v.j + (1)
    return 54
  end
  blocks[56] = function()
    v.x[v.j] = v.xu[v.j]
    return 55
  end
  blocks[57] = function()
    if truth((v.xpt[(v.nf + (v.j * v.xpt_dim1))] == v.su[v.j])) then return 56 else return 55 end
  end
  blocks[58] = function()
    v.x[v.j] = v.xl[v.j]
    return 57
  end
  blocks[62] = function()
    v.d__3 = v.xl[v.j]; v.d__4 = (v.xbase[v.j] + v.xpt[(v.nf + (v.j * v.xpt_dim1))])
    v.d__1 = math.max(v.d__3, v.d__4); v.d__2 = v.xu[v.j]
    v.x[v.j] = math.min(v.d__1, v.d__2)
    if truth((v.xpt[(v.nf + (v.j * v.xpt_dim1))] == v.sl[v.j])) then return 58 else return 57 end
  end
  blocks[64] = function()
    v.i__1 = v.n[0]
    v.j = 1
    return 54
  end
  blocks[65] = function()
    v.xpt[(v.nf + (v.nfm * v.xpt_dim1))] = v.stepa
    return 64
  end
  blocks[66] = function()
    v.stepa = (-v.stepa)
    return 65
  end
  blocks[68] = function()
    v.stepa = v.rhobeg[0]
    if truth((v.su[v.nfm] == v.zero)) then return 66 else return 65 end
  end
  blocks[69] = function()
    v.xpt[(v.nf + (v.nfx * v.xpt_dim1))] = v.stepb
    return 64
  end
  blocks[71] = function()
    v.d__1 = ((-v.two) * v.rhobeg[0]); v.d__2 = v.sl[v.nfx]
    v.stepb = math.max(v.d__1, v.d__2)
    return 69
  end
  blocks[72] = function()
    if truth((v.su[v.nfx] == v.zero)) then return 71 else return 69 end
  end
  blocks[74] = function()
    v.d__1 = (v.two * v.rhobeg[0]); v.d__2 = v.su[v.nfx]
    v.stepb = math.min(v.d__1, v.d__2)
    return 72
  end
  blocks[77] = function()
    v.stepa = v.xpt[((v.nf - v.n[0]) + (v.nfx * v.xpt_dim1))]
    v.stepb = (-v.rhobeg[0])
    if truth((v.sl[v.nfx] == v.zero)) then return 74 else return 72 end
  end
  blocks[78] = function()
    if truth((v.nfm > v.n[0])) then return 77 else return 64 end
  end
  blocks[79] = function()
    if truth((truth((v.nfm >= 1)) and truth((v.nfm <= v.n[0])))) then return 68 else return 78 end
  end
  blocks[81] = function()
    v.xpt[(v.nf + (v.ipt * v.xpt_dim1))] = v.xpt[((v.ipt + 1) + (v.ipt * v.xpt_dim1))]
    v.xpt[(v.nf + (v.jpt * v.xpt_dim1))] = v.xpt[((v.jpt + 1) + (v.jpt * v.xpt_dim1))]
    return 64
  end
  blocks[84] = function()
    v.itemp = v.jpt
    v.jpt = (v.ipt - v.n[0])
    v.ipt = v.itemp
    return 81
  end
  blocks[88] = function()
    v.itemp = trunc(((v.nfm - v.np))/(v.n[0]))
    v.jpt = ((v.nfm - (v.itemp * v.n[0])) - v.n[0])
    v.ipt = (v.jpt + v.itemp)
    if truth((v.ipt > v.n[0])) then return 84 else return 81 end
  end
  blocks[92] = function()
    v.nfm = v.nf
    v.nfx = (v.nf - v.n[0])
    v.nf = v.nf + (1)
    if truth((v.nfm <= (v.n[0] * 2^(1)))) then return 79 else return 88 end
  end
  blocks[93] = function()
    v.nf = 0
    return 92
  end
  blocks[94] = function()
    if truth((v.k <= v.i__2)) then return 101 else return 93 end
  end
  blocks[95] = function()
    v.k = v.k + (1)
    return 94
  end
  blocks[96] = function()
    if truth((v.j <= v.i__1)) then return 98 else return 95 end
  end
  blocks[98] = function()
    v.zmat[(v.k + (v.j * v.zmat_dim1))] = v.zero
    v.j = v.j + (1)
    return 96
  end
  blocks[101] = function()
    v.pq[v.k] = v.zero
    v.i__1 = (v.npt[0] - v.np)
    v.j = 1
    return 96
  end
  blocks[103] = function()
    v.i__2 = v.npt[0]
    v.k = 1
    return 94
  end
  blocks[104] = function()
    if truth((v.ih <= v.i__2)) then return 106 else return 103 end
  end
  blocks[106] = function()
    v.hq[v.ih] = v.zero
    v.ih = v.ih + (1)
    return 104
  end
  blocks[108] = function()
    v.i__2 = trunc(((v.n[0] * v.np))/(2))
    v.ih = 1
    return 104
  end
  blocks[109] = function()
    if truth((v.j <= v.i__1)) then return 121 else return 108 end
  end
  blocks[110] = function()
    v.j = v.j + (1)
    return 109
  end
  blocks[111] = function()
    if truth((v.i__ <= v.i__2)) then return 113 else return 110 end
  end
  blocks[113] = function()
    v.bmat[(v.i__ + (v.j * v.bmat_dim1))] = v.zero
    v.i__ = v.i__ + (1)
    return 111
  end
  blocks[115] = function()
    v.i__2 = v.ndim[0]
    v.i__ = 1
    return 111
  end
  blocks[116] = function()
    if truth((v.k <= v.i__2)) then return 118 else return 115 end
  end
  blocks[118] = function()
    v.xpt[(v.k + (v.j * v.xpt_dim1))] = v.zero
    v.k = v.k + (1)
    return 116
  end
  blocks[121] = function()
    v.xbase[v.j] = v.x[v.j]
    v.i__2 = v.npt[0]
    v.k = 1
    return 116
  end
  blocks[184] = function()
    v.ipt = 0
    v.jpt = 0
    v.fbeg = 0.0
    v.stepa = 0.0
    v.stepb = 0.0
    v.zmat_dim1 = v.npt[0]
    v.zmat_offset = (1 + v.zmat_dim1)
    v.zmat = add(v.zmat,(-v.zmat_offset))
    v.xpt_dim1 = v.npt[0]
    v.xpt_offset = (1 + v.xpt_dim1)
    v.xpt = add(v.xpt,(-v.xpt_offset))
    v.x = add(v.x,-1)
    v.xl = add(v.xl,-1)
    v.xu = add(v.xu,-1)
    v.xbase = add(v.xbase,-1)
    v.fval = add(v.fval,-1)
    v.gopt = add(v.gopt,-1)
    v.hq = add(v.hq,-1)
    v.pq = add(v.pq,-1)
    v.bmat_dim1 = v.ndim[0]
    v.bmat_offset = (1 + v.bmat_dim1)
    v.bmat = add(v.bmat,(-v.bmat_offset))
    v.sl = add(v.sl,-1)
    v.su = add(v.su,-1)
    v.half = .5
    v.one = 1.
    v.two = 2.
    v.zero = 0.
    v.rhosq = (v.rhobeg[0] * v.rhobeg[0])
    v.recip = (v.one / v.rhosq)
    v.np = (v.n[0] + 1)
    v.i__1 = v.n[0]
    v.j = 1
    return 109
  end
  local pc = 184
  while pc do pc = blocks[pc]() end
  return result
end

bobyqb_ = function(n, npt, x, xl, xu, rhobeg, rhoend, stop, calfun, calfun_data, minf, xbase, xpt, fval, xopt, gopt, hq, pq, bmat, zmat, ndim, sl, su, xnew, xalt, d__, vlag, w)
  local v = {}
  v.n = n
  v.npt = npt
  v.x = x
  v.xl = xl
  v.xu = xu
  v.rhobeg = rhobeg
  v.rhoend = rhoend
  v.stop = stop
  v.calfun = calfun
  v.calfun_data = calfun_data
  v.minf = minf
  v.xbase = xbase
  v.xpt = xpt
  v.fval = fval
  v.xopt = xopt
  v.gopt = gopt
  v.hq = hq
  v.pq = pq
  v.bmat = bmat
  v.zmat = zmat
  v.ndim = ndim
  v.sl = sl
  v.su = su
  v.xnew = xnew
  v.xalt = xalt
  v.d__ = d__
  v.vlag = vlag
  v.w = w
  local blocks, result = {}, nil
  blocks[3] = function()
    v.f = v.fval[v.kopt]
    v.minf[0] = v.f
    result = v.rc; return nil
  end
  blocks[4] = function()
    if truth((v.i__ <= v.i__1)) then return 12 else return 3 end
  end
  blocks[5] = function()
    v.i__ = v.i__ + (1)
    return 4
  end
  blocks[6] = function()
    v.x[v.i__] = v.xu[v.i__]
    return 5
  end
  blocks[7] = function()
    if truth((v.xopt[v.i__] == v.su[v.i__])) then return 6 else return 5 end
  end
  blocks[8] = function()
    v.x[v.i__] = v.xl[v.i__]
    return 7
  end
  blocks[12] = function()
    v.d__3 = v.xl[v.i__]; v.d__4 = (v.xbase[v.i__] + v.xopt[v.i__])
    v.d__1 = math.max(v.d__3, v.d__4); v.d__2 = v.xu[v.i__]
    v.x[v.i__] = math.min(v.d__1, v.d__2)
    if truth((v.xopt[v.i__] == v.sl[v.i__])) then return 8 else return 7 end
  end
  blocks[14] = function()
    v.i__1 = v.n[0]
    v.i__ = 1
    return 4
  end
  blocks[15] = function()
    return 362
  end
  blocks[16] = function()
    if truth((v.ntrits == (-1))) then return 15 else return 14 end
  end
  blocks[20] = function()
    v.delta = math.max(v.delta, v.rho)
    v.ntrits = 0
    v.nfsav = v.stop.nevals_p[0]
    return 647
  end
  blocks[21] = function()
    v.rho = v.rhoend[0]
    return 20
  end
  blocks[22] = function()
    v.rho = (math.sqrt(v.ratio) * v.rhoend[0])
    return 20
  end
  blocks[23] = function()
    v.rho = (v.tenth * v.rho)
    return 20
  end
  blocks[24] = function()
    if truth((v.ratio <= 250.)) then return 22 else return 23 end
  end
  blocks[27] = function()
    v.delta = (v.half * v.rho)
    v.ratio = (v.rho / v.rhoend[0])
    if truth((v.ratio <= 16.)) then return 21 else return 24 end
  end
  blocks[28] = function()
    if truth((v.rho > v.rhoend[0])) then return 27 else return 16 end
  end
  blocks[29] = function()
    return 647
  end
  blocks[30] = function()
    if truth((math.max(v.delta, v.dnorm) > v.rho)) then return 29 else return 28 end
  end
  blocks[31] = function()
    return 647
  end
  blocks[32] = function()
    if truth((v.ratio > v.zero)) then return 31 else return 30 end
  end
  blocks[33] = function()
    return 28
  end
  blocks[34] = function()
    if truth((v.ntrits == (-1))) then return 33 else return 32 end
  end
  blocks[40] = function()
    v.ntrits = 0
    v.d__2 = (v.tenth * v.dist)
    v.d__1 = math.min(v.d__2, v.delta)
    v.adelt = math.max(v.d__1, v.rho)
    v.dsq = (v.adelt * v.adelt)
    return 610
  end
  blocks[41] = function()
    v.delta = v.rho
    return 40
  end
  blocks[44] = function()
    v.d__1 = (v.tenth * v.delta); v.d__2 = (v.half * v.dist)
    v.delta = math.min(v.d__1, v.d__2)
    if truth((v.delta <= (v.rho * 1.5))) then return 41 else return 40 end
  end
  blocks[46] = function()
    v.dist = math.sqrt(v.distsq)
    if truth((v.ntrits == (-1))) then return 44 else return 40 end
  end
  blocks[47] = function()
    if truth((v.knew > 0)) then return 46 else return 34 end
  end
  blocks[48] = function()
    if truth((v.k <= v.i__1)) then return 59 else return 47 end
  end
  blocks[49] = function()
    v.k = v.k + (1)
    return 48
  end
  blocks[51] = function()
    v.knew = v.k
    v.distsq = v.sum
    return 49
  end
  blocks[52] = function()
    if truth((v.sum > v.distsq)) then return 51 else return 49 end
  end
  blocks[53] = function()
    if truth((v.j <= v.i__2)) then return 56 else return 52 end
  end
  blocks[56] = function()
    v.d__1 = (v.xpt[(v.k + (v.j * v.xpt_dim1))] - v.xopt[v.j])
    v.sum = (v.sum + (v.d__1 * v.d__1))
    v.j = v.j + (1)
    return 53
  end
  blocks[59] = function()
    v.sum = v.zero
    v.i__2 = v.n[0]
    v.j = 1
    return 53
  end
  blocks[62] = function()
    v.knew = 0
    v.i__1 = v.npt[0]
    v.k = 1
    return 48
  end
  blocks[66] = function()
    v.d__3 = (v.two * v.delta)
    v.d__4 = (v.ten * v.rho)
    v.d__1 = (v.d__3 * v.d__3); v.d__2 = (v.d__4 * v.d__4)
    v.distsq = math.max(v.d__1, v.d__2)
    return 62
  end
  blocks[67] = function()
    return 647
  end
  blocks[68] = function()
    if truth((v.f <= (v.fopt + (v.tenth * v.vquad)))) then return 67 else return 66 end
  end
  blocks[69] = function()
    return 647
  end
  blocks[70] = function()
    if truth((v.ntrits == 0)) then return 69 else return 68 end
  end
  blocks[71] = function()
    if truth((v.i__ <= v.i__1)) then return 79 else return 70 end
  end
  blocks[73] = function()
    v.itest = 0
    v.i__ = v.i__ + (1)
    return 71
  end
  blocks[74] = function()
    v.hq[v.i__] = v.zero
    return 73
  end
  blocks[75] = function()
    if truth((v.i__ <= v.nh)) then return 74 else return 73 end
  end
  blocks[76] = function()
    v.pq[v.i__] = v.w[(v.npt[0] + v.i__)]
    return 75
  end
  blocks[77] = function()
    if truth((v.i__ <= v.npt[0])) then return 76 else return 75 end
  end
  blocks[78] = function()
    v.gopt[v.i__] = v.vlag[(v.npt[0] + v.i__)]
    return 77
  end
  blocks[79] = function()
    if truth((v.i__ <= v.n[0])) then return 78 else return 77 end
  end
  blocks[81] = function()
    v.i__1 = math.max(v.npt[0], v.nh)
    v.i__ = 1
    return 71
  end
  blocks[82] = function()
    if truth((v.itest >= 3)) then return 81 else return 70 end
  end
  blocks[83] = function()
    v.itest = 0
    return 82
  end
  blocks[85] = function()
    v.itest = v.itest + (1)
    if truth((v.gqsq < (v.ten * v.gisq))) then return 83 else return 82 end
  end
  blocks[86] = function()
    if truth((v.i__ <= v.i__1)) then return 109 else return 85 end
  end
  blocks[88] = function()
    v.vlag[(v.npt[0] + v.i__)] = v.sum
    v.i__ = v.i__ + (1)
    return 86
  end
  blocks[93] = function()
    v.d__2 = v.zero; v.d__3 = v.gopt[v.i__]
    v.d__1 = math.min(v.d__2, v.d__3)
    v.gqsq = (v.gqsq + (v.d__1 * v.d__1))
    v.d__1 = math.min(v.zero, v.sum)
    v.gisq = (v.gisq + (v.d__1 * v.d__1))
    return 88
  end
  blocks[98] = function()
    v.d__2 = v.zero; v.d__3 = v.gopt[v.i__]
    v.d__1 = math.max(v.d__2, v.d__3)
    v.gqsq = (v.gqsq + (v.d__1 * v.d__1))
    v.d__1 = math.max(v.zero, v.sum)
    v.gisq = (v.gisq + (v.d__1 * v.d__1))
    return 88
  end
  blocks[101] = function()
    v.d__1 = v.gopt[v.i__]
    v.gqsq = (v.gqsq + (v.d__1 * v.d__1))
    v.gisq = (v.gisq + (v.sum * v.sum))
    return 88
  end
  blocks[102] = function()
    if truth((v.xopt[v.i__] == v.su[v.i__])) then return 98 else return 101 end
  end
  blocks[103] = function()
    if truth((v.xopt[v.i__] == v.sl[v.i__])) then return 93 else return 102 end
  end
  blocks[104] = function()
    if truth((v.k <= v.i__2)) then return 106 else return 103 end
  end
  blocks[106] = function()
    v.sum = ((v.sum + (v.bmat[(v.k + (v.i__ * v.bmat_dim1))] * v.vlag[v.k])) + (v.xpt[(v.k + (v.i__ * v.xpt_dim1))] * v.w[v.k]))
    v.k = v.k + (1)
    return 104
  end
  blocks[109] = function()
    v.sum = v.zero
    v.i__2 = v.npt[0]
    v.k = 1
    return 104
  end
  blocks[113] = function()
    v.gqsq = v.zero
    v.gisq = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 86
  end
  blocks[114] = function()
    if truth((v.k <= v.i__1)) then return 123 else return 113 end
  end
  blocks[117] = function()
    v.w[(v.k + v.npt[0])] = v.w[v.k]
    v.w[v.k] = (v.sum * v.w[v.k])
    v.k = v.k + (1)
    return 114
  end
  blocks[118] = function()
    if truth((v.j <= v.i__2)) then return 120 else return 117 end
  end
  blocks[120] = function()
    v.sum = (v.sum + (v.xpt[(v.k + (v.j * v.xpt_dim1))] * v.xopt[v.j]))
    v.j = v.j + (1)
    return 118
  end
  blocks[123] = function()
    v.sum = v.zero
    v.i__2 = v.n[0]
    v.j = 1
    return 118
  end
  blocks[125] = function()
    v.i__1 = v.npt[0]
    v.k = 1
    return 114
  end
  blocks[126] = function()
    if truth((v.j <= v.i__2)) then return 138 else return 125 end
  end
  blocks[127] = function()
    v.j = v.j + (1)
    return 126
  end
  blocks[128] = function()
    if truth((v.k <= v.i__1)) then return 130 else return 127 end
  end
  blocks[130] = function()
    v.w[v.k] = (v.w[v.k] + (v.sum * v.zmat[(v.k + (v.j * v.zmat_dim1))]))
    v.k = v.k + (1)
    return 128
  end
  blocks[132] = function()
    v.i__1 = v.npt[0]
    v.k = 1
    return 128
  end
  blocks[133] = function()
    if truth((v.k <= v.i__1)) then return 135 else return 132 end
  end
  blocks[135] = function()
    v.sum = (v.sum + (v.zmat[(v.k + (v.j * v.zmat_dim1))] * v.vlag[v.k]))
    v.k = v.k + (1)
    return 133
  end
  blocks[138] = function()
    v.sum = v.zero
    v.i__1 = v.npt[0]
    v.k = 1
    return 133
  end
  blocks[140] = function()
    v.i__2 = v.nptm
    v.j = 1
    return 126
  end
  blocks[141] = function()
    if truth((v.k <= v.i__2)) then return 144 else return 140 end
  end
  blocks[144] = function()
    v.vlag[v.k] = (v.fval[v.k] - v.fval[v.kopt])
    v.w[v.k] = v.zero
    v.k = v.k + (1)
    return 141
  end
  blocks[146] = function()
    v.i__2 = v.npt[0]
    v.k = 1
    return 141
  end
  blocks[147] = function()
    if truth((v.ntrits > 0)) then return 146 else return 70 end
  end
  blocks[149] = function()
    v.rc = NLOPT_FTOL_REACHED
    return 14
  end
  blocks[150] = function()
    if truth(nlopt_stop_ftol(v.stop, v.f, v.fopt)) then return 149 else return 147 end
  end
  blocks[151] = function()
    if truth((v.k <= v.i__1)) then return 164 else return 150 end
  end
  blocks[152] = function()
    v.k = v.k + (1)
    return 151
  end
  blocks[153] = function()
    if truth((v.i__ <= v.i__2)) then return 155 else return 152 end
  end
  blocks[155] = function()
    v.gopt[v.i__] = (v.gopt[v.i__] + (v.temp * v.xpt[(v.k + (v.i__ * v.xpt_dim1))]))
    v.i__ = v.i__ + (1)
    return 153
  end
  blocks[158] = function()
    v.temp = (v.pq[v.k] * v.temp)
    v.i__2 = v.n[0]
    v.i__ = 1
    return 153
  end
  blocks[159] = function()
    if truth((v.j <= v.i__2)) then return 161 else return 158 end
  end
  blocks[161] = function()
    v.temp = (v.temp + (v.xpt[(v.k + (v.j * v.xpt_dim1))] * v.d__[v.j]))
    v.j = v.j + (1)
    return 159
  end
  blocks[164] = function()
    v.temp = v.zero
    v.i__2 = v.n[0]
    v.j = 1
    return 159
  end
  blocks[166] = function()
    v.i__1 = v.npt[0]
    v.k = 1
    return 151
  end
  blocks[167] = function()
    if truth((v.j <= v.i__2)) then return 179 else return 166 end
  end
  blocks[168] = function()
    v.j = v.j + (1)
    return 167
  end
  blocks[169] = function()
    if truth((v.i__ <= v.i__1)) then return 174 else return 168 end
  end
  blocks[171] = function()
    v.gopt[v.i__] = (v.gopt[v.i__] + (v.hq[v.ih] * v.d__[v.j]))
    v.i__ = v.i__ + (1)
    return 169
  end
  blocks[172] = function()
    v.gopt[v.j] = (v.gopt[v.j] + (v.hq[v.ih] * v.d__[v.i__]))
    return 171
  end
  blocks[174] = function()
    v.ih = v.ih + (1)
    if truth((v.i__ < v.j)) then return 172 else return 171 end
  end
  blocks[179] = function()
    v.xopt[v.j] = v.xnew[v.j]
    v.d__1 = v.xopt[v.j]
    v.xoptsq = (v.xoptsq + (v.d__1 * v.d__1))
    v.i__1 = v.j
    v.i__ = 1
    return 169
  end
  blocks[184] = function()
    v.kopt = v.knew
    v.xoptsq = v.zero
    v.ih = 0
    v.i__2 = v.n[0]
    v.j = 1
    return 167
  end
  blocks[185] = function()
    if truth((v.f < v.fopt)) then return 184 else return 147 end
  end
  blocks[186] = function()
    if truth((v.i__ <= v.i__2)) then return 188 else return 185 end
  end
  blocks[188] = function()
    v.gopt[v.i__] = (v.gopt[v.i__] + (v.diff * v.w[v.i__]))
    v.i__ = v.i__ + (1)
    return 186
  end
  blocks[190] = function()
    v.i__2 = v.n[0]
    v.i__ = 1
    return 186
  end
  blocks[191] = function()
    if truth((v.k <= v.i__1)) then return 213 else return 190 end
  end
  blocks[192] = function()
    v.k = v.k + (1)
    return 191
  end
  blocks[193] = function()
    if truth((v.i__ <= v.i__2)) then return 195 else return 192 end
  end
  blocks[195] = function()
    v.w[v.i__] = (v.w[v.i__] + (v.temp * v.xpt[(v.k + (v.i__ * v.xpt_dim1))]))
    v.i__ = v.i__ + (1)
    return 193
  end
  blocks[198] = function()
    v.temp = (v.suma * v.sumb)
    v.i__2 = v.n[0]
    v.i__ = 1
    return 193
  end
  blocks[199] = function()
    if truth((v.j <= v.i__2)) then return 201 else return 198 end
  end
  blocks[201] = function()
    v.sumb = (v.sumb + (v.xpt[(v.k + (v.j * v.xpt_dim1))] * v.xopt[v.j]))
    v.j = v.j + (1)
    return 199
  end
  blocks[204] = function()
    v.sumb = v.zero
    v.i__2 = v.n[0]
    v.j = 1
    return 199
  end
  blocks[206] = function()
    v.rc = NLOPT_ROUNDOFF_LIMITED
    return 14
  end
  blocks[207] = function()
    if truth(nlopt_isinf(v.suma)) then return 206 else return 204 end
  end
  blocks[208] = function()
    if truth((v.jj <= v.i__2)) then return 210 else return 207 end
  end
  blocks[210] = function()
    v.suma = (v.suma + (v.zmat[(v.knew + (v.jj * v.zmat_dim1))] * v.zmat[(v.k + (v.jj * v.zmat_dim1))]))
    v.jj = v.jj + (1)
    return 208
  end
  blocks[213] = function()
    v.suma = v.zero
    v.i__2 = v.nptm
    v.jj = 1
    return 208
  end
  blocks[215] = function()
    v.i__1 = v.npt[0]
    v.k = 1
    return 191
  end
  blocks[216] = function()
    if truth((v.i__ <= v.i__1)) then return 219 else return 215 end
  end
  blocks[219] = function()
    v.xpt[(v.knew + (v.i__ * v.xpt_dim1))] = v.xnew[v.i__]
    v.w[v.i__] = v.bmat[(v.knew + (v.i__ * v.bmat_dim1))]
    v.i__ = v.i__ + (1)
    return 216
  end
  blocks[222] = function()
    v.fval[v.knew] = v.f
    v.i__1 = v.n[0]
    v.i__ = 1
    return 216
  end
  blocks[223] = function()
    if truth((v.jj <= v.i__2)) then return 230 else return 222 end
  end
  blocks[224] = function()
    v.jj = v.jj + (1)
    return 223
  end
  blocks[225] = function()
    if truth((v.k <= v.i__1)) then return 227 else return 224 end
  end
  blocks[227] = function()
    v.pq[v.k] = (v.pq[v.k] + (v.temp * v.zmat[(v.k + (v.jj * v.zmat_dim1))]))
    v.k = v.k + (1)
    return 225
  end
  blocks[230] = function()
    v.temp = (v.diff * v.zmat[(v.knew + (v.jj * v.zmat_dim1))])
    v.i__1 = v.npt[0]
    v.k = 1
    return 225
  end
  blocks[232] = function()
    v.i__2 = v.nptm
    v.jj = 1
    return 223
  end
  blocks[233] = function()
    if truth((v.i__ <= v.i__1)) then return 241 else return 232 end
  end
  blocks[234] = function()
    v.i__ = v.i__ + (1)
    return 233
  end
  blocks[235] = function()
    if truth((v.j <= v.i__2)) then return 238 else return 234 end
  end
  blocks[238] = function()
    v.ih = v.ih + (1)
    v.hq[v.ih] = (v.hq[v.ih] + (v.temp * v.xpt[(v.knew + (v.j * v.xpt_dim1))]))
    v.j = v.j + (1)
    return 235
  end
  blocks[241] = function()
    v.temp = (v.pqold * v.xpt[(v.knew + (v.i__ * v.xpt_dim1))])
    v.i__2 = v.i__
    v.j = 1
    return 235
  end
  blocks[247] = function()
    update_(v.n, v.npt, add(v.bmat,v.bmat_offset), add(v.zmat,v.zmat_offset), v.ndim, add(v.vlag,1), ptr(v,"beta"), ptr(v,"denom"), ptr(v,"knew"), add(v.w,1))
    v.ih = 0
    v.pqold = v.pq[v.knew]
    v.pq[v.knew] = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 233
  end
  blocks[249] = function()
    v.knew = v.ksav
    v.denom = v.densav
    return 247
  end
  blocks[250] = function()
    if truth((v.scaden <= (v.half * v.biglsq))) then return 249 else return 247 end
  end
  blocks[251] = function()
    if truth((v.k <= v.i__1)) then return 278 else return 250 end
  end
  blocks[255] = function()
    v.d__3 = v.vlag[v.k]
    v.d__1 = v.biglsq; v.d__2 = (v.temp * (v.d__3 * v.d__3))
    v.biglsq = math.max(v.d__1, v.d__2)
    v.k = v.k + (1)
    return 251
  end
  blocks[258] = function()
    v.scaden = (v.temp * v.den)
    v.knew = v.k
    v.denom = v.den
    return 255
  end
  blocks[262] = function()
    v.d__3 = (v.distsq / v.delsq)
    v.d__1 = v.one; v.d__2 = (v.d__3 * v.d__3)
    v.temp = math.max(v.d__1, v.d__2)
    if truth(((v.temp * v.den) > v.scaden)) then return 258 else return 255 end
  end
  blocks[263] = function()
    if truth((v.j <= v.i__2)) then return 266 else return 262 end
  end
  blocks[266] = function()
    v.d__1 = (v.xpt[(v.k + (v.j * v.xpt_dim1))] - v.xnew[v.j])
    v.distsq = (v.distsq + (v.d__1 * v.d__1))
    v.j = v.j + (1)
    return 263
  end
  blocks[271] = function()
    v.d__1 = v.vlag[v.k]
    v.den = ((v.beta * v.hdiag) + (v.d__1 * v.d__1))
    v.distsq = v.zero
    v.i__2 = v.n[0]
    v.j = 1
    return 263
  end
  blocks[272] = function()
    if truth((v.jj <= v.i__2)) then return 275 else return 271 end
  end
  blocks[275] = function()
    v.d__1 = v.zmat[(v.k + (v.jj * v.zmat_dim1))]
    v.hdiag = (v.hdiag + (v.d__1 * v.d__1))
    v.jj = v.jj + (1)
    return 272
  end
  blocks[278] = function()
    v.hdiag = v.zero
    v.i__2 = v.nptm
    v.jj = 1
    return 272
  end
  blocks[286] = function()
    v.ksav = v.knew
    v.densav = v.denom
    v.delsq = (v.delta * v.delta)
    v.scaden = v.zero
    v.biglsq = v.zero
    v.knew = 0
    v.i__1 = v.npt[0]
    v.k = 1
    return 251
  end
  blocks[287] = function()
    if truth((v.f < v.fopt)) then return 286 else return 247 end
  end
  blocks[288] = function()
    v.delta = v.rho
    return 287
  end
  blocks[289] = function()
    if truth((v.delta <= (v.rho * 1.5))) then return 288 else return 287 end
  end
  blocks[291] = function()
    v.d__1 = (v.half * v.delta)
    v.delta = math.min(v.d__1, v.dnorm)
    return 289
  end
  blocks[293] = function()
    v.d__1 = (v.half * v.delta)
    v.delta = math.max(v.d__1, v.dnorm)
    return 289
  end
  blocks[295] = function()
    v.d__1 = (v.half * v.delta); v.d__2 = (v.dnorm + v.dnorm)
    v.delta = math.max(v.d__1, v.d__2)
    return 289
  end
  blocks[296] = function()
    if truth((v.ratio <= .7)) then return 293 else return 295 end
  end
  blocks[298] = function()
    v.ratio = ((v.f - v.fopt) / v.vquad)
    if truth((v.ratio <= v.tenth)) then return 291 else return 296 end
  end
  blocks[300] = function()
    v.rc = NLOPT_ROUNDOFF_LIMITED
    return 14
  end
  blocks[301] = function()
    if truth((v.vquad >= v.zero)) then return 300 else return 298 end
  end
  blocks[302] = function()
    if truth((v.ntrits > 0)) then return 301 else return 247 end
  end
  blocks[303] = function()
    v.nfsav = v.stop.nevals_p[0]
    return 302
  end
  blocks[308] = function()
    v.diff = ((v.f - v.fopt) - v.vquad)
    v.diffc = v.diffb
    v.diffb = v.diffa
    v.diffa = math.abs(v.diff)
    if truth((v.dnorm > v.rho)) then return 303 else return 302 end
  end
  blocks[309] = function()
    if truth((v.k <= v.i__1)) then return 312 else return 308 end
  end
  blocks[312] = function()
    v.d__1 = v.w[(v.npt[0] + v.k)]
    v.vquad = (v.vquad + ((v.half * v.pq[v.k]) * (v.d__1 * v.d__1)))
    v.k = v.k + (1)
    return 309
  end
  blocks[314] = function()
    v.i__1 = v.npt[0]
    v.k = 1
    return 309
  end
  blocks[315] = function()
    if truth((v.j <= v.i__2)) then return 326 else return 314 end
  end
  blocks[316] = function()
    v.j = v.j + (1)
    return 315
  end
  blocks[317] = function()
    if truth((v.i__ <= v.i__1)) then return 323 else return 316 end
  end
  blocks[319] = function()
    v.vquad = (v.vquad + (v.hq[v.ih] * v.temp))
    v.i__ = v.i__ + (1)
    return 317
  end
  blocks[320] = function()
    v.temp = (v.half * v.temp)
    return 319
  end
  blocks[323] = function()
    v.ih = v.ih + (1)
    v.temp = (v.d__[v.i__] * v.d__[v.j])
    if truth((v.i__ == v.j)) then return 320 else return 319 end
  end
  blocks[326] = function()
    v.vquad = (v.vquad + (v.d__[v.j] * v.gopt[v.j]))
    v.i__1 = v.j
    v.i__ = 1
    return 317
  end
  blocks[331] = function()
    v.fopt = v.fval[v.kopt]
    v.vquad = v.zero
    v.ih = 0
    v.i__2 = v.n[0]
    v.j = 1
    return 315
  end
  blocks[333] = function()
    v.minf[0] = v.f
    result = NLOPT_MINF_MAX_REACHED; return nil
  end
  blocks[334] = function()
    if truth((v.f < v.stop.minf_max)) then return 333 else return 331 end
  end
  blocks[335] = function()
    return 14
  end
  blocks[337] = function()
    v.minf[0] = v.f
    result = v.rc; return nil
  end
  blocks[340] = function()
    v.fsave = v.f
    v.rc = NLOPT_XTOL_REACHED
    if truth((v.fsave < v.fval[v.kopt])) then return 337 else return 335 end
  end
  blocks[343] = function()
    v.stop.nevals_p[0] = v.stop.nevals_p[0] + (1)
    v.f = v.calfun(v.n[0], add(v.x,1), v.calfun_data)
    if truth((v.ntrits == (-1))) then return 340 else return 334 end
  end
  blocks[344] = function()
    return 14
  end
  blocks[345] = function()
    if truth((v.rc ~= NLOPT_SUCCESS)) then return 344 else return 343 end
  end
  blocks[346] = function()
    v.rc = NLOPT_FORCED_STOP
    return 345
  end
  blocks[347] = function()
    v.rc = NLOPT_MAXEVAL_REACHED
    return 345
  end
  blocks[348] = function()
    v.rc = NLOPT_MAXTIME_REACHED
    return 345
  end
  blocks[349] = function()
    if truth(nlopt_stop_time(v.stop)) then return 348 else return 345 end
  end
  blocks[350] = function()
    if truth(nlopt_stop_evals(v.stop)) then return 347 else return 349 end
  end
  blocks[351] = function()
    if truth(nlopt_stop_forced(v.stop)) then return 346 else return 350 end
  end
  blocks[352] = function()
    if truth((v.i__ <= v.i__2)) then return 360 else return 351 end
  end
  blocks[353] = function()
    v.i__ = v.i__ + (1)
    return 352
  end
  blocks[354] = function()
    v.x[v.i__] = v.xu[v.i__]
    return 353
  end
  blocks[355] = function()
    if truth((v.xnew[v.i__] == v.su[v.i__])) then return 354 else return 353 end
  end
  blocks[356] = function()
    v.x[v.i__] = v.xl[v.i__]
    return 355
  end
  blocks[360] = function()
    v.d__3 = v.xl[v.i__]; v.d__4 = (v.xbase[v.i__] + v.xnew[v.i__])
    v.d__1 = math.max(v.d__3, v.d__4); v.d__2 = v.xu[v.i__]
    v.x[v.i__] = math.min(v.d__1, v.d__2)
    if truth((v.xnew[v.i__] == v.sl[v.i__])) then return 356 else return 355 end
  end
  blocks[362] = function()
    v.i__2 = v.n[0]
    v.i__ = 1
    return 352
  end
  blocks[364] = function()
    v.rc = NLOPT_ROUNDOFF_LIMITED
    return 14
  end
  blocks[365] = function()
    return 510
  end
  blocks[366] = function()
    if truth((v.stop.nevals_p[0] > v.nresc)) then return 365 else return 364 end
  end
  blocks[368] = function()
    v.d__1 = v.vlag[v.knew]
    if truth((v.denom <= (v.half * (v.d__1 * v.d__1)))) then return 366 else return 362 end
  end
  blocks[370] = function()
    v.cauchy = v.zero
    return 483
  end
  blocks[371] = function()
    if truth((v.i__ <= v.i__2)) then return 374 else return 370 end
  end
  blocks[374] = function()
    v.xnew[v.i__] = v.xalt[v.i__]
    v.d__[v.i__] = (v.xnew[v.i__] - v.xopt[v.i__])
    v.i__ = v.i__ + (1)
    return 371
  end
  blocks[376] = function()
    v.i__2 = v.n[0]
    v.i__ = 1
    return 371
  end
  blocks[379] = function()
    v.d__1 = v.vlag[v.knew]
    v.denom = ((v.d__1 * v.d__1) + (v.alpha * v.beta))
    if truth((truth((v.denom < v.cauchy)) and truth((v.cauchy > v.zero)))) then return 376 else return 368 end
  end
  blocks[381] = function()
    v.rc = NLOPT_ROUNDOFF_LIMITED
    return 14
  end
  blocks[382] = function()
    return 510
  end
  blocks[383] = function()
    if truth((v.stop.nevals_p[0] > v.nresc)) then return 382 else return 381 end
  end
  blocks[384] = function()
    if truth((v.scaden <= (v.half * v.biglsq))) then return 383 else return 362 end
  end
  blocks[385] = function()
    if truth((v.k <= v.i__2)) then return 415 else return 384 end
  end
  blocks[387] = function()
    v.k = v.k + (1)
    return 385
  end
  blocks[390] = function()
    v.d__3 = v.vlag[v.k]
    v.d__1 = v.biglsq; v.d__2 = (v.temp * (v.d__3 * v.d__3))
    v.biglsq = math.max(v.d__1, v.d__2)
    return 387
  end
  blocks[393] = function()
    v.scaden = (v.temp * v.den)
    v.knew = v.k
    v.denom = v.den
    return 390
  end
  blocks[397] = function()
    v.d__3 = (v.distsq / v.delsq)
    v.d__1 = v.one; v.d__2 = (v.d__3 * v.d__3)
    v.temp = math.max(v.d__1, v.d__2)
    if truth(((v.temp * v.den) > v.scaden)) then return 393 else return 390 end
  end
  blocks[398] = function()
    if truth((v.j <= v.i__1)) then return 401 else return 397 end
  end
  blocks[401] = function()
    v.d__1 = (v.xpt[(v.k + (v.j * v.xpt_dim1))] - v.xopt[v.j])
    v.distsq = (v.distsq + (v.d__1 * v.d__1))
    v.j = v.j + (1)
    return 398
  end
  blocks[406] = function()
    v.d__1 = v.vlag[v.k]
    v.den = ((v.beta * v.hdiag) + (v.d__1 * v.d__1))
    v.distsq = v.zero
    v.i__1 = v.n[0]
    v.j = 1
    return 398
  end
  blocks[407] = function()
    if truth((v.jj <= v.i__1)) then return 410 else return 406 end
  end
  blocks[410] = function()
    v.d__1 = v.zmat[(v.k + (v.jj * v.zmat_dim1))]
    v.hdiag = (v.hdiag + (v.d__1 * v.d__1))
    v.jj = v.jj + (1)
    return 407
  end
  blocks[413] = function()
    v.hdiag = v.zero
    v.i__1 = v.nptm
    v.jj = 1
    return 407
  end
  blocks[414] = function()
    return 387
  end
  blocks[415] = function()
    if truth((v.k == v.kopt)) then return 414 else return 413 end
  end
  blocks[421] = function()
    v.delsq = (v.delta * v.delta)
    v.scaden = v.zero
    v.biglsq = v.zero
    v.knew = 0
    v.i__2 = v.npt[0]
    v.k = 1
    return 385
  end
  blocks[424] = function()
    v.beta = ((((v.dx * v.dx) + (v.dsq * (((v.xoptsq + v.dx) + v.dx) + (v.half * v.dsq)))) + v.beta) - v.bsum)
    v.vlag[v.kopt] = (v.vlag[v.kopt] + v.one)
    if truth((v.ntrits == 0)) then return 379 else return 421 end
  end
  blocks[425] = function()
    if truth((v.j <= v.i__2)) then return 444 else return 424 end
  end
  blocks[429] = function()
    v.vlag[v.jp] = v.sum
    v.bsum = (v.bsum + (v.sum * v.d__[v.j]))
    v.dx = (v.dx + (v.d__[v.j] * v.xopt[v.j]))
    v.j = v.j + (1)
    return 425
  end
  blocks[430] = function()
    if truth((v.i__ <= v.i__1)) then return 432 else return 429 end
  end
  blocks[432] = function()
    v.sum = (v.sum + (v.bmat[(v.jp + (v.i__ * v.bmat_dim1))] * v.d__[v.i__]))
    v.i__ = v.i__ + (1)
    return 430
  end
  blocks[436] = function()
    v.bsum = (v.bsum + (v.sum * v.d__[v.j]))
    v.jp = (v.npt[0] + v.j)
    v.i__1 = v.n[0]
    v.i__ = 1
    return 430
  end
  blocks[437] = function()
    if truth((v.k <= v.i__1)) then return 439 else return 436 end
  end
  blocks[439] = function()
    v.sum = (v.sum + (v.w[v.k] * v.bmat[(v.k + (v.j * v.bmat_dim1))]))
    v.k = v.k + (1)
    return 437
  end
  blocks[444] = function()
    v.d__1 = v.d__[v.j]
    v.dsq = (v.dsq + (v.d__1 * v.d__1))
    v.sum = v.zero
    v.i__1 = v.npt[0]
    v.k = 1
    return 437
  end
  blocks[449] = function()
    v.dsq = v.zero
    v.bsum = v.zero
    v.dx = v.zero
    v.i__2 = v.n[0]
    v.j = 1
    return 425
  end
  blocks[450] = function()
    if truth((v.jj <= v.i__1)) then return 463 else return 449 end
  end
  blocks[451] = function()
    v.jj = v.jj + (1)
    return 450
  end
  blocks[452] = function()
    if truth((v.k <= v.i__2)) then return 454 else return 451 end
  end
  blocks[454] = function()
    v.vlag[v.k] = (v.vlag[v.k] + (v.sum * v.zmat[(v.k + (v.jj * v.zmat_dim1))]))
    v.k = v.k + (1)
    return 452
  end
  blocks[457] = function()
    v.beta = (v.beta - (v.sum * v.sum))
    v.i__2 = v.npt[0]
    v.k = 1
    return 452
  end
  blocks[458] = function()
    if truth((v.k <= v.i__2)) then return 460 else return 457 end
  end
  blocks[460] = function()
    v.sum = (v.sum + (v.zmat[(v.k + (v.jj * v.zmat_dim1))] * v.w[v.k]))
    v.k = v.k + (1)
    return 458
  end
  blocks[463] = function()
    v.sum = v.zero
    v.i__2 = v.npt[0]
    v.k = 1
    return 458
  end
  blocks[466] = function()
    v.beta = v.zero
    v.i__1 = v.nptm
    v.jj = 1
    return 450
  end
  blocks[467] = function()
    if truth((v.k <= v.i__1)) then return 481 else return 466 end
  end
  blocks[471] = function()
    v.w[v.k] = (v.suma * ((v.half * v.suma) + v.sumb))
    v.vlag[v.k] = v.sum
    v.w[(v.npt[0] + v.k)] = v.suma
    v.k = v.k + (1)
    return 467
  end
  blocks[472] = function()
    if truth((v.j <= v.i__2)) then return 476 else return 471 end
  end
  blocks[476] = function()
    v.suma = (v.suma + (v.xpt[(v.k + (v.j * v.xpt_dim1))] * v.d__[v.j]))
    v.sumb = (v.sumb + (v.xpt[(v.k + (v.j * v.xpt_dim1))] * v.xopt[v.j]))
    v.sum = (v.sum + (v.bmat[(v.k + (v.j * v.bmat_dim1))] * v.d__[v.j]))
    v.j = v.j + (1)
    return 472
  end
  blocks[481] = function()
    v.suma = v.zero
    v.sumb = v.zero
    v.sum = v.zero
    v.i__2 = v.n[0]
    v.j = 1
    return 472
  end
  blocks[483] = function()
    v.i__1 = v.npt[0]
    v.k = 1
    return 467
  end
  blocks[484] = function()
    if truth((v.i__ <= v.i__1)) then return 486 else return 483 end
  end
  blocks[486] = function()
    v.d__[v.i__] = (v.xnew[v.i__] - v.xopt[v.i__])
    v.i__ = v.i__ + (1)
    return 484
  end
  blocks[489] = function()
    altmov_(v.n, v.npt, add(v.xpt,v.xpt_offset), add(v.xopt,1), add(v.bmat,v.bmat_offset), add(v.zmat,v.zmat_offset), v.ndim, add(v.sl,1), add(v.su,1), ptr(v,"kopt"), ptr(v,"knew"), ptr(v,"adelt"), add(v.xnew,1), add(v.xalt,1), ptr(v,"alpha"), ptr(v,"cauchy"), add(v.w,1), add(v.w,v.np), add(v.w,(v.ndim[0] + 1)))
    v.i__1 = v.n[0]
    v.i__ = 1
    return 484
  end
  blocks[490] = function()
    return 647
  end
  blocks[491] = function()
    if truth((v.ntrits > 0)) then return 490 else return 489 end
  end
  blocks[493] = function()
    v.nfsav = v.stop.nevals_p[0]
    return 678
  end
  blocks[495] = function()
    v.nresc = v.stop.nevals_p[0]
    if truth((v.nfsav < v.stop.nevals_p[0])) then return 493 else return 491 end
  end
  blocks[497] = function()
    v.rc = v.rc2
    return 14
  end
  blocks[498] = function()
    if truth((v.rc2 ~= NLOPT_SUCCESS)) then return 497 else return 495 end
  end
  blocks[499] = function()
    if truth((v.i__ <= v.i__1)) then return 503 else return 498 end
  end
  blocks[503] = function()
    v.xopt[v.i__] = v.xpt[(v.kopt + (v.i__ * v.xpt_dim1))]
    v.d__1 = v.xopt[v.i__]
    v.xoptsq = (v.xoptsq + (v.d__1 * v.d__1))
    v.i__ = v.i__ + (1)
    return 499
  end
  blocks[505] = function()
    v.i__1 = v.n[0]
    v.i__ = 1
    return 499
  end
  blocks[510] = function()
    v.nfsav = v.stop.nevals_p[0]
    v.kbase = v.kopt
    v.rc2 = rescue_(v.n, v.npt, add(v.xl,1), add(v.xu,1), v.stop, v.calfun, v.calfun_data, add(v.xbase,1), add(v.xpt,v.xpt_offset), add(v.fval,1), add(v.xopt,1), add(v.gopt,1), add(v.hq,1), add(v.pq,1), add(v.bmat,v.bmat_offset), add(v.zmat,v.zmat_offset), v.ndim, add(v.sl,1), add(v.su,1), ptr(v,"delta"), ptr(v,"kopt"), add(v.vlag,1), add(v.w,1), add(v.w,(v.n[0] + v.np)), add(v.w,(v.ndim[0] + v.np)))
    v.xoptsq = v.zero
    if truth((v.kopt ~= v.kbase)) then return 505 else return 498 end
  end
  blocks[511] = function()
    return 483
  end
  blocks[512] = function()
    return 489
  end
  blocks[513] = function()
    if truth((v.ntrits == 0)) then return 512 else return 511 end
  end
  blocks[514] = function()
    v.xoptsq = v.zero
    return 513
  end
  blocks[515] = function()
    if truth((v.i__ <= v.i__1)) then return 521 else return 514 end
  end
  blocks[521] = function()
    v.xbase[v.i__] = (v.xbase[v.i__] + v.xopt[v.i__])
    v.xnew[v.i__] = (v.xnew[v.i__] - v.xopt[v.i__])
    v.sl[v.i__] = (v.sl[v.i__] - v.xopt[v.i__])
    v.su[v.i__] = (v.su[v.i__] - v.xopt[v.i__])
    v.xopt[v.i__] = v.zero
    v.i__ = v.i__ + (1)
    return 515
  end
  blocks[523] = function()
    v.i__1 = v.n[0]
    v.i__ = 1
    return 515
  end
  blocks[524] = function()
    if truth((v.j <= v.i__2)) then return 539 else return 523 end
  end
  blocks[525] = function()
    v.j = v.j + (1)
    return 524
  end
  blocks[526] = function()
    if truth((v.i__ <= v.i__1)) then return 530 else return 525 end
  end
  blocks[530] = function()
    v.ih = v.ih + (1)
    v.hq[v.ih] = ((v.hq[v.ih] + (v.w[v.i__] * v.xopt[v.j])) + (v.xopt[v.i__] * v.w[v.j]))
    v.bmat[((v.npt[0] + v.i__) + (v.j * v.bmat_dim1))] = v.bmat[((v.npt[0] + v.j) + (v.i__ * v.bmat_dim1))]
    v.i__ = v.i__ + (1)
    return 526
  end
  blocks[532] = function()
    v.i__1 = v.j
    v.i__ = 1
    return 526
  end
  blocks[533] = function()
    if truth((v.k <= v.i__1)) then return 536 else return 532 end
  end
  blocks[536] = function()
    v.w[v.j] = (v.w[v.j] + (v.pq[v.k] * v.xpt[(v.k + (v.j * v.xpt_dim1))]))
    v.xpt[(v.k + (v.j * v.xpt_dim1))] = (v.xpt[(v.k + (v.j * v.xpt_dim1))] - v.xopt[v.j])
    v.k = v.k + (1)
    return 533
  end
  blocks[539] = function()
    v.w[v.j] = (((-v.half) * v.sumpq) * v.xopt[v.j])
    v.i__1 = v.npt[0]
    v.k = 1
    return 533
  end
  blocks[542] = function()
    v.ih = 0
    v.i__2 = v.n[0]
    v.j = 1
    return 524
  end
  blocks[543] = function()
    if truth((v.jj <= v.i__3)) then return 580 else return 542 end
  end
  blocks[544] = function()
    v.jj = v.jj + (1)
    return 543
  end
  blocks[545] = function()
    if truth((v.i__ <= v.i__1)) then return 553 else return 544 end
  end
  blocks[546] = function()
    v.i__ = v.i__ + (1)
    return 545
  end
  blocks[547] = function()
    if truth((v.j <= v.i__2)) then return 549 else return 546 end
  end
  blocks[549] = function()
    v.bmat[(v.ip + (v.j * v.bmat_dim1))] = (v.bmat[(v.ip + (v.j * v.bmat_dim1))] + (v.temp * v.w[v.j]))
    v.j = v.j + (1)
    return 547
  end
  blocks[553] = function()
    v.ip = (v.i__ + v.npt[0])
    v.temp = v.w[v.i__]
    v.i__2 = v.i__
    v.j = 1
    return 547
  end
  blocks[555] = function()
    v.i__1 = v.n[0]
    v.i__ = 1
    return 545
  end
  blocks[556] = function()
    if truth((v.j <= v.i__2)) then return 569 else return 555 end
  end
  blocks[557] = function()
    v.j = v.j + (1)
    return 556
  end
  blocks[558] = function()
    if truth((v.k <= v.i__1)) then return 560 else return 557 end
  end
  blocks[560] = function()
    v.bmat[(v.k + (v.j * v.bmat_dim1))] = (v.bmat[(v.k + (v.j * v.bmat_dim1))] + (v.sum * v.zmat[(v.k + (v.jj * v.zmat_dim1))]))
    v.k = v.k + (1)
    return 558
  end
  blocks[563] = function()
    v.w[v.j] = v.sum
    v.i__1 = v.npt[0]
    v.k = 1
    return 558
  end
  blocks[564] = function()
    if truth((v.k <= v.i__1)) then return 566 else return 563 end
  end
  blocks[566] = function()
    v.sum = (v.sum + (v.vlag[v.k] * v.xpt[(v.k + (v.j * v.xpt_dim1))]))
    v.k = v.k + (1)
    return 564
  end
  blocks[569] = function()
    v.sum = (((v.fracsq * v.sumz) - (v.half * v.sumw)) * v.xopt[v.j])
    v.i__1 = v.npt[0]
    v.k = 1
    return 564
  end
  blocks[571] = function()
    v.i__2 = v.n[0]
    v.j = 1
    return 556
  end
  blocks[572] = function()
    if truth((v.k <= v.i__2)) then return 576 else return 571 end
  end
  blocks[576] = function()
    v.sumz = (v.sumz + v.zmat[(v.k + (v.jj * v.zmat_dim1))])
    v.vlag[v.k] = (v.w[(v.npt[0] + v.k)] * v.zmat[(v.k + (v.jj * v.zmat_dim1))])
    v.sumw = (v.sumw + v.vlag[v.k])
    v.k = v.k + (1)
    return 572
  end
  blocks[580] = function()
    v.sumz = v.zero
    v.sumw = v.zero
    v.i__2 = v.npt[0]
    v.k = 1
    return 572
  end
  blocks[582] = function()
    v.i__3 = v.nptm
    v.jj = 1
    return 543
  end
  blocks[583] = function()
    if truth((v.k <= v.i__1)) then return 605 else return 582 end
  end
  blocks[584] = function()
    v.k = v.k + (1)
    return 583
  end
  blocks[585] = function()
    if truth((v.i__ <= v.i__2)) then return 594 else return 584 end
  end
  blocks[586] = function()
    v.i__ = v.i__ + (1)
    return 585
  end
  blocks[587] = function()
    if truth((v.j <= v.i__3)) then return 589 else return 586 end
  end
  blocks[589] = function()
    v.bmat[(v.ip + (v.j * v.bmat_dim1))] = ((v.bmat[(v.ip + (v.j * v.bmat_dim1))] + (v.w[v.i__] * v.vlag[v.j])) + (v.vlag[v.i__] * v.w[v.j]))
    v.j = v.j + (1)
    return 587
  end
  blocks[594] = function()
    v.w[v.i__] = v.bmat[(v.k + (v.i__ * v.bmat_dim1))]
    v.vlag[v.i__] = ((v.sum * v.xpt[(v.k + (v.i__ * v.xpt_dim1))]) + (v.temp * v.xopt[v.i__]))
    v.ip = (v.npt[0] + v.i__)
    v.i__3 = v.i__
    v.j = 1
    return 587
  end
  blocks[598] = function()
    v.w[(v.npt[0] + v.k)] = v.sum
    v.temp = (v.fracsq - (v.half * v.sum))
    v.i__2 = v.n[0]
    v.i__ = 1
    return 585
  end
  blocks[599] = function()
    if truth((v.i__ <= v.i__2)) then return 601 else return 598 end
  end
  blocks[601] = function()
    v.sum = (v.sum + (v.xpt[(v.k + (v.i__ * v.xpt_dim1))] * v.xopt[v.i__]))
    v.i__ = v.i__ + (1)
    return 599
  end
  blocks[605] = function()
    v.sumpq = (v.sumpq + v.pq[v.k])
    v.sum = ((-v.half) * v.xoptsq)
    v.i__2 = v.n[0]
    v.i__ = 1
    return 599
  end
  blocks[609] = function()
    v.fracsq = (v.xoptsq * .25)
    v.sumpq = v.zero
    v.i__1 = v.npt[0]
    v.k = 1
    return 583
  end
  blocks[610] = function()
    if truth((v.dsq <= (v.xoptsq * .001))) then return 609 else return 513 end
  end
  blocks[611] = function()
    v.ntrits = v.ntrits + (1)
    return 610
  end
  blocks[612] = function()
    return 28
  end
  blocks[613] = function()
    if truth((v.j <= v.i__1)) then return 630 else return 612 end
  end
  blocks[614] = function()
    v.j = v.j + (1)
    return 613
  end
  blocks[615] = function()
    return 62
  end
  blocks[617] = function()
    v.bdtest = (v.bdtest + ((v.half * v.curv) * v.rho))
    if truth((v.bdtest < v.bdtol)) then return 615 else return 614 end
  end
  blocks[618] = function()
    if truth((v.k <= v.i__2)) then return 621 else return 617 end
  end
  blocks[621] = function()
    v.d__1 = v.xpt[(v.k + (v.j * v.xpt_dim1))]
    v.curv = (v.curv + (v.pq[v.k] * (v.d__1 * v.d__1)))
    v.k = v.k + (1)
    return 618
  end
  blocks[624] = function()
    v.curv = v.hq[trunc(((v.j + (v.j * v.j)))/(2))]
    v.i__2 = v.npt[0]
    v.k = 1
    return 618
  end
  blocks[625] = function()
    if truth((v.bdtest < v.bdtol)) then return 624 else return 614 end
  end
  blocks[626] = function()
    v.bdtest = (-v.w[v.j])
    return 625
  end
  blocks[627] = function()
    if truth((v.xnew[v.j] == v.su[v.j])) then return 626 else return 625 end
  end
  blocks[628] = function()
    v.bdtest = v.w[v.j]
    return 627
  end
  blocks[630] = function()
    v.bdtest = v.bdtol
    if truth((v.xnew[v.j] == v.sl[v.j])) then return 628 else return 627 end
  end
  blocks[633] = function()
    v.bdtol = (v.errbig / v.rho)
    v.i__1 = v.n[0]
    v.j = 1
    return 613
  end
  blocks[634] = function()
    return 62
  end
  blocks[638] = function()
    v.d__1 = math.max(v.diffa, v.diffb)
    v.errbig = math.max(v.d__1, v.diffc)
    v.frhosq = ((v.rho * .125) * v.rho)
    if truth((truth((v.crvmin > v.zero)) and truth((v.errbig > (v.frhosq * v.crvmin))))) then return 634 else return 633 end
  end
  blocks[639] = function()
    return 62
  end
  blocks[643] = function()
    v.ntrits = (-1)
    v.d__1 = (v.ten * v.rho)
    v.distsq = (v.d__1 * v.d__1)
    if truth((v.stop.nevals_p[0] <= (v.nfsav + 2))) then return 639 else return 638 end
  end
  blocks[647] = function()
    trsbox_(v.n, v.npt, add(v.xpt,v.xpt_offset), add(v.xopt,1), add(v.gopt,1), add(v.hq,1), add(v.pq,1), add(v.sl,1), add(v.su,1), ptr(v,"delta"), add(v.xnew,1), add(v.d__,1), add(v.w,1), add(v.w,v.np), add(v.w,(v.np + v.n[0])), add(v.w,(v.np + (v.n[0] * 2^(1)))), add(v.w,(v.np + (v.n[0] * 3))), ptr(v,"dsq"), ptr(v,"crvmin"))
    v.d__1 = v.delta; v.d__2 = math.sqrt(v.dsq)
    v.dnorm = math.min(v.d__1, v.d__2)
    if truth((v.dnorm < (v.half * v.rho))) then return 643 else return 611 end
  end
  blocks[648] = function()
    if truth((v.k <= v.i__2)) then return 661 else return 647 end
  end
  blocks[649] = function()
    v.k = v.k + (1)
    return 648
  end
  blocks[650] = function()
    if truth((v.i__ <= v.i__1)) then return 652 else return 649 end
  end
  blocks[652] = function()
    v.gopt[v.i__] = (v.gopt[v.i__] + (v.temp * v.xpt[(v.k + (v.i__ * v.xpt_dim1))]))
    v.i__ = v.i__ + (1)
    return 650
  end
  blocks[655] = function()
    v.temp = (v.pq[v.k] * v.temp)
    v.i__1 = v.n[0]
    v.i__ = 1
    return 650
  end
  blocks[656] = function()
    if truth((v.j <= v.i__1)) then return 658 else return 655 end
  end
  blocks[658] = function()
    v.temp = (v.temp + (v.xpt[(v.k + (v.j * v.xpt_dim1))] * v.xopt[v.j]))
    v.j = v.j + (1)
    return 656
  end
  blocks[661] = function()
    v.temp = v.zero
    v.i__1 = v.n[0]
    v.j = 1
    return 656
  end
  blocks[663] = function()
    v.i__2 = v.npt[0]
    v.k = 1
    return 648
  end
  blocks[664] = function()
    if truth((v.stop.nevals_p[0] > v.npt[0])) then return 663 else return 647 end
  end
  blocks[665] = function()
    if truth((v.j <= v.i__1)) then return 674 else return 664 end
  end
  blocks[666] = function()
    v.j = v.j + (1)
    return 665
  end
  blocks[667] = function()
    if truth((v.i__ <= v.i__2)) then return 672 else return 666 end
  end
  blocks[669] = function()
    v.gopt[v.i__] = (v.gopt[v.i__] + (v.hq[v.ih] * v.xopt[v.j]))
    v.i__ = v.i__ + (1)
    return 667
  end
  blocks[670] = function()
    v.gopt[v.j] = (v.gopt[v.j] + (v.hq[v.ih] * v.xopt[v.i__]))
    return 669
  end
  blocks[672] = function()
    v.ih = v.ih + (1)
    if truth((v.i__ < v.j)) then return 670 else return 669 end
  end
  blocks[674] = function()
    v.i__2 = v.j
    v.i__ = 1
    return 667
  end
  blocks[677] = function()
    v.ih = 0
    v.i__1 = v.n[0]
    v.j = 1
    return 665
  end
  blocks[678] = function()
    if truth((v.kopt ~= v.kbase)) then return 677 else return 647 end
  end
  blocks[687] = function()
    v.kbase = 1
    v.rho = v.rhobeg[0]
    v.delta = v.rho
    v.nresc = v.stop.nevals_p[0]
    v.ntrits = 0
    v.diffa = v.zero
    v.diffb = v.zero
    v.itest = 0
    v.nfsav = v.stop.nevals_p[0]
    return 678
  end
  blocks[689] = function()
    v.rc = v.rc2
    return 14
  end
  blocks[691] = function()
    v.fsave = v.fval[1]
    if truth((v.rc2 ~= NLOPT_SUCCESS)) then return 689 else return 687 end
  end
  blocks[692] = function()
    if truth((v.i__ <= v.i__1)) then return 696 else return 691 end
  end
  blocks[696] = function()
    v.xopt[v.i__] = v.xpt[(v.kopt + (v.i__ * v.xpt_dim1))]
    v.d__1 = v.xopt[v.i__]
    v.xoptsq = (v.xoptsq + (v.d__1 * v.d__1))
    v.i__ = v.i__ + (1)
    return 692
  end
  blocks[819] = function()
    v.diffc = 0.0
    v.alpha = 0.0
    v.adelt = 0.0
    v.ratio = 0.0
    v.cauchy = 0.0
    v.rc = NLOPT_SUCCESS
    v.zmat_dim1 = v.npt[0]
    v.zmat_offset = (1 + v.zmat_dim1)
    v.zmat = add(v.zmat,(-v.zmat_offset))
    v.xpt_dim1 = v.npt[0]
    v.xpt_offset = (1 + v.xpt_dim1)
    v.xpt = add(v.xpt,(-v.xpt_offset))
    v.x = add(v.x,-1)
    v.xl = add(v.xl,-1)
    v.xu = add(v.xu,-1)
    v.xbase = add(v.xbase,-1)
    v.fval = add(v.fval,-1)
    v.xopt = add(v.xopt,-1)
    v.gopt = add(v.gopt,-1)
    v.hq = add(v.hq,-1)
    v.pq = add(v.pq,-1)
    v.bmat_dim1 = v.ndim[0]
    v.bmat_offset = (1 + v.bmat_dim1)
    v.bmat = add(v.bmat,(-v.bmat_offset))
    v.sl = add(v.sl,-1)
    v.su = add(v.su,-1)
    v.xnew = add(v.xnew,-1)
    v.xalt = add(v.xalt,-1)
    v.d__ = add(v.d__,-1)
    v.vlag = add(v.vlag,-1)
    v.w = add(v.w,-1)
    v.half = .5
    v.one = 1.
    v.ten = 10.
    v.tenth = .1
    v.two = 2.
    v.zero = 0.
    v.np = (v.n[0] + 1)
    v.nptm = (v.npt[0] - v.np)
    v.nh = trunc(((v.n[0] * v.np))/(2))
    v.rc2 = prelim_(v.n, v.npt, add(v.x,1), add(v.xl,1), add(v.xu,1), v.rhobeg, v.stop, v.calfun, v.calfun_data, add(v.xbase,1), add(v.xpt,v.xpt_offset), add(v.fval,1), add(v.gopt,1), add(v.hq,1), add(v.pq,1), add(v.bmat,v.bmat_offset), add(v.zmat,v.zmat_offset), v.ndim, add(v.sl,1), add(v.su,1), ptr(v,"kopt"))
    v.xoptsq = v.zero
    v.i__1 = v.n[0]
    v.i__ = 1
    return 692
  end
  local pc = 819
  while pc do pc = blocks[pc]() end
  return result, v.rho
end

-- Array partition and initial bound adjustments follow bobyqa() in the source.
-- _solve is exposed for differential verification of the numerical core.
-- Public wrapper vectors are one-based; the preserved core uses zero-based
-- pointer views and aliased work arrays. Do not rewrite those indexing conventions.
function M._solve(n, npt, initial, lower, upper, rhobeg, rhoend, maxeval, objective, stopval)
    assert(n >= 2 and n == math.floor(n), 'BOBYQA needs at least two active variables')
    assert(
        npt >= n + 2 and npt <= (n + 1) * (n + 2) / 2 and npt == math.floor(npt),
        'invalid BOBYQA npt'
    )
    assert(rhobeg > 0 and rhoend > 0 and rhoend <= rhobeg, 'invalid BOBYQA radii')
    assert(
        maxeval >= 1 and maxeval == math.floor(maxeval),
        'BOBYQA maxeval must be a positive integer'
    )
    local v = { n = n, npt = npt, rhobeg = rhobeg, rhoend = rhoend, ndim = n + npt }
    local x, xl, xu = array(n), array(n), array(n)
    for j = 0, n - 1 do
        assert(
            U.finite(lower[j + 1]) and U.finite(upper[j + 1]),
            'BOBYQA requires finite lower and upper bounds on every active variable'
        )
        assert(U.finite(initial[j + 1]), 'BOBYQA initial coordinates must be finite')
        x[j], xl[j], xu[j] = initial[j + 1], lower[j + 1], upper[j + 1]
    end
    local np, ndim = n + 1, n + npt
    local ixb = 1
    local ixp = ixb + n
    local ifv = ixp + n * npt
    local ixo = ifv + npt
    local igo = ixo + n
    local ihq = igo + n
    local ipq = ihq + n * np / 2
    local ibmat = ipq + npt
    local izmat = ibmat + ndim * n
    local isl = izmat + npt * (npt - np)
    local isu = isl + n
    local ixn = isu + n
    local ixa = ixn + n
    local id = ixa + n
    local ivl = id + n
    local iw = ivl + ndim
    local w = add(array((npt + 5) * (npt + n) + 3 * n * (n + 5) / 2), -1)
    for j = 1, n do
        local width = xu[j - 1] - xl[j - 1]
        assert(width >= 2 * rhobeg, 'BOBYQA bounds narrower than twice rhobeg')
        local jsl, jsu = isl + j - 1, isu + j - 1
        w[jsl], w[jsu] = xl[j - 1] - x[j - 1], xu[j - 1] - x[j - 1]
        if w[jsl] >= -rhobeg then
            if w[jsl] >= 0 then
                x[j - 1], w[jsl], w[jsu] = xl[j - 1], 0, width
            else
                x[j - 1] = xl[j - 1] + rhobeg;
                w[jsl] = -rhobeg;
                w[jsu] = math.max(xu[j - 1] - x[j - 1], rhobeg)
            end
        elseif w[jsu] <= rhobeg then
            if w[jsu] <= 0 then
                x[j - 1], w[jsl], w[jsu] = xu[j - 1], -width, 0
            else
                x[j - 1] = xu[j - 1] - rhobeg;
                w[jsl] = math.min(xl[j - 1] - x[j - 1], -rhobeg);
                w[jsu] = rhobeg
            end
        end
    end
    local stop = { nevals_p = array(1), maxeval = maxeval, minf_max = stopval or -math.huge }
    local minimum = array(1)

    local function calfun(count, point)
        local values = {}
        for j = 1, count do
            values[j] = point[j - 1]
        end
        return objective(values)
    end
    local status, radius = bobyqb_(
        ptr(v, 'n'),
        ptr(v, 'npt'),
        x,
        xl,
        xu,
        ptr(v, 'rhobeg'),
        ptr(v, 'rhoend'),
        stop,
        calfun,
        nil,
        minimum,
        add(w, ixb),
        add(w, ixp),
        add(w, ifv),
        add(w, ixo),
        add(w, igo),
        add(w, ihq),
        add(w, ipq),
        add(w, ibmat),
        add(w, izmat),
        ptr(v, 'ndim'),
        add(w, isl),
        add(w, isu),
        add(w, ixn),
        add(w, ixa),
        add(w, id),
        add(w, ivl),
        add(w, iw)
    )
    local result = {}
    for j = 1, n do
        result[j] = x[j - 1]
    end
    return result, minimum[0], status, stop.nevals_p[0], radius or rhobeg
end

-- Optimize active coordinates in abs(step) units, with finite bounds and costs.
-- rhobeg is reduced to fit the bounds; rhoend and the reported radius use the
-- wrapper's existing scaling. Core return codes map to the statuses below.
function M.run(self)
    local o = U.options(self, { rhobeg = 1 })
    local d = U.domain(self, o)
    local n = #d.active
    if n == 0 then
        d.evaluate(d.zero, { phase = 'initial', iteration = 0 })
        self._radius = 0;
        self._status = 'fixed_variables';
        return
    end
    assert(
        n >= 2,
        'BOBYQA requires at least two active variables; use Brent_Minimize for one variable'
    )
    U.number(o, 'rhobeg', 1e-300, math.huge)
    local rhobeg = o.rhobeg
    local lower, upper, start = {}, {}, {}
    for k, j in ipairs(d.active) do
        lower[k], upper[k], start[k] = d.lower[j], d.upper[j], 0
        assert(
            U.finite(lower[k]) and U.finite(upper[k]),
            'BOBYQA requires finite lower and upper bounds on every active variable'
        )
        rhobeg = math.min(rhobeg, 0.5 * (upper[k] - lower[k]))
    end
    local rhoend = o.rhoend or self._minradius / d.max_step
    assert(
        U.finite(rhoend) and rhoend > 0 and rhoend <= rhobeg,
        'BOBYQA rhoend must be positive and <= effective rhobeg'
    )
    local npt = o.npt or 2 * n + 1
    assert(
        U.finite(npt) and npt == math.floor(npt) and npt >= n + 2 and npt <= (n + 1) * (n + 2) / 2,
        'invalid BOBYQA npt'
    )
    o.rhobeg, o.rhoend, o.npt = rhobeg, rhoend, npt
    if o.stopval ~= nil then
        assert(U.finite(o.stopval), 'BOBYQA stopval must be finite')
    end
    local remaining = (self._maxcalls or 1000 * n) - self:ncalls()
    if remaining <= 0 then
        return
    end
    local _, _, status, _, radius = M._solve(
        n,
        npt,
        start,
        lower,
        upper,
        rhobeg,
        rhoend,
        remaining,
        function(point)
            local z = U.copy(d.zero)
            for k, j in ipairs(d.active) do
                z[j] = point[k]
            end
            local value = d.evaluate(z, { phase = 'bobyqa', iteration = self:ncalls() })
            -- Interpolation requires finite values. Fail explicitly rather than fit
            -- an arbitrary penalty that changes the objective and destroys models.
            assert(
                U.finite(value),
                'BOBYQA requires finite objective values; provide an explicit finite physical penalty'
            )
            return value
        end,
        o.stopval
    )
    self._bobyqa_status = status
    self._status = ({ [4] = 'radius_tolerance', [5] = 'budget', [2] = 'target', [-4] = 'roundoff' })[status]
        or 'stopped'
    self._radius = radius * d.max_step
end
return M
