# OptSi reference

Start with [README.md](README.md). This optional reference explains the public
constructor, options, model integration and source credits.

Source comments may name the former
`ALGORITHM_DETAILS_AND_VALIDATION.md`, `Docs/Reference.md`,
`SCIENTIFIC_REFERENCES.md` or an example `README.md`. Their documentation,
scientific references and source notices are consolidated
in this reference; CPOpt instructions are under [Advanced CPOpt](#advanced-cpopt).
These comment-only historical paths are not runtime dependencies.

## Direct multiobjective names

`algorithm='NSGA-II'` selects the existing GA/vector engine and
`algorithm='MOPSO'` selects the existing PSO/vector engine. Both infer
`mode='multiobjective'` when omitted and accept that explicit mode.
Both reject `mode='scalar'` with a clear error. These are names for existing
implementations, not two additional numerical methods. Legacy
`algorithm='GA', mode='multiobjective'` and `algorithm='PSO', mode='multiobjective'`
remain equivalent, including fixed-seed requests and archives. Scalar GA/PSO
behavior is unchanged. ParEGO also infers vector mode when omitted, accepts
explicit vector mode and rejects an explicitly conflicting scalar mode.
Internally the dispatcher retains the
canonical GA/PSO engine names; the caller's public spelling does not select a
different random stream or set of options.

## 1. Minimize a scalar objective

The objective returns one number to **minimize**. Convert a quantity to maximize
into a cost yourself, for example by returning its negative. Typical use is the following:

```lua
local Optimizer = require 'Optimization_Algorithms'

local opt = Optimizer{
    algorithm = 'BFGS',
    start = {2, -1},
    step = {1, 1},
    maxcalls = 100,
    options = {lower = {-3, -3}, upper = {3, 3}},
    func = function(x, y)
        return (x - 0.3)^2 + 2 * (y + 0.4)^2
    end
}

local point, cost = opt:run()
print(opt:status(), opt:ncalls(), cost)
```

The callback receives the coordinates as separate arguments. It can run an
expensive simulation, perform measurements, and then return the cost. Do not
call `run()` again on the same optimizer; create a fresh object for another run.

### Driving SIMION one evaluation at a time

If the workbench program needs to control each simulation, omit `func`. The
constructor immediately prepares the first request:

```lua
local opt = Optimizer{
    algorithm = 'SPSA',
    start = initial_parameters,
    step = parameter_scales,
    maxcalls = 100,
    options = {lower = lower_bounds, upper = upper_bounds, seed = 1}
}

while opt:running() do
    local parameters = {opt:values()}
    -- Apply parameters, rebuild/refine if needed, fly the beam, then measure.
    local cost = evaluate_model(parameters)
    opt:result(cost)
end

local best_parameters, best_cost = opt:best()
```

`evaluate_model` is the part of **your workbench Lua program** that applies a
candidate parameter set and measures its result. There is no additional program
to install. For the electrostatic GEM-based models used here:

- If only independently adjustable electrode voltages change, reuse the refined
  fast-adjust arrays and apply the new voltages with **Fast Adjust**.
- If electrode shapes, sizes or internal positions change in the potential
  array, rebuild that geometry from the GEM description and run **Refine** before
  flying the next beam. The old electrode solutions describe the old geometry.

Your workbench program performs these steps; the optimizer only proposes numbers
and receives the resulting objective value. Moving an unchanged PA instance in
an IOB is a different kind of geometry change and does not automatically require
recompiling its GEM. See [SIMION's potential-array documentation](https://simion.com/info/potential_array_types.html).
Always submit the result of the final requested evaluation: it may improve the
returned solution even when it exhausts the budget.

## 2. Minimize several objectives

There are two ways to use this library with several objectives:

| Approach | Available methods | What the evaluator returns |
|---|---|---|
| Combine objectives into one cost (scalarization) | Any scalar minimizer, including GA, PSO, JADE, CMA-ES, SPSA and the local methods | One number defining the chosen compromise. |
| Search using the objective vector and maintain a Pareto archive | NSGA-II/MOPSO/ParEGO (vector mode inferred); EHVI for exactly two objectives, or legacy GA/PSO with `mode='multiobjective'` | A table of two or more objective values. |

**GA and PSO support both modes.** With `mode='scalar'` (the default), they
minimize one cost using their original algorithms. With `mode='multiobjective'`,
GA uses NSGA-II selection and variation, and PSO uses a crowding-based Pareto
archive to guide its particles. ParEGO and EHVI always use multiobjective mode,
inferred when the mode argument is omitted. EHVI requires exactly two objectives
and a fixed reference point; see its dedicated section below.
JADE and the other scalar-only minimizers require one cost. LM instead accepts
signed residual vectors for least squares; these are not Pareto objectives.

### Combine the objectives and use a scalar minimizer

Any scalar minimizer can optimize a user-defined compromise. For example, given
dimensionless costs `J[i]` and nonnegative weights `w[i]`:

```lua
local function scalarize(J, w)
    local cost = 0
    for i = 1, #J do cost = cost + w[i] * J[i] end
    return cost
end
```

Select scales and weights explicitly. One run finds one compromise. Repeating
runs with different weights explores other compromises; you can collect their
objective vectors and retain the non-dominated solutions. Weighted sums can miss
nonconvex parts of a Pareto front. The current scalar interfaces do not build
that multi-run Pareto archive automatically.

### Give the objective vector to GA, PSO or ParEGO

All three receive a table `{f1, f2, ..., fm}`, with **m >= 2**, where every component
is to be minimized. The number of objectives is independent of the number of
parameters. Objective values must be finite and the vector length must remain
constant during the run.

```lua
local opt = Optimizer{
    algorithm = 'NSGA-II', -- mode inferred; 'MOPSO' also works here.
    start = {0.5, 0.5}, step = {0.2, 0.2}, maxcalls = 60,
    options = {
        lower = {0, 0}, upper = {1, 1},
        seed = 1, population = 12
    },
    func = function(x, y)
        -- Three synthetic objectives; these are not an optical model.
        return {x*x + y*y, (1-x)^2 + y*y, x*x + (1-y)^2}
    end
}

local front = opt:run() -- equivalently opt:front() after completion
for _, solution in ipairs(front) do
    local parameters = solution.x
    local objectives = solution.objectives
end
```

The same `values()` / `result()` loop also works: pass an objective table to
`result`. There is no unique best multiobjective solution, so use `front()`
instead of `best()`. The returned archive is the **observed non-dominated set**,
not a guarantee of the full Pareto front. Equal objective vectors share one
archive entry. The archive includes observations from unfinished generations
when the budget ends; it is not restricted to the final population.

To use ParEGO in this example, change only the algorithm name to `'ParEGO'`.
It infers vector mode, retains explicit `mode='multiobjective'` compatibility,
and rejects explicit `mode='scalar'`, as do NSGA-II and MOPSO. ParEGO ignores
`population`; optionally replace it with `initial_points=7` to shorten its
default initial design. All three methods require finite bounds
on active parameters. A quantity to maximize must first be converted to a cost,
for example `-transmission`. No weights are required for GA/PSO Pareto selection.

## 3. Find a zero or solve a system of equations

All 22 algorithms can be constructed through `require('Optimization_Algorithms')`.
Newton, Broyden and Brent delegate directly to `root_solvers`; the common
constructor does not reinterpret root options, methods, statuses or evaluation
accounting. Newton/Broyden expose the finite-difference options described below
through either entry point. The original `require('root_solvers')` constructor remains
supported. Root names accept omitted mode or explicit `mode='root'`; `scalar`,
`multiobjective` and unrecognized modes are rejected. Minimizer names reject
`mode='root'`. The common constructor still defaults to Nelder-Mead; the legacy
root constructor still defaults to Newton.

All three root solvers make signed functions zero. Choose them according to the
information available:

| Method | Number of unknowns | Starting information |
|---|---|---|
| `Brent` | One | An interval whose endpoint values have opposite signs, or an endpoint already at zero. |
| `Newton` | One or several | An initial point; derivatives or the Jacobian are estimated by finite differences. |
| `Broyden` | One or several | An initial point; the estimated Jacobian is updated as the search proceeds. |

Return **signed conditions**, not squared errors or a scalar optimization cost
duplicated across coordinates. For a coupled system, for example:

```lua
local Roots = require 'Optimization_Algorithms'

local solver = Roots{
    algorithm = 'Broyden', start = {1, 1}, step = {1, 1}, maxcalls = 50,
    options = {ftol = 1e-8, residual_scale = {1, 1}},
    func = function(x, y)
        return {x + y - 3, x - y - 1}
    end
}

local point, signed_values, status = solver:solve()
```

Newton and Broyden require **n independent signed equations for n unknowns**.
Newton builds a finite-difference Jacobian; Broyden applies a rank-one Jacobian
update, with refreshes when necessary. Both use a pivoted linear solve and a
damped step. They are local methods and may fail if the equations are singular,
inconsistent, invalid at trial points, or poorly scaled.

`residual_scale[i]` is a positive characteristic scale in the units of condition
`i`. The root tolerance applies to `max(abs(F[i])/residual_scale[i])`. Remove
fixed unknowns and their corresponding equations from a root problem; zero
parameter steps and equal bounds are not supported for Newton/Broyden.

For workbench control, omit `func` and use `running()`, `values()` and
`result(signed_values)`. Afterward `best()` returns the point, signed values, and
their scaled infinity norm. A scalar result is accepted only in one dimension.

For one signed equation with a known sign-changing interval, choose Brent through
the same interface:

```lua
local solver = Roots{
    algorithm = 'Brent', bracket = {0, 3}, maxcalls = 50,
    func = function(x) return x*x - 4 end
}
local point, signed_values, status = solver:solve() -- zero at +2
```

Brent assumes continuity within its bracket. Newton and Broyden can also solve a
single equation, but start from a point and use derivative information instead
of relying on a sign-changing bracket. A small step or bracket does not by itself
certify a small residual; inspect both the status and the returned values.

Both Brent names use the common constructor: `Brent_Minimize` minimizes a scalar
cost and `Brent` finds a signed zero. The names are never inferred from `mode`.
For example, on `[0,3]`, the function `x*x-4` has its **zero at 2**, but its
**minimum at 0**, with value -4. `Brent` finds the zero; `Brent_Minimize` searches
for the minimum. This distinction applies regardless of which root solver you
choose. Bounded minimization does not guarantee the global minimum of a
multimodal function.

## Method names

These are the exact public names. The documented legacy GA/PSO vector forms remain supported.
Every scalar minimizer can also optimize a scalarized multiobjective cost.

| Name | Problem | Main implementation details |
|---|---|---|
| `Nelder-Mead` | Scalar minimum | Native SIMION simplex. |
| `SPSA` | Scalar minimum | Simultaneous perturbations, directional curvature and Armijo line search. |
| `JADE` | Scalar minimum | Adaptive differential evolution with current-to-pbest/1 and an archive. |
| `Pattern_Search` | Scalar minimum | Coordinate pattern search with bound handling. |
| `MOPSO` | Pareto search; vector mode inferred | Alias of PSO/vector, with identical implementation and options. |
| `NSGA-II` | Pareto search; vector mode inferred | Alias of GA/vector, with identical implementation and options. |
| `PSO` | Scalar minimum or Pareto search | Scalar: global-best inertia PSO. Vector: crowding-based MOPSO with archive leaders. |
| `GA` | Scalar minimum or Pareto search | Scalar: elitist real GA. Vector: NSGA-II with bounded simulated-binary crossover and polynomial mutation. |
| `Simulated_Annealing` | Scalar minimum | Gaussian proposals and Metropolis acceptance. |
| `Gradient_Descent` | Scalar minimum | Finite differences, projection and Armijo line search; optional Barzilai--Borwein initialization of the line-search step. |
| `BFGS` | Scalar minimum | Safeguarded inverse-BFGS update and Armijo line search. |
| `Brent_Minimize` | Scalar minimum | One active parameter, finite interval. |
| `BOBYQA` | Scalar minimum | Powell/NLopt numerical routines ported to Lua; at least two active parameters, or one observation if all fixed. |
| `CMA-ES` | Scalar minimum | Full covariance, rank-one/rank-mu updates and cumulative step-size adaptation. |
| `BADS` | Scalar minimum | Local Gaussian process, Bayesian search and MADS-inspired polling. |
| `ParEGO` | Objective vector / Pareto search | Augmented Tchebycheff scalarization, GP and expected improvement. |
| `EGO` | Scalar minimum | Global ordinary-kriging GP and stable logarithmic expected improvement; finite bounds. |
| `EHVI` | Exactly two minimized objectives; vector mode inferred | Sequential analytical Gaussian EHVI in log space; finite bounds and fixed reference. |
| `LM` | Signed nonlinear least squares, m >= n_active | Safeguarded damped steps using pivoted QR; box bounds and fixed parameters. A minimum need not be a root. |
| `Newton` | Signed equations | Finite-difference Jacobian through `root_solvers`; scalar or square system. |
| `Broyden` | Signed equations | Initial/refreshed finite-difference Jacobian and rank-one updates through `root_solvers`. |
| `Brent` | Signed equation | Common constructor delegates to `root_solvers`; one variable and a sign bracket. |

SPSA can optionally use `options.restart=true` for stagnation recovery. This is
a setting of the same method, not a separate public algorithm.

## Parameters, options and results

For minimization, the constructor accepts `algorithm`, `start`, `step`,
`maxcalls`, `minradius`, `mode`, `func` and `options`. Put method-specific settings
inside **one `options` table**. The default minimizer is `Nelder-Mead`, the default
mode is `scalar` except for the vector-only names NSGA-II, MOPSO, ParEGO and EHVI,
and the default budget is 250 requests.

LM delegates before those scalar/Pareto defaults. It uses a separate
least-squares object, omitted or `least_squares` mode, a 200-request default,
and its own documented result tuple and option validation; see [LM](#lm-signed-nonlinear-least-squares).

Root names are dispatched before minimizer-only validation and defaults. They
retain the root API's 200-request default, zero-budget support, optional Brent
start and top-level `bracket` (also accepted in `options`, or derived from finite
bounds). Root options remain in `options`; `minradius` is not a root tolerance.
With a callback, minimizers use `run()` and roots use `solve()`; the common
constructor does not rename methods or alter their return values. Root `best()`
returns point, raw signed residuals and scaled infinity norm; `solve()` returns
point, raw signed residuals and status. Roots retain their existing `ncalls()`
accounting and do not acquire minimizer-only `nevaluations()`/`nrejected()` methods.

- `start` contains initial values in your chosen units.
- `step` does not have an identical effect in every method. For most scalar
  methods, `abs(step[i])` supplies a coordinate scale; native Nelder-Mead uses
  initial simplex displacements. It is not a fixed increment on every iteration.
  Omitted steps default to one for minimization. See the scale table below.
- For minimization, `step[i]=0` or equal lower/upper bounds fix a parameter.
- `options.lower` and `options.upper` have one entry per parameter. The initial
  point must satisfy the bounds. Scalar requests outside the bounds consume the
  request budget but are rejected without evaluating the physical model.
- Randomized methods use `options.seed` (default `1`) for a private deterministic
  sequence; their draws do not change Lua's global particle random generator.
  Deterministic methods do not acquire randomization from this option; LM rejects
  it. Native Nelder–Mead uses the installed SIMION implementation.
- `minradius` is a method-dependent parameter-space stopping tolerance, not a
  universal estimate of solution accuracy. Objective/gradient tolerances are
  separate method options.

### Dimension, bounds and fixed parameters

| Methods | Requirements |
| --- | --- |
| Most scalar minimizers | Optional bounds; zero step or equal bounds fixes a coordinate. |
| `Brent_Minimize` | Exactly one active coordinate with finite bounds; an all-fixed problem is rejected. |
| `BOBYQA` | Finite active bounds; at least two active coordinates, or an all-fixed one-evaluation problem. |
| NSGA-II/MOPSO and vector GA/PSO, ParEGO, EGO | Finite active bounds; fixed coordinates allowed. |
| `EHVI` | Finite bounds on every coordinate, including fixed ones; exactly two objectives and explicit fixed reference. |
| `BADS` | Finite plausible bounds inside hard bounds; hard bounds may be infinite. Missing plausible bounds use finite hard bounds or start/step ranges. |
| Newton/Broyden | Square signed system, positive coordinate scales, no fixed unknowns; bounds optional. |
| Root `Brent` | One signed function and a finite bracket (or finite bounds); positive scale, no fixed unknown. |
| `LM` | At least as many signed residuals as active coordinates; explicit positive residual scales; optional bounds/fixed coordinates. |

### What controls the size of a parameter change?

| Method / mode | Practical meaning of the scale settings |
|---|---|
| Most scalar methods, including SPSA, Pattern Search, BFGS and the BOBYQA wrapper | `abs(step)` scales the coordinates. Probes or moves also depend on the method's options. |
| Native Nelder-Mead | `step` supplies the initial simplex displacements. |
| GA/PSO in multiobjective mode, ParEGO, EHVI and scalar EGO | Finite bound widths set the coordinate scale. Changing a positive `step` magnitude alone does not rescale the search; zero still fixes a coordinate. |
| BADS | Plausible bounds and optional transformations set the scale. Finite hard bounds supply default plausible bounds; steps provide fallback ranges when those bounds are absent. |

For example, a 100 V scale and SPSA's `c_iteration=0.2` give a 20 V probe
component. This interpretation does not carry over to ParEGO or bounded BADS.
A very small finite-difference probe may also be too small to resolve your
model's numerical response. Check an actual voltage perturbation before relying
on a computed derivative.

### Method-specific settings

Useful method options include `population` for population methods, `fd_step`,
`gradient_tolerance` and `initial_alpha` for Gradient Descent/BFGS, and
`c_iteration` for SPSA. Gradient Descent additionally accepts
`bb_variant='none'|'long'|'short'|'alternate'`; see the
[BB formulas and safeguards](#gradient-descent-and-barzilai--borwein-step-initialization).
BB initializes the existing line search and needs no extra objective evaluation.

Selected older scalar defaults are listed below; `n` means active coordinates.
These are method controls, not interchangeable tolerances. The source headers
contain the remaining detailed safeguards.

| Method | Selected defaults and normal stopping |
| --- | --- |
| Pattern Search | `shrink=0.5`, `expansion=1`, `opportunistic=true`; physical poll radius or budget. |
| GA / JADE / PSO (scalar) | `population=max(8,min(24,2*n))`, `initial_spread=2`; method radius or budget. GA: `crossover=0.8`, `mutation=0.3`, `mutation_scale=1`, `elite=1`. |
| JADE | `pbest_fraction=0.1`, `adaptation=0.1`, `initial_F=initial_CR=0.5`, `F_spread=CR_spread=0.1`. |
| PSO (scalar) | `inertia=0.7298`, `cognitive=social=1.49618`, `velocity_limit=4` in step-normalized coordinates; radius includes spread and speed. |
| Simulated Annealing | `temperature=0.05`, `cooling=0.98`, `temperature_min=1e-6`, `proposal_scale=2`, `min_proposal_scale=0.05`, `reheat_after=0`, `max_reheats=2`; proposal radius or budget. |
| Gradient Descent / BFGS | `gradient_tolerance=1e-7`, `initial_alpha=1`, `armijo=1e-4`, `backtrack=0.5`, `min_alpha=1e-14`, `curvature_tolerance=1e-10`, `max_iterations=1000`, `max_linesearch=40`; gradient/step, line-search/derivative failure, iteration limit or budget. GD `bb_variant=none`. |
| Brent minimization | `relative_tolerance=sqrt(epsilon)`, `x_tolerance=minradius`, `max_iterations=1000`; interval tolerance or budget. |
| BOBYQA | `rhobeg=1` (reduced to half the narrowest normalized bound width), `npt=2*n+1`, `rhoend=minradius/max(abs(step))`; trust radius, optional `stopval`, or budget. |
| CMA-ES | `sigma=1`, `population=4+floor(3*log(max(1,n)))`, `max_iterations=100000`; physical covariance radius, conditioning/representation limits or budget. |
| BADS | `objective_scale=1`, `min_poll=1e-6`, `search_candidates=256`, `gp_sweeps=3`, `log_transform=true`, `deterministic=true`, `search_steps=max(n,3+floor(n/2))`, `max_stall=4+floor(n/2)`; radius/poll/stall or budget. |

### Finite-difference options: GD, BFGS, Newton, Broyden and LM

Only `Gradient_Descent` (including every `bb_variant`), `BFGS`, `Newton`,
`Broyden` and `LM` apply these options to the physical objective or signed residuals:

| Option | Default | Accepted values |
|---|---|---|
| `difference` | `'central'` | `'central'` or `'forward'`, with exact spelling. |
| `fd_step` | `1e-5` | A scalar, or a dense vector with one entry per original parameter. Each entry must be finite and in `[1e-12, 1]`. |

The defaults remain **central differences and scalar `fd_step=1e-5`**. A scalar
applies the same normalized value to every coordinate. A vector follows the
original `start`/`step` order, including positive entries for fixed coordinates
in scalar minimizers; it is not compressed to active coordinates. Newton and
Broyden still require positive characteristic scales and no fixed unknowns.
LM skips fixed parameters and accepts rectangular residual Jacobians; its
distinct least-squares API and stopping rules are documented below.

The values are **dimensionless normalized half-steps**, not volts or millimetres.
For an active coordinate, let `s[j]=abs(step[j])` and
`z[j]=(x[j]-start[j])/s[j]`. The requested normalized probe distance is

```text
h[j] = max(fd_step[j] * max(1, abs(z[j])),
           8 * machine_epsilon * abs(x[j]) / s[j])
```

For scalar `fd_step`, use that scalar in place of `fd_step[j]`. Before bound
shortening and floating-point reconstruction, the physical half-step is
`s[j]*h[j]`. Thus it grows with displacement from the supplied start, not simply
with the magnitude of the physical coordinate. `residual_scale` rescales root
conditions and does not set the coordinate probe size.

For a scalar forward stencil, for example:

```lua
options = {difference='forward', fd_step=1e-4}
```

For a two-parameter voltage/radius problem with `start={60,4}` and
`step={10,0.5}`, separate the voltage and geometry probes:

```lua
options = {
    difference = 'forward', -- omit this to retain central differences
    fd_step = {1e-5, 0.05},
    lower = {30, 3.4}, upper = {85, 4.8}
}
```

At that start, these values request a voltage displacement of `0.0001 V` and a
radius displacement of `0.025 mm`. They are interface examples, **not recommended
universal probe sizes**. For Newton/Broyden with `step={2,0.5}`, the same vector
instead requests an initial voltage displacement of `0.00002 V`. Both formulas
can be compared without changing the physical problem or request budget.

Central mode uses both feasible sides and the existing unequal-spacing
three-point formula; it uses a one-sided formula when only one side is usable.
Forward mode normally uses `(f_plus-f_center)/h_plus`. At an upper bound, or when
the existing near-bound rule prefers the longer negative side, it uses
`(f_center-f_minus)/h_minus`. Actual reconstructed distances are used in the
denominators. Every formula reuses the already measured center value; it does
not request another baseline for each coordinate or Jacobian column.

A **returned nonfinite numeric value** (NaN or infinity), or a root residual
vector with a nonfinite raw/scaled component, makes a probe unusable. In forward
mode the opposite feasible, representable side is then investigated, including
a short side excluded by the normal near-bound preference. Failed probes and
fallback evaluations spend the same `maxcalls` budget. Exhaustion before the
fallback reports `budget`; no usable side reports `finite_difference_failed`
or `jacobian_failed`.

A **Lua callback exception**, such as `error(...)` or a failed `assert`, still
aborts and propagates; it is not converted into a recoverable numeric result.
Scalar minimizers require a numeric result; root solvers require a scalar in one
dimension or a vector of the required length. An invalid outer type or length
causes an error; an unusable component inside a correctly sized root vector is
treated as an invalid evaluation. Use numeric infinity for an explicit
recoverable scalar failure, or an appropriately sized residual vector
containing infinity for a recoverable system failure. Returning `nil` is not
that failure contract. Finite penalties remain ordinary objective values and
do not automatically trigger opposite-side recovery. Independent final
verification/display flights remain application work outside `maxcalls` unless
the application explicitly reserves them in a larger budget.

### Methods that do not apply these options

Do not pass `difference` or vector `fd_step` to other algorithms. `Nelder-Mead`,
`SPSA`, `JADE`, `Pattern_Search`, `PSO`, `GA`, `Simulated_Annealing`,
`Brent_Minimize`, `BOBYQA`, `CMA-ES`, `BADS`, `ParEGO`, `EGO`, `NSGA-II` and `MOPSO`
do not apply them. This includes the legacy vector modes of GA/PSO. Under the
existing permissive method-options API, those keys can be silently retained or
ignored; their appearance in `options()` is not evidence of support. This
change does not introduce a new global rejection policy for unused options.

Signed-root `Brent` also performs no finite-difference probes. Its existing
constructor still validates an unused **scalar** `fd_step`, so a vector is
rejected there; `difference` does not alter its interpolation/bisection search.
Omit both settings for this method. SPSA's physical perturbation size is still
`c_iteration`, BOBYQA's interpolation design is controlled by `rhobeg`/`npt`, and
BADS/ParEGO/EGO/EHVI surrogate calculations are not physical finite-difference probes.
EHVI rejects undocumented options, including `difference` and `fd_step`; its
acquisition calculations never apply those physical finite-difference options.

### Direct-call return conventions

| Task | Callback entry point and return |
| --- | --- |
| Scalar minimization | `run()` / `best()` return parameter table, scalar cost. |
| Pareto search | `run()` / `front()` return observed archive entries; scalar `best()` is not defined. |
| Signed roots | `solve()` returns point, raw signed residuals, status; `best()` returns point, residuals, scaled infinity norm. |
| Least squares | LM `run()` / `best()` return point, raw signed residuals, measured scaled least-squares cost; obtain status separately. No `solve()`. |

All modes support the incremental `running()` / `values()` / `result()` pattern.
Root callback exceptions propagate. In the legacy direct `solve()` path the
status may remain `running` after an exception; coroutine failures set `error`.
Treat the exception as an aborted call rather than relying on a uniform status.

### Other method-specific settings

SPSA's optional `invalid_cost` threshold excludes unusable costs from slope
estimation; choose it in the units of your objective. It needs a usable starting
point below this threshold when one is specified.

Scalar PSO reflects trial positions into the supplied bounds and resets a velocity
component only when the proposed position actually crosses that component's bound.
Interior proposals, including reflection roundoff and proposals exactly on a bound,
retain their velocity. Its stopping radius includes both swarm spread and speed in
caller-coordinate scales. This corrected scalar rule is separate from MOPSO's
clamping, damped reversal and mutation rules described below.

BOBYQA requires finite bounds on active parameters and finite objective values.
`rhobeg`, `rhoend` and `npt` control its trust region and interpolation set.
Its initial point may be adjusted close to a bound, as in the source algorithm.
BADS accepts `plausible_lower` and `plausible_upper` inside the hard bounds;
plausible bounds describe a likely region and do not restrict the final search.
Its `objective_scale` should make meaningful cost differences of order one.
Defaults and further options are documented near the beginning of each method.
BADS already reuses values within a run: its transformed-coordinate key formats
each component with `%.15g`. A matching key returns the recorded value without
a new objective dispatch, request, flight or Refine. This includes recorded
invalid values, which are excluded from GP fitting. This is existing local
memoization, not persistent physical-backend caching; key collisions at that
precision are an existing limitation. EGO/EHVI instead exclude attempted points
from proposals without substituting a cached physical result.

Use only settings documented for the selected method and mode, with exact
spelling. The common scalar/Pareto object rejects unknown top-level constructor
fields but can retain unused method-option keys. Legacy root constructors also
lack that top-level check. EHVI and LM validate their supported fields/options
strictly and reject unknown keys. An entry in the permissive objects' `options()`
is not proof of use: `populaton` does not set `population`.

For **ParEGO**, set finite bounds and an evaluation budget. By default, the initial
sampling uses `5*n+1` points, where `n` is the number of active parameters. If the
budget is less than or equal to that number, it leaves no evaluations for the
model-guided search. For a short trial, reduce `options.initial_points` to leave
some of the budget for that search. This is a budget-setting choice, not a change
to the number of objectives. The algorithmic explanation is in
[Algorithm details and validation](#implementation-notes).
For long ParEGO runs, the Gaussian-process cost can become significant because
dense GP fitting scales cubically with the number of training observations.
`options.gp_max_points` optionally limits the number of observations used to fit
the surrogate; it does not truncate the observed Pareto archive. The default
`nil` uses all observations. `options.gp_refit_interval` controls how often GP
length scales are re-optimized; between these refits, the previous length scales
are reused. The default `1` preserves full refitting at every ParEGO iteration.
These options affect only the surrogate used to select new evaluations, not the
measured objective vectors retained by `front()`.
| ParEGO setting | Meaning / default |
|---|---|
| `initial_points` | Initial design size; default `5*n+1`. |
| `rho` | Augmentation coefficient in the augmented Tchebycheff scalarization; default `0.05`. |
| `gp_nugget` | Positive GP diagonal nugget; default `1e-9`. |
| `gp_sweeps` | Coordinate sweeps used when optimizing GP length scales; default `2`. |
| `gp_max_points` | Maximum observations used to train the GP; default `nil` (all observations). |
| `gp_refit_interval` | Number of ParEGO iterations between GP length-scale re-optimizations; default `1`. |
| `candidates` | Acquisition candidates; default `max(64,64*n)`. |
| `acquisition_starts` | Number of local acquisition searches; default `4`. |
| `acquisition_iterations` | Local acquisition-search iterations; default `12`. |


### EGO: a scalar GP for expensive deterministic evaluations

`algorithm='EGO'` selects Efficient Global Optimization for one scalar cost.
It uses the same direct callback or `values()`/`result()` interface as other
scalar minimizers. Finite bounds are required on every active coordinate.
The supplied start is evaluated first, followed by a seeded Latin hypercube,
then one measured point per logEI acquisition search. `best()` always returns
the best finite **physical observation**. There is no unbudgeted polishing or
final verification; the application can request a separate verification flight.

```lua
local opt = require('Optimization_Algorithms'){
    algorithm='EGO', start={60,4}, step={10,0.5}, maxcalls=60,
    options={lower={30,3.4}, upper={85,4.8}, seed=1}
}
while opt:running() do
    local x={opt:values()}
    opt:result(evaluate_model(x)) -- one scalar cost in the original units
end
local best, cost=opt:best()
```

EGO maps active parameters to `[0,1]` using their bound widths. Nonzero `step`
magnitudes do not affect this mapping; zero steps and equal bounds fix variables.
It reuses ParEGO's ARD squared-exponential ordinary-kriging GP, GLS constant mean,
Cholesky solves and concentrated-likelihood length-scale search. Costs receive
an affine normalization from the finite observations. **The physical objective
is never logarithmically transformed.** LogEI is a logarithm of an acquisition,
not a logarithm of measured costs; zero and negative costs are supported.

| EGO option | Meaning / default |
|---|---|
| `seed` | Private integer RNG seed; `1`. |
| `initial_points` | Includes the start; `min(2*n+1,max(1,floor(maxcalls/3)))`, for `n` active variables. Capped by the available budget. |
| `gp_nugget` | Correlation diagonal regularization; `1e-9`, increased up to `0.1` only if fitting fails. |
| `gp_sweeps` | Length-scale optimization sweeps per refit; `2`. |
| `gp_max_points` | Maximum valid GP training observations; `80`. Keeps the incumbent, newest observation and a spread in observation order. All observations still count for `best()`. |
| `gp_refit_interval` | Length-scale reoptimization interval in model-search iterations; `5`. The GP is rebuilt with new data between these refits. |
| `candidates` | Cheap acquisition candidates; `max(128,64*n)`, plus axis endpoints. |
| `acquisition_starts` | Bounded pattern-search starts for logEI; `4`. |
| `acquisition_iterations` | Pattern-search iterations per start; `16`. |

All GP fitting, prediction and acquisition searches are Lua calculations and
spend **no physical evaluations, flights or Refines**. Every proposed physical
evaluation, including an invalid returned cost, spends one `maxcalls` request.
EGO stays within bounds and does not re-evaluate an attempted physical point.
It also excludes points within squared distance `1e-14` of previous attempts in
normalized active coordinates, including invalid attempts. This is proposal
exclusion, not a physical-result cache: no stored result substitutes for a new
physical measurement. Exclusion does not model feasibility between those points.

The existing scalar contract requires a number. To mark an unavailable physical
measurement, return numeric NaN or positive/negative infinity. EGO excludes it
from GP training and from the incumbent; it does not train on an arbitrary
penalty. `nil`, booleans and tables are API errors. Lua callback exceptions
abort the run and are not converted into beam losses. With fewer than two valid
observations, or if GP fitting fails, seeded space filling continues. An entirely
invalid run can exhaust the budget with `best()` returning `nil, math.huge`.
No feasibility GP or hidden constraint model is implemented.

Normal stopping is `budget`; all-fixed problems take one measurement and stop as
`fixed_variables`. `design_resolution` means the candidate search found no new
representable point at the exclusion resolution; it is not a global-optimum
certificate. `minradius` is not an EGO stopping criterion. `info()` identifies
`initial`, `ego_design`, `ego_logei` or fallback `ego_space_filling` requests;
model requests also report training-point count, preceding invalid count,
refit flag and normalized-cost logEI. A fallback logEI is `-math.huge`.

This compact sequential optimizer targets deterministic problems of roughly
1--12 parameters and 50--250 evaluations. Dense fitting is cubic in the training
limit; surrogate work can still matter for fast physical models. Multistart
pattern search only approximates the acquisition maximum. A smooth stationary
kernel can fit discontinuous particle-loss boundaries poorly, and point exclusion
does not learn those boundaries. No noisy, batch or constrained-EGO claim is made.

For **GA/PSO in multiobjective mode**, `options.population` defaults to 20.
The starting point is evaluated first; the rest of the initial population is
sampled across the supplied bounds. Choose a population smaller than `maxcalls`
to leave evaluations for evolution or particle moves. For example, a population
of 12 and a budget of 60 allow initialization and four generations. These modes
stop at the evaluation budget (or immediately after evaluating an all-fixed
problem); `minradius` does not stop them or certify Pareto convergence.

| Multiobjective setting | Meaning / default |
|---|---|
| `population` | Number of GA individuals or PSO particles; 20. |
| `mutation` | Polynomial-mutation probability **per active coordinate**; `1/n`. Different from scalar GA's per-child mutation. |
| `mutation_eta` | Mutation locality; 20 (larger means smaller typical changes). |
| GA `crossover`, `crossover_eta` | Crossover probability per pair, 0.9; locality, 15. |
| PSO `inertia`, `cognitive`, `social` | Velocity weights; 0.7298, 1.49618, 1.49618. |
| PSO `velocity_limit` | Maximum move before mutation, as a fraction of each bound width; 0.2. Different units from scalar PSO. |
| PSO `archive_size` | Maximum number of leaders retained for particle guidance; 100. This does **not** truncate `front()`. |

Use settings appropriate to the selected mode. Scalar GA's `elite` and
`mutation_scale`, and scalar GA/PSO's `initial_spread`, do not control these
Pareto implementations. Their full observed archive uses memory proportional
to the number of non-dominated evaluations, at most `maxcalls` entries.

The following table applies to common scalar/Pareto objects and, where listed
in its section, EHVI. Roots and LM have their own result/counter APIs.

| Scalar/Pareto method | Result |
|---|---|
| `ncalls()` | All optimization requests, including probes, line searches and internal bound rejections. |
| `nevaluations()` | Requests actually passed to the objective. |
| `nrejected()` | Requests rejected internally because of bounds. |
| `best()` | Best evaluated parameter table and scalar cost. |
| `front()` | A copy of the observed non-dominated entries `{x=..., objectives=...}` in multiobjective mode: GA/NSGA-II, PSO/MOPSO, ParEGO or EHVI. Objectives remain in original cost units; surrogate predictions are excluded. |
| `status()` | Stop reason; budget exhaustion does not imply convergence. |
| `info()` | Metadata for the current evaluation, such as phase and iteration. |
| `options()` | Resolved method settings, when available. |

Request and evaluation counters advance before feedback is supplied, so they can
include one pending evaluation. For minimizers, `ncalls() = nevaluations() + nrejected()`;
`nevaluations()` counts objective dispatches, not completed physical flights.
The application must separately count particle flights, Fast Adjust and Refine:
one objective dispatch may perform none, one or several such operations.

The returned best point may come from an exploratory probe rather than the last
accepted iterate. Save it explicitly if your model should finish at that point.
The library does not perform an unbudgeted final simulation.

Invalid physics must be identified by the application. Some minimizers require
an explicit finite physical penalty for a lost beam. EGO and EHVI instead exclude
returned nonfinite observations from their GPs; follow their documented rules
for invalid returned values rather than inventing GP penalties. EHVI also accepts an
explicit invalid-result record, as described in its dedicated section below.
A loss event is not a signed zero.
For root finding, inspect solver status rather than turning an invalid
measurement into a zero. Keep the same particle sample across evaluations when
testing deterministic methods.


## 5. Invalid flights, constraints and accounting

The starting policy in this guide is **stop with an explicit error** when the
measurement is unusable. That is safer than silently returning zero or reusing
the previous beam. It also makes missing measurement code visible on the first
baseline evaluation. The unmodified root solver should never receive a fabricated
signed value for an invalid optical measurement.

For scalar or Pareto searches that must continue through lost beams, explicitly
replace this policy in the evaluator and objective mapping. For methods requiring
finite costs, use finite penalties in objective units and return the same vector
length on every trial.
Choose penalties worse than the admissible valid costs under a documented bound
or normalization; an arbitrary large number is not a universal guarantee.
Keep raw validity and transmission in the log, and never accept a final design
solely because it has the lowest recorded penalty. If all trials are invalid,
there is no usable optical solution even if `best()` or `front()` is nonempty.
EGO and EHVI use their documented invalid-result conventions instead of GP
penalties. Invalid returned observations still spend the request budget; Lua
callback exceptions abort. Technical SIMION errors are not physical penalties.

Bounds constrain the chosen parameters. A requirement such as minimum transmission
or magnification tolerance is an additional physical requirement, not automatically
a hard constraint enforced by the optimizer. A weighted cost or Pareto archive
must still be checked against it. This library has no general nonlinear-constraint
interface. Do not inherit CPOpt's reference-ion gate or `SPSA.invalid_cost=1` unless
your objective has exactly the intended semantics and scale.

`maxcalls` counts requests, including probes and line-search trials; it is not a
number of optimizer iterations. For minimizers, `nevaluations()` counts requests
passed to the evaluator and `nrejected()` counts internal bound rejections. If
one evaluation runs several beams, maintains a cache, or includes non-flight work,
record those counts separately. The example's `beam_count` counts its own `run()`
calls. No calls to Refine are present in its fixed-geometry loop.

Always deliver the result for the final pending request before leaving the loop.
An optimizer's best point may be an earlier probe. Save the returned vector,
not just the last voltages. Reapplying it or running a final validation beam is
an explicit application action inside the appropriate SIMION flight context;
record that extra beam separately from the optimization budget. Nothing in the
library automatically runs it or chooses a Pareto compromise for you.


## Implementation notes

BADS here supports deterministic objectives with seeded randomized searches.
Its documented adaptations include a Latin-hypercube initial design, coordinate
optimization of GP hyperparameters, rebuilt GP factorizations, a smaller
default candidate population and a simplified refitting trigger. It does not
implement noisy objectives, periodic variables or nonlinear barriers, and is
not claimed to be numerically identical to MATLAB BADS or PyBADS.

ParEGO uses the published augmented Tchebycheff/GP/expected-improvement framework
with an ARD squared-exponential GP and multistart pattern optimization of the
acquisition. CMA-ES uses positive recombination weights and full covariance;
it does not include the optional negative-weight active-CMA extension.

DFO-LS and a complete TRF implementation remain outside the present scope.
For explicitly supplied signed residuals, see [LM least squares](#lm-signed-nonlinear-least-squares).
There is no requirement to rewrite a physical model into per-particle residuals.


### Gradient Descent and Barzilai--Borwein step initialization

Gradient Descent can optionally initialize its Armijo line search with a
Barzilai--Borwein (BB) spectral step. This is an option of the same gradient
method, not a separate optimizer. With successive accepted normalized-coordinate
steps $s_k=x_{k+1}-x_k$ and gradient changes
$y_k=\nabla f_{k+1}-\nabla f_k$, the available choices are

- `bb_variant='long'` (BB1): $\alpha_k=s_k^T s_k/(s_k^T y_k)$;
- `bb_variant='short'` (BB2): $\alpha_k=s_k^T y_k/(y_k^T y_k)$;
- `bb_variant='alternate'`: alternate BB1 and BB2, starting with BB1;
- `bb_variant='none'`: retain the previous behavior and start every line search
  from `initial_alpha`.

A BB estimate is accepted only when the secant curvature is finite and positive;
otherwise the implementation falls back to `initial_alpha`. The resulting value
is only the starting step for the existing Armijo backtracking and projected
bound handling. BB therefore adds no objective evaluations by itself; its value
is to propose a curvature-informed first trial and potentially reduce unnecessary
line-search contractions. BFGS does not use this option because it already keeps
a matrix approximation of inverse curvature.

Reference: J. Barzilai and J. M. Borwein, *Two-point step size gradient methods*,
IMA Journal of Numerical Analysis **8**, 141--148 (1988),
https://doi.org/10.1093/imanum/8.1.141.

Its physical benefit depends on the objective, numerical settings and resolved
probe sizes; it is not a guarantee of fewer trajectories on every problem.

### How ParEGO proposes new points

ParEGO changes the scalarization weights, rescales the observed objective values,
fits a Gaussian process, and searches an expected-improvement acquisition.
The `parego.lua` header and the method-specific settings above explain the initial
design and its evaluation budget. These details describe the present implementation,
not a requirement that physical objectives have any particular meaning.

### EGO and stable logEI

For minimization, let `delta=best-mean`, `sigma=sqrt(variance)` and
`z=delta/sigma`. Expected improvement is `sigma*(phi(z)+z*Phi(z))`, with standard
normal density `phi` and CDF `Phi`. EGO maximizes
`log(sigma)+log(phi(z)+z*Phi(z))` directly. It never takes a logarithm after an
ordinary EI calculation has underflowed. At zero variance the exact limit is
`log(delta)` for positive improvement and negative infinity otherwise.

For `z<=-1`, set `a=-z` and use Laplace's normal Mills-ratio continued fraction
`R(a)=1/(a+1/(a+2/(a+3/...)))`. If `b=1/(a+2/(a+3/...))`, the cancellation-prone
factor `1-a*R(a)` equals the positive ratio `b/(a+b)`. Thus logEI uses
`log(sigma)-a*a/2-log(sqrt(2*pi))+log(b)-log(a+b)` without subtracting nearly
equal tail terms. The continued fraction is evaluated backwards with depth
`max(24,ceil(400/a^2))`. The center uses a convergent normal-integral series;
the large positive tail uses the deterministic-improvement limit. A log value
beyond the floating-point range can still be negative infinity.

References: Jones, Schonlau and Welch (1998),
[Efficient Global Optimization of Expensive Black-Box Functions](https://doi.org/10.1023/A:1008306431147);
Ament et al. (2023),
[Unexpected Improvements to Expected Improvement for Bayesian Optimization](https://arxiv.org/abs/2310.20708);
and [NIST DLMF section 7.9](https://dlmf.nist.gov/7.9) for equivalent error-function
continued fractions. The GP follows
[Rasmussen and Williams, chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
Independent mathematical references and portable regressions are maintained
separately from this minimal distribution. The Lua optimizer requires no Python,
MATLAB, DLL or external optimization backend.

### Multiobjective GA and PSO

`algorithm='NSGA-II'` (or legacy `algorithm='GA', mode='multiobjective'`) selects a real-valued NSGA-II
implementation: nondominated sorting, normalized crowding distance, rank/crowding
binary tournaments and elitist parent/offspring survival. Variation uses bounded
simulated-binary crossover and bounded polynomial mutation. A constant objective
adds zero crowding distance. Exact ties in environmental selection use evaluation
order; tournament ties are random. The initial population includes `start`.
The scalar GA's operators and behavior are unchanged.
Reference: Deb, Pratap, Agarwal and Meyarivan (2002),
[A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017).

`algorithm='MOPSO'` (or legacy `algorithm='PSO', mode='multiobjective'`) selects a **crowding-based MOPSO variant**.
Personal memory uses dominance, with a random choice between incomparable
vectors and no replacement on equality. Each generation freezes a leader set
from the observed front, prunes it by repeatedly removing a least-crowded entry,
and selects guides with crowding tournaments. Velocities are capped in normalized
parameter coordinates; bound crossings clamp the position and reverse half the
velocity. Polynomial mutation maintains exploration and resets affected velocity
components. This differs from the adaptive-grid repository and mutation schedule
of Coello Coello, Toscano Pulido and Salazar Lechuga (2004),
[Handling multiple objectives with particle swarm optimization](https://doi.org/10.1109/TEVC.2004.826067).
It is not claimed to reproduce that implementation. This MOPSO rule is separate from the corrected scalar PSO rule.

The API now records every finite vector evaluation in a common exact-dominance
archive for GA, PSO and ParEGO. Equal objective vectors retain one representative.
GA populations and PSO leader sets can discard points for diversity, but the
public `front()` never discards a non-dominated observation for a size limit.
It contains measured points, not an interpolated or guaranteed complete front.
No nonlinear-constraint or noise-specific dominance rules are implemented.
All components are minimized; objective meanings remain application-defined.



## Advanced CPOpt

The reconstructed two-lens/aperture example in `Examples/CPOpt` follows Huber,
Wirtz & Hoang (2024), doi:10.1016/j.nima.2024.169702. It teaches six-voltage
scalar optimization at fixed radii. This teaching example is not the complete
manuscript benchmark. Begin with the Einzel examples in [README.md](README.md).

Prepare a fresh source copy **before opening its workbench**. In PowerShell,
enter `Examples/CPOpt` and use your installed SIMION executable:

```powershell
$simion = "C:/path to/SIMION/simion-8.1.3.9-TEST-20200324.exe"
& $simion --nogui --noprompt --num-threads=2 lua prepare.lua
```

Preparation generates `cpopt.pa#`, `.pa0` through `.pa10`, and `cpopt_fields.lua`
from the local GEM/configuration, using one complete-family Refine at .005 V
with other native Refine defaults. It refuses existing fields. For a geometry or
field-setting change, prepare a new source copy.

Open `cpopt.iob`; under **Particles -> Define**, choose **Grouped (.FLY2)** and
load this directory's `cpopt_beam_601.fly2`; verify 601 particles. Press Fly'm.
The initial `mode=0` in `cpopt_config.lua` measures the nominal point. To try the
short scalar search, change it to `mode=1`, save and reopen the workbench.
Keep `maxcalls=10` for the wiring smoke test: PSO's population of 12 can exhaust this
budget during swarm initialization. A separate learning run may use 100 requests.

The six voltages on electrodes 2,3,4,6,7,8 vary; the six radii stay at 6 mm.
Transmission is detected ions divided by all 601 launched ions. FW50 is twice
the median detector radius about the axis. The cost combines spot and loss
terms with weight .5 and a 2 mm spot scale. The reference-ion gate and capped
scores are defined in `cpopt_metrics.lua`. Previously measured nominal values
were FW50 about .202 mm, 371/601 detected and cost .242.

The prepared family is reused by Fast Adjust during flights. Optimizer requests,
physical ensemble flights and preparation Refine are reported separately.
The ten-request budget is a smoke test, not convergence evidence.

## Native Einzel details

Both Einzel examples use two 100 eV, 40 u, +1 rays starting at x=20 mm, one axial
and one at +2 degrees. The virtual detector is x=80 mm. In Particles Define,
“Grouped (.FLY2)” selects the definition format; the Lua `sim_grouped=0` flies
those particles sequentially. These are separate settings. Each native `run()`
resets measurements once in `segment.initialize_run`; missing rays abort before
any objective is submitted.

`einzel.iob` loads its matching `einzel.lua` workbench program. The independently
named templates become active only after copying one over `einzel.lua` and
reopening the IOB. Each Einzel directory includes `einzel.fly2`, matching
`einzel.iob`. Check the Particles tab if the two expected rays are not selected;
load that directory's `einzel.fly2` under Particles -> Define -> Grouped (.FLY2).

The one-voltage example optimizes V2 in 0..90 V, starts at 30 V with step 15 V and
uses 32 requests plus a final flight, with no Refine. Its prepared family uses
convergence 1e-5 V and maximum overrelaxation .95. The two-parameter example uses
V2 in 30..85 V and r2 in 3.4..4.8 mm, starting at (60 V,4 mm) for scalar/Pareto.
Its complete-family Refine uses the same numerical field settings whenever r2
changes. A complete family comprises three electrode bases and the combined PA0.
The included r2=4 mm family is reloaded at each Fly'm.

The Roots template starts at (68 V,4.1 mm). Its prescribed targets are
`target_y=0.9` mm and `target_angle=-6` mrad. They are chosen conditions, not
rounded measurements from a reference configuration. With signed differences
`dy=y2-y1` and `da=theta2-theta1`, the solver receives the signed residuals
`F_y=dy-0.9` (mm) and `F_angle=da-(-6)=da+6` (mrad).

The independent acceptance tolerances are `abs(F_y)<=0.001` mm and
`abs(F_angle)<=0.01` mrad. The solver uses `residual_scale={0.001,0.01}` and
`ftol=1`, so both scaled residuals must be at most one in magnitude. A separate
final flight forces a fresh field rebuild/Refine and checks both residuals
again. Use `algorithm='Newton'` or `'Broyden'` through the common constructor.
`'Brent'` is a one-variable signed-root method; `'Brent_Minimize'` is scalar
minimization.

These small native lens examples are didactic analogues of the full Einzel
benchmark. They expose measurement/evaluation interfaces with modest budgets.

## Source credits and redistribution notices

This file records the provenance of third-party material and external
dependencies. It does **not** assign a license to the project as a whole or
to other project files. The project owner must choose any project-wide
publication license separately.

## BOBYQA numerical routines

`bobyqa.lua` contains a Lua 5.1 translation of the BOBYQA numerical routines
written by M. J. D. Powell and translated to C for NLopt by S. G. Johnson.
The routines `update_`, `rescue_`, `altmov_`, `trsbox_`, `prelim_` and
`bobyqb_` are retained, including interpolation-model updates, geometry
improvement, trust-region steps and model restoration.

The source is pinned to [NLopt commit
6e6593f131ba3a38bc9edbed0a357bc01526e54b](https://github.com/stevengj/nlopt/tree/6e6593f131ba3a38bc9edbed0a357bc01526e54b/src/algs/bobyqa).
The [original C source](https://github.com/stevengj/nlopt/blob/6e6593f131ba3a38bc9edbed0a357bc01526e54b/src/algs/bobyqa/bobyqa.c)
has SHA-256
`a1733948888e927a861197b9c8f4a84b561439fbe286d8020e6078e10135d95d`.
The upstream [README](https://github.com/stevengj/nlopt/blob/6e6593f131ba3a38bc9edbed0a357bc01526e54b/src/algs/bobyqa/README)
describes its history; the [copyright notice](https://github.com/stevengj/nlopt/blob/6e6593f131ba3a38bc9edbed0a357bc01526e54b/src/algs/bobyqa/COPYRIGHT)
grants the permissions reproduced below.

The Lua translation was generated before release. It preserves pointer
aliasing, C integer arithmetic and numerical control flow. The surrounding
adapter supplies parameter scaling, fixed-coordinate handling, finite-bound
checks and the library's evaluation interface. No compiler, Python runtime,
DLL or external optimization backend is needed to use the distributed Lua
module.

The following notice must remain with copies or substantial portions of the
BOBYQA-derived code. It is also included at the beginning of `bobyqa.lua`.

```text
Copyright (c) 2009 M. J. D. Powell (mjdp@cam.ac.uk)
Modifications Copyright (c) 2010 Massachusetts Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

Algorithm reference: M. J. D. Powell, *The BOBYQA algorithm for bound
constrained optimization without derivatives*, DAMTP 2009/NA06, 2009.

## SIMION dependency and genetic-algorithm acknowledgment

The Nelder–Mead adapter loads `simionx.SimplexOptimizer` from the user's
installed SIMION distribution. This publication directory does not contain
that module, SIMION executables, license keys or other SIMION runtime
files. SIMION remains a separately licensed external dependency. Its
[official SimplexOptimizer documentation](https://simion.com/info/lua_simionx.SimplexOptimizer.html)
describes the native implementation.

The genetic-algorithm operator choices in `evolutionary.lua` were informed
by the supplied SIMION example `geneticlib.lua`. Its historical attribution
is retained:

```text
2016-12-06, D. Manura
(c) 2016 Scientific Instrument Services, Inc. (Licensed SIMION 8.1)
```

The current module has its own implementation of roulette selection with
distinct parents and whole arithmetic crossover, together with elitism,
private random-number generation, bound handling and Gaussian mutation.
The original `geneticlib.lua` is not included. This acknowledgment identifies
the design reference; it grants no license to SIMION source files and does
not impose or declare a project-wide license.

## Other algorithm references

The [scientific references below](#scientific-references-and-implementation-scope) map
all 22 public names to verified primary sources and explains the implementation
scope. Mathematical citations do not grant software redistribution rights.
The BOBYQA notice and SIMION acknowledgment above remain separate. No MATLAB
BADS or PyBADS package is included; OptSi does not claim to reproduce them exactly.


## EHVI: analytical two-objective hypervolume search

`algorithm='EHVI'` delegates to its dedicated two-objective object. It infers multiobjective mode, accepts that explicit mode, and
rejects scalar/root modes. Exactly two finite scalar costs are minimized.
There is no hard input-dimension cap. Model cost and calibration can deteriorate
in higher dimensions.

```lua
local Optimizer = require 'Optimization_Algorithms'
local opt = Optimizer {
  algorithm='EHVI', start={0.5,0.5}, step={1,1}, maxcalls=60,
  options={lower={0,0}, upper={1,1}, seed=7, reference={2,2},
           initial_points=7, objective_offset={0,0}, objective_scale={1,1}}
}
while opt:running() do
  local x1,x2 = opt:values()
  local measured_costs = measure_two_costs(x1,x2) -- your workbench measurement
  opt:result(measured_costs)
end
local measured_front = opt:front() -- entries {x={...},objectives={cost1,cost2}}
```

For direct use, supply `func=function(x1,x2) return {cost1,cost2} end` and call
`opt:run()` once; it returns the measured front. `running`, `values`, `result`,
`front`, `status`, `ncalls`, `nevaluations`, `nrejected`, `options`, `info` and
`stats` are supported. Front/options access returns copies. `best()` rejects a
scalar-best request. Create a new object for every independent run.

All input bounds must be finite, ordered and have representable widths. Inputs
are normalized by bound width; positive step magnitudes do not rescale EHVI.
Zero step or equal bounds fixes a coordinate. An all-fixed problem measures the
start once and stops. The reference is a **fixed worse point in original cost
units**, supplied before optimization. Points outside its box can train the GP
but contribute no hypervolume there. Never choose the reference from a winning
run. Negate maximized physical quantities yourself, including their reference.

| Option | Meaning / default |
| --- | --- |
| `lower`, `upper`, `reference` | Required finite vectors; reference length exactly two. |
| `n_objectives` | Optional explicit `2`; other counts rejected. |
| `seed` | Private reproducible RNG; default `1`. |
| `initial_points` | Includes start; default `2*n_active+1`, capped by remaining budget. Leave room for model-guided evaluations. |
| `initial_design` | Optional list of distinct bounded coordinates, beginning with the exact start; overrides generated LHS. All are freshly measured. |
| `objective_offset`, `objective_scale` | Optional finite two-vectors, scale strictly positive. Transform costs/reference together by `(value-offset)/scale`. Missing partner defaults to zero/one. |
| `gp_sweeps`, `gp_nugget` | Existing GP fitter: `2`, `1e-6`. Numerical regularization is not inferred physical noise. |
| `candidates` | Global/local acquisition pool size; `256`. |
| `acquisition_starts`, `acquisition_iterations` | Bounded coordinate-search effort: `3`, `12`. No global maximum certificate. |

Without supplied fixed objective scales, observed-range normalization is used.
All returned objective vectors remain in original units. Undocumented method
options are rejected. `minradius` is not an EHVI convergence tolerance; stopping
uses budget, fixed variables or representable design resolution.

`maxcalls` includes initialization, invalid returned evaluations and an outstanding
request before feedback. Every callback dispatch spends one request. Internal
GP predictions/acquisition calculations spend no physical requests. Submit the
last feedback: there is no hidden polishing or verification call. A returned
`{valid=false,reason='known loss'}` or a numeric vector containing NaN/infinity
is charged and excluded from fitting/archive. Malformed vectors and Lua callback
exceptions abort. Technical SIMION errors must not become physical penalties.
If your objective deliberately defines finite loss scores, those are measured
cost vectors; preserve raw validity/counts separately. No result cache exists.

### Mathematical scope and method choice

For independent predictive costs `Yj ~ N(muj,vj)`, EHVI integrates
`F1(t1)*F2(t2)` over the uncovered region below the fixed reference. Its disjoint
staircase strips give `sum [A(b)-A(a)]*A(h)`, where
`A(t)=E[(t-Y)+]`. Stable scalar logEI, positive strip integrals and log-sum-exp
avoid underflow/cancellation; a narrow-strip Gaussian quadrature fallback is
validated independently. The mathematical target is analytical EHVI, evaluated
numerically in log space, not a log of a previously underflowed sum.

| Method | Criterion and scope | Practical use |
| --- | --- | --- |
| ParEGO | Scalarization, one scalar GP/EI; multiple objectives | Existing baseline; no fixed HV reference required by its search. |
| EHVI | Two independent ordinary-kriging GPs; analytical q=1 EHVI | Pure Lua, two deterministic costs, explicit fixed HV reference. |

The analytical sequential criterion is distinct from batch Monte Carlo
acquisitions. Neither surrogate predictions nor sampled fronts are physical
observations. Model calculations do not dispatch physical evaluations; their
computational cost can nevertheless be material. `stats().model_seconds` and
`acquisition_seconds` use Lua `os.clock()` and are runtime/platform diagnostics,
not portable wall-time or universally comparable CPU measurements. Bounded
multistart search only approximates the acquisition maximum. No noise inference,
batch search, correlated-objective posterior, learned physical-constraint model
or universal superiority claim is included in public EHVI.


## LM: signed nonlinear least squares

`algorithm='LM'` minimizes

```text
F(x) = 0.5 * sum_i (r_i(x) / residual_scale[i])^2
```

Supply a fixed-length dense vector of **signed** residuals and explicit positive
`options.residual_scale` entries. The number of conditions must be at least the
number of active parameters. These can be square, consistent overdetermined or
inconsistent overdetermined systems. LM minimizes a scalar least-squares cost;
it is distinct from Pareto search and from solving an exact root.

With a callback, use `run()`, which returns the same three values as `best()`:
**copied physical parameters, copied raw signed residuals, measured scaled cost**.
There is deliberately no `solve()` method. Status is obtained separately with
`status()`. A least-squares minimum can retain nonzero residuals even when
the numerical stopping test succeeds.

The native workbench workflow is unchanged:

```lua
local opt = require('Optimization_Algorithms'){
    algorithm='LM', start={60,4}, step={10,0.5}, maxcalls=100,
    options={
        lower={30,3.4}, upper={85,4.8},
        residual_scale={0.001,0.01}, ftol=1,
        difference='central', fd_step={1e-5,0.05}
    }
}
while opt:running() do
    local V2,r2 = opt:values()
    local dy,da = evaluate_signed_measurements(V2,r2) -- mm, mrad
    opt:result{dy-0.9, da+6} -- prescribed signed targets
end
local point, raw_residuals, cost = opt:best()
print(opt:status(), opt:residual_norm(), opt:residuals_satisfied())
```

This illustrates the API, not a universal probe-size recommendation. The
existing two-ray scripts are preserved. More conditions require genuinely
additional measurements or independently prescribed conditions; do not copy a
scalar RMS cost into multiple residual entries.

`residual_norm()` is the best measured point's scaled infinity norm.
`residuals_satisfied()` tests `max(abs(r_i)/residual_scale[i]) <= ftol`.
When scales are desired physical tolerances, set `ftol=1`; the default `ftol`
is `1e-8`. Low least-squares cost alone does not establish that all tolerances
are satisfied. Best is selected from **all valid supplied measurements**,
including finite-difference probes and rejected trials, and can differ from
the accepted iterate. Cost ties prefer the smaller scaled infinity norm.

### LM options and coordinate conventions

| Option | Default | Meaning |
|---|---|---|
| `residual_scale` | Required | Dense positive finite m-vector, in residual units. |
| `lower`, `upper` | Unbounded | Per-parameter box bounds; start must be feasible. Finite normalized limits must be representable. |
| `difference`, `fd_step` | `'central'`, scalar `1e-5` | Same normalized formula and baseline reuse as the finite-difference section; scalar or full parameter vector. |
| `ftol` | `1e-8` | Maximum scaled residual tolerance; zero disables tolerance above an exact zero. |
| `gtol` | `1e-8` | Numerical projected-gradient cosine threshold. |
| `xtol` | `1e-10` | Stop when the proposed feasible step infinity norm is at most `xtol*(1+norminf(z))`. |
| `cost_tol` | `1e-12` | Accepted reduction at most this fraction of the prior cost, with gain ratio at least 0.25. Zero disables this positive-reduction test. |
| `initial_damping`, `min_damping`, `max_damping` | `1e-3`, `1e-15`, `1e15` | Positive ordered damping limits. |
| `acceptance_ratio` | `1e-4` | Minimum actual/predicted reduction ratio; actual reduction must also be positive. |
| `rank_tolerance` | `1e-12` | Relative undamped QR rank threshold. |
| `max_iterations`, `max_rejections` | `200`, `20` | Jacobian iterations and damping attempts per model. |

LM defaults to a **200-request** budget and supports `maxcalls=0`.
Omitted `step` entries default to one; otherwise supply one finite entry per
parameter. Normalized coordinates are `z=(x-start)/abs(step)` on active
parameters. `step=0` or equal bounds fixes a coordinate, whose returned physical
value stays unchanged. Residual count is compared with active parameters,
while `fd_step` always uses the complete original parameter order. Both finite
and one-sided/unbounded boxes are supported. Negative step signs supply the
same absolute scale. LM accepts omitted mode or `mode='least_squares'`, and
rejects scalar/root/Pareto modes, unknown constructor fields and unknown options.
`minradius` and `seed` are not LM options. LM is deterministic and does not use
the global RNG. None of these LM conventions changes existing methods.

### Safeguards and numerical stopping

At each accepted iterate LM rebuilds the rectangular scaled-residual Jacobian.
The central/forward formulas, actual representable displacements, near-bound
longer-side preference, backward fallback and nonfinite-probe recovery follow
the finite-difference contract above. Fixed columns require no evaluations.
The already measured center is always reused. Every attempted probe spends a
request, including a failed probe and an opposite-side recovery attempt.

For a working set of free parameters the damped system is
`[J/c; sqrt(mu)*D] p ~= [-r_scaled/c; 0]`, where `c` is the largest absolute
Jacobian entry, floored at `1e-150`, and `D[j]` is the corresponding column norm
of `J/c`, floored at `1e-6`. Column-pivoted Householder QR solves the rectangular
system without forming or inverting `J^T J`. The positive diagonal damping
regularizes rank-deficient models; undamped pivoted QR separately diagnoses
rank and weak columns. The reported QR diagonal ratio is **not** a certified
condition number or a uniqueness certificate.

At a box face, outward-gradient components are held fixed in that model's
working set. Remaining components of the damped step are clipped to the box;
rounding is mapped back to actual physical coordinates. Predicted reduction
is recomputed for that **actual feasible displacement** as
`-r_scaled^T Jp - 0.5*||Jp||^2`. Positive actual reduction and a sufficient gain
ratio accept the trial. Accepted trials update damping using
`mu*max(1/3,1-(2*rho-1)^3)`; weak agreement can increase it even after acceptance.
Rejection increases it by successive factors 2,4,8,... within the configured
limits, reusing the model and measured baseline.

This is a safeguarded **box-projected LM**, not a complete trust-region
reflective solver or a general nonlinear-constraint solver. No physical geometry
is repaired. Cheap geometry rejection can return a nonfinite residual vector
without launching SIMION. A vector containing nonfinite raw/scaled entries, or
an overflowing scaled cost, is unusable. Initial invalid data stop the run;
invalid probes investigate the other feasible side, while invalid trials are
rejected and damping increases. A changed vector length, malformed result or
Lua callback exception aborts as `error`; technical SIMION errors should throw.

Inspect these distinct statuses:

- `residual_tolerance`: the best measured residual vector satisfies `ftol`.
- `least_squares_stationary`: the latest numerical model's projected-gradient
  cosine test passes; this need not be an exact root or a local minimum.
- `unresolved_jacobian`: that stopping test passes with one or more exactly zero
  measured Jacobian columns. This explicitly does **not** certify physical
  stationarity. Analytically zero derivatives and numerically unresolved
  physical derivatives cannot be distinguished automatically.
- `cost_tolerance`, `step_tolerance`, `fixed_parameters`: the named numerical
  condition stopped progress, without a root or stationarity certificate.
- `budget`, `iteration_limit`, `no_progress`, `damping_limit`: a resource or
  safeguarding limit stopped the run.
- `jacobian_failed`, `nonfinite_initial_value`, `numerical_failure`, `error`:
  unusable measurements/model, numerical failure or an aborting error.

For each nonblocked column, the gradient cosine is
`abs(J_j^T r_scaled)/(||J_j||*||r_scaled||)`; maximum is compared to `gtol`.
Blocked outward-gradient columns contribute zero. Calculations rescale the
Jacobian and residuals to avoid overflow. Any exactly zero active column changes
an otherwise stationary status to `unresolved_jacobian`. Residual satisfaction
has priority when the final permitted measurement meets tolerance.

`diagnostics()` returns copied accepted/rejected-step counts and history,
predicted/actual reductions, ratios, damping, failed probes/linear solves,
last complete Jacobian/rank/weak/unresolved/one-sided columns, working-set
indices and accepted iterate. Jacobian diagnostics describe the **last completed
model build**; after an accepted trial they may precede that final iterate and
are not an independent derivative verification at `best()`.

`ncalls()` and `nevaluations()` include the outstanding dispatched request;
`ncompleted()` counts feedback received. `ninvalid()` counts unusable returned
measurements. `nrejected()` is zero because LM generates feasible box candidates;
rejected LM steps are separately counted in `diagnostics().rejected_steps`.
An adapter's cheap geometry rejects, actual beam runs, particles and Refines
remain separate physical counters. Supplying the last pending result always
records it before stopping. There is no hidden final evaluation, physical-result
cache, automatic physical probe-size tuning or unbudgeted polishing.

In geometric SIMION models, the default half-step can be far below the field
mesh. Provide physically justified per-coordinate probes, investigate size
sensitivity separately and inspect unresolved/rank diagnostics. Increasing a
probe does not by itself validate a physical derivative. Native validation is
repeatability at fixed numerical settings, not mesh convergence.

The algorithm design follows damped least-squares principles described by
[Madsen, Nielsen and Tingleff (2004)](https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf).
Independent development checks use
[SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
and high-precision reference calculations; neither is a runtime dependency.
The bounded working-set/clipping policy is an OptSi implementation choice,
not MINPACK LM or SciPy's trust-region reflective method. Probe resolution,
rank diagnostics and nonzero-residual stopping require independent physical
interpretation; no general convergence or identification guarantee is implied.


## Scientific references and implementation scope

These sources identify mathematical foundations and implementation scope for
all 22 public names. They do not assign a project-wide license or establish exact
equivalence to external software. Required software notices remain in
[Source credits](#source-credits-and-redistribution-notices).

### Public-name map

| Public name | Primary references below | OptSi scope |
| --- | --- | --- |
| Nelder-Mead | NM; SIMION | Adapter to installed `simionx.SimplexOptimizer`, not bundled source. |
| SPSA | SPSA; Armijo | Two-sided simultaneous perturbations with OptSi curvature, line-search and optional restart safeguards. |
| JADE | JADE | Adaptive differential evolution with optional external population archive; OptSi initialization/budget adapter. |
| Pattern_Search | Pattern | Bounded compass polling with rotated order; related pattern-search framework, not a full MADS implementation or an unchanged convergence theorem. |
| PSO | PSO | Scalar swarm; explicit vector mode uses the MOPSO engine below. |
| GA | GA_background; SIMION GA acknowledgment | Independently implemented roulette selection, arithmetic crossover, Gaussian mutation and elitism; vector mode uses NSGA-II. |
| Simulated_Annealing | SA | Continuous Gaussian proposals and explicit cooling/reheat controls, not a reproduction of a particular physics application. |
| Gradient_Descent | Newton_GD; Armijo; BB | Numerical gradient and Armijo search; optional BB step initialization, not a separate public method. |
| BFGS | BFGS_B, BFGS_F, BFGS_G, BFGS_S; Armijo | Inverse-Hessian secant updates with OptSi curvature safeguards and numerical gradients. |
| Brent_Minimize | Brent | Bounded scalar parabolic/golden-section minimization; distinct from signed-root Brent. |
| BOBYQA | Powell; pinned NLopt | Mechanical Lua translation of the documented Powell/Johnson C lineage plus OptSi adapter. |
| CMA-ES | CMA; Hansen tutorial | Full positive covariance adaptation; no claim of the complete current reference-software feature set. |
| BADS | BADS | Deterministic local GP/search/poll adaptation; not identical to MATLAB BADS or PyBADS. |
| ParEGO | ParEGO; GPML | Augmented Tchebycheff scalarization; OptSi bounded acquisition pattern search instead of the paper's inner GA. |
| EGO | EGO; logEI; GPML | Sequential scalar ordinary kriging, stable logEI and approximate bounded acquisition search; measured costs are not logged. |
| EHVI | EHVI; logEI; GPML | Analytical q=1 two-objective EHVI in log space, independent Gaussian predictions, fixed reference; not BoTorch qLogEHVI. |
| Newton | Newton_GD; Armijo | Finite-difference square signed Jacobian and safeguarded line search. |
| Broyden | Broyden; Armijo | Square signed solve with initial/refreshed numerical Jacobian and rank-one secant updates. |
| Brent | Brent | One signed equation with sign bracket; interpolation plus bisection. |
| NSGA-II | NSGA | Public alias of existing GA/vector engine; no new stream or numerical method. |
| MOPSO | MOPSO | Public alias of PSO/vector engine; crowding-based leader selection, not the original adaptive-grid implementation. |
| LM | Levenberg; Marquardt; Madsen–Nielsen–Tingleff; QR | Signed nonlinear least squares with original pivoted QR and bounded working-set/clipping safeguards; not MINPACK, SciPy LM or SciPy TRF. |

### Registered mathematical references

- **NM** — J. A. Nelder, R. Mead (1965). *A Simplex Method for Function Minimization*. The Computer Journal 7(4):308-313. [DOI:10.1093/comjnl/7.4.308](https://doi.org/10.1093/comjnl/7.4.308).

- **SPSA** — J.C. Spall (1992). *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation*. IEEE Transactions on Automatic Control 37(3):332-341. [DOI:10.1109/9.119632](https://doi.org/10.1109/9.119632).

- **JADE** — Jingqiao Zhang, A.C. Sanderson (2009). *JADE: Adaptive Differential Evolution With Optional External Archive*. IEEE Transactions on Evolutionary Computation 13(5):945-958. [DOI:10.1109/tevc.2009.2014613](https://doi.org/10.1109/tevc.2009.2014613).

- **Pattern** — Virginia Torczon (1997). *On the Convergence of Pattern Search Algorithms*. SIAM Journal on Optimization 7(1):1-25. [DOI:10.1137/s1052623493250780](https://doi.org/10.1137/s1052623493250780).

- **PSO** — J. Kennedy, R. Eberhart (1995). *Particle swarm optimization*. Proceedings of ICNN'95 - International Conference on Neural Networks 4:1942-1948. [DOI:10.1109/icnn.1995.488968](https://doi.org/10.1109/icnn.1995.488968).

- **GA_background** — John H. Holland (1992). *Adaptation in Natural and Artificial Systems*. MIT Press. [DOI:10.7551/mitpress/1090.001.0001](https://doi.org/10.7551/mitpress/1090.001.0001).

- **SA** — S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi (1983). *Optimization by Simulated Annealing*. Science 220(4598):671-680. [DOI:10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671).

- **BB** — JONATHAN BARZILAI, JONATHAN M. BORWEIN (1988). *Two-Point Step Size Gradient Methods*. IMA Journal of Numerical Analysis 8(1):141-148. [DOI:10.1093/imanum/8.1.141](https://doi.org/10.1093/imanum/8.1.141).

- **BFGS_B** — C. G. BROYDEN (1970). *The Convergence of a Class of Double-rank Minimization Algorithms 1. General Considerations*. IMA Journal of Applied Mathematics 6(1):76-90. [DOI:10.1093/imamat/6.1.76](https://doi.org/10.1093/imamat/6.1.76).

- **BFGS_F** — R. Fletcher (1970). *A new approach to variable metric algorithms*. The Computer Journal 13(3):317-322. [DOI:10.1093/comjnl/13.3.317](https://doi.org/10.1093/comjnl/13.3.317).

- **BFGS_G** — Donald Goldfarb (1970). *A family of variable-metric methods derived by variational means*. Mathematics of Computation 24(109):23-26. [DOI:10.1090/s0025-5718-1970-0258249-6](https://doi.org/10.1090/s0025-5718-1970-0258249-6).

- **BFGS_S** — D. F. Shanno (1970). *Conditioning of quasi-Newton methods for function minimization*. Mathematics of Computation 24(111):647-656. [DOI:10.1090/s0025-5718-1970-0274029-x](https://doi.org/10.1090/s0025-5718-1970-0274029-x).

- **Newton_GD** — Jorge Nocedal, Stephen J. Wright (2006). *Numerical Optimization*. Springer Series in Operations Research and Financial Engineering . [DOI:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5).

- **Armijo** — Larry Armijo (1966). *Minimization of functions having Lipschitz continuous first partial derivatives*. Pacific Journal of Mathematics 16(1):1-3. [DOI:10.2140/pjm.1966.16.1](https://doi.org/10.2140/pjm.1966.16.1).

- **CMA** — Nikolaus Hansen, Andreas Ostermeier (2001). *Completely Derandomized Self-Adaptation in Evolution Strategies*. Evolutionary Computation 9(2):159-195. [DOI:10.1162/106365601750190398](https://doi.org/10.1162/106365601750190398).

- **ParEGO** — J. Knowles (2006). *ParEGO: a hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems*. IEEE Transactions on Evolutionary Computation 10(1):50-66. [DOI:10.1109/tevc.2005.851274](https://doi.org/10.1109/tevc.2005.851274).

- **NSGA** — K. Deb, A. Pratap, S. Agarwal, T. Meyarivan (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II*. IEEE Transactions on Evolutionary Computation 6(2):182-197. [DOI:10.1109/4235.996017](https://doi.org/10.1109/4235.996017).

- **MOPSO** — C.A.C. Coello, G.T. Pulido, M.S. Lechuga (2004). *Handling multiple objectives with particle swarm optimization*. IEEE Transactions on Evolutionary Computation 8(3):256-279. [DOI:10.1109/tevc.2004.826067](https://doi.org/10.1109/tevc.2004.826067).

- **EGO** — Donald R. Jones, Matthias Schonlau, William J. Welch (1998). *Efficient Global Optimization of Expensive Black-Box Functions*. Journal of Global Optimization 13(4):455-492. [DOI:10.1023/a:1008306431147](https://doi.org/10.1023/a:1008306431147).

- **EHVI** — Michael T. M. Emmerich, Andre H. Deutz, Jan Willem Klinkenberg (2011). *Hypervolume-based expected improvement: Monotonicity properties and exact computation*. 2011 IEEE Congress of Evolutionary Computation (CEC) :2147-2154. [DOI:10.1109/cec.2011.5949880](https://doi.org/10.1109/cec.2011.5949880).

- **Broyden** — C. G. Broyden (1965). *A class of methods for solving nonlinear simultaneous equations*. Mathematics of Computation 19(92):577-593. [DOI:10.1090/s0025-5718-1965-0198670-6](https://doi.org/10.1090/s0025-5718-1965-0198670-6).

- **Levenberg** — Kenneth Levenberg (1944). *A method for the solution of certain non-linear problems in least squares*. Quarterly of Applied Mathematics 2(2):164-168. [DOI:10.1090/qam/10666](https://doi.org/10.1090/qam/10666).

- **Marquardt** — Donald W. Marquardt (1963). *An Algorithm for Least-Squares Estimation of Nonlinear Parameters*. Journal of the Society for Industrial and Applied Mathematics 11(2):431-441. [DOI:10.1137/0111030](https://doi.org/10.1137/0111030).

- **QR** — Peter Businger, Gene H. Golub (1965). *Linear least squares solutions by householder transformations*. Numerische Mathematik 7(3):269-276. [DOI:10.1007/bf01436084](https://doi.org/10.1007/bf01436084).

The Holland citation identifies the verified **1992 MIT Press edition**; it is
background for genetic algorithms, not a claim that OptSi copies its software
or uses exactly the original 1975 operators.

### Author and original-project sources

- **SIMION** — [Official SimplexOptimizer documentation](https://simion.com/info/lua_simionx.SimplexOptimizer.html), module version 20170106. SIMION is separately licensed; its source is not redistributed here.
- **SIMION GA acknowledgment** — D. Manura (2016-12-06), supplied `geneticlib.lua` example, Scientific Instrument Services, Inc., licensed SIMION 8.1. The exact historical notice is preserved in [REFERENCE.md](REFERENCE.md#simion-dependency-and-genetic-algorithm-acknowledgment); the original file is not bundled.
- **Brent** — Richard P. Brent (1973), *Algorithms for Minimization without Derivatives*, Prentice-Hall. [Author's publication page](https://maths-people.anu.edu.au/brent/pub/pub011.html); Dover reprint, 2002. Chapter 4 treats roots and chapter 5 minimization. No unverified DOI is assigned here.
- **Powell** — M. J. D. Powell (2009), *The BOBYQA algorithm for bound constrained optimization without derivatives*, DAMTP NA2009/06. [Author/institution report](https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf). The report host can be inaccessible; this is not evidence that the citation is invalid.
- **Pinned NLopt source** — [BOBYQA source and notices at commit 6e6593f131ba3a38bc9edbed0a357bc01526e54b](https://github.com/stevengj/nlopt/tree/6e6593f131ba3a38bc9edbed0a357bc01526e54b/src/algs/bobyqa). Original C SHA-256: `a1733948888e927a861197b9c8f4a84b561439fbe286d8020e6078e10135d95d`. Preserve the Powell/MIT notice verbatim. This proven software lineage is separate from the independently written solvers.
- **Hansen tutorial** — Nikolaus Hansen (2016), *The CMA Evolution Strategy: A Tutorial*, [author manuscript](https://arxiv.org/abs/1604.00772).
- **BADS** — Luigi Acerbi and Wei Ji Ma (2017), *Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search*, Advances in Neural Information Processing Systems 30. [Author manuscript](https://arxiv.org/abs/1705.04405), DOI:10.48550/arXiv.1705.04405. OptSi's adaptations and omitted features are described in [REFERENCE.md](REFERENCE.md#implementation-notes).
- **GPML** — Carl Edward Rasmussen and Christopher K. I. Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press. [Authors' book site](https://gaussianprocess.org/gpml/). Used for ordinary kriging, Gaussian prediction and model fitting; no GPML runtime or copied package is bundled.
- **logEI** — Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat and Eytan Bakshy (2023), *Unexpected Improvements to Expected Improvement for Bayesian Optimization*, Advances in Neural Information Processing Systems 36. [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/419f72cbd568ad62183f8132a3605a2a-Abstract-Conference.html), DOI:10.52202/075280-0904. OptSi's scalar logEI and analytical log-EHVI calculations do not make it the paper's complete external software implementation.
- **Madsen–Nielsen–Tingleff** — Kaj Madsen, Hans Bruun Nielsen and Ole Tingleff (2004), *Methods for Non-Linear Least Squares Problems*, second edition, Technical University of Denmark. [Institutional catalog](https://www2.imm.dtu.dk/pubdb/pubs/3215-full.html), [author/institution PDF](https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf). No DOI is invented for these lecture notes. OptSi's box policy and stopping/diagnostic contracts differ from MINPACK and SciPy's trust-region reflective solver; see [LM](#lm-signed-nonlinear-least-squares) for the public API and limitations.

The QR reference above is specifically **Businger–Golub (1965),
DOI:10.1007/BF01436084**. Independent numerical reference tests support the
original Lua implementation; they do not establish source equivalence to a
particular LAPACK or SciPy routine. Analytical EHVI follows the
**Emmerich–Deutz–Klinkenberg (2011)** expected-hypervolume framework, specialized
to two independent objectives and validated narrow-strip numerical safeguards.
Its expected improvement criterion differs from smoothed Monte Carlo qLogEHVI.

### Physical example context

- **CPOpt** — Kilian Huber, Tom Wirtz, Hung Quang Hoang (2024). *CPOpt: A modular framework for genetic algorithm optimization and post-optimization analysis in complex charged particle optical design*. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment 1067:169702. [DOI:10.1016/j.nima.2024.169702](https://doi.org/10.1016/j.nima.2024.169702).

The CPOpt reference supplies physical context. The shipped teaching model and
its short budget do not establish a new optical optimum or a complete design
specification. Its preparation and measurements are described above.


Mathematical citations, software redistribution notices and the separately
licensed SIMION runtime are distinct. No DOI has been invented for the Brent
book, Powell report or Madsen–Nielsen–Tingleff notes. The OptSi project-wide
license remains an author decision.
