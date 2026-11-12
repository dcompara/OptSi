# OptSi

OptSi chooses parameters for particle-optics simulations in SIMION from measured
trajectory results. It offers **22 public Lua algorithm names** for scalar
minimization, Pareto search, signed roots and nonlinear least squares.

Use **SIMION 8.1 with Lua 5.1**; native validation used SIMION 8.1.3.9.
SIMION is separately licensed. The Nelder–Mead adapter uses its installed
`simionx.SimplexOptimizer`; the other methods have no external numerical runtime.

## First run: one voltage

1. Extract the **entire OptSi archive**. Keep `Algorithms` and `Examples` as
   sibling directories; example scripts load the library by relative paths.
2. Open `Examples/Einzel_minimal/einzel.iob` in SIMION.
3. In the Particles tab, select that directory's `einzel.fly2` (two rays).
4. Press **Fly'm**. The default `Pattern_Search` adjusts the middle-electrode
   voltage with fixed geometry; watch the console for requests and the result.

The two rays are measured at the detector (`x=80 mm`). The
[trajectory illustration](Examples/Einzel_minimal/trajectories.png) accompanies
this teaching example.
The script makes a separate final display flight after optimization. These
two rays illustrate the interface; they do not establish a full beam optimum.

## Choose the mathematical task

| Task | Feedback from your model | Example choices |
| --- | --- | --- |
| Minimize one cost | One number to minimize | Pattern Search, BOBYQA, EGO |
| Find a Pareto trade-off | A vector of costs to minimize | ParEGO, NSGA-II, MOPSO; EHVI for exactly two |
| Find signed zeros | Signed scalar or square residual vector | Brent, Newton, Broyden |
| Fit signed residuals | Residual vector and explicit scales | LM; its minimum may have nonzero residuals |

Set the exact public name in `algorithm`, for example
`algorithm='Pattern_Search'`. Change names only within a compatible task, with
the selected method's feedback, dimension, bounds and options. The
[reference](REFERENCE.md) lists every name, default and return convention.

## Next: voltage and geometry

In `Examples/Einzel_two_parameters`, copy **one** template **as `einzel.lua`**:

| Template | Task | Default |
| --- | --- | --- |
| `einzel_scalar.lua` | Minimize a scaled combination of position/angular separation | Pattern Search |
| `einzel_pareto.lua` | Minimize the two separations independently | ParEGO |
| `einzel_roots.lua` | Match prescribed signed position/angular targets | Newton |

Open `einzel.iob`, select its two-ray `einzel.fly2`, and press **Fly'm**.
The scripts adjust voltage `V2` and radius `r2`. A radius change rebuilds and
refines the field; a voltage-only change reuses it. They print results and
overwrite `results.txt`. Their small budgets teach the API.

For [signed least squares with LM](REFERENCE.md#lm-signed-nonlinear-least-squares),
use `run()` and `best()`. A least-squares minimum may have nonzero residuals.
The reference explains bounds, residual scales and unresolved-derivative limits.

## Further use

[CPOpt preparation](REFERENCE.md#advanced-cpopt) describes a six-voltage,
fixed-geometry teaching model. Prepare fields in a fresh source copy before
opening its workbench.

Use [ParEGO](REFERENCE.md#2-minimize-several-objectives) for scalarization-based
Bayesian Pareto search, or [EHVI](REFERENCE.md#ehvi-analytical-two-objective-hypervolume-search)
for sequential analytical two-objective hypervolume improvement with a fixed
reference point. See their mathematical scope and limitations before choosing
a method; no universal performance advantage is implied.

[Scientific references](REFERENCE.md#scientific-references-and-implementation-scope)
and [source credits and notices](REFERENCE.md#source-credits-and-redistribution-notices)
are included in the reference. This distribution contains the Lua library and
three SIMION teaching examples. Development tests and research data are retained
separately. A project-wide licensing decision remains pending; existing third-party
notices and the separately licensed SIMION dependency remain in effect.
