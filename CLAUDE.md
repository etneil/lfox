# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent-human collaboration

This file and the top level @README.md file should be treated as __human-only artifacts__; for agent partners, they are read-only and should not be modified.

## Project

`lfox` ("Lattice Fields Over jaX") is a JAX-based library for lattice field theory simulations: Markov-chain field evolution (HMC), lattice operators, and (in progress) fermion solvers / gauge fields. The core abstraction is an `eqx.Module`-backed `Lattice` + `LatticeField` pair that participates in JAX pytrees, supports `jax.jit` / `jax.grad`, and exposes overloaded arithmetic.

## Environment & commands

- Python ≥ 3.12, managed with `uv` (lockfile at `uv.lock`).
- Sync env: `uv sync` (use `uv sync --extra mps` on Apple Silicon for `jax-mps`, experimental).
- Run a script in-env: `uv run python <script>`.
- Run tests: `uv run pytest`. `tests/` pins physics invariants (action normalization, autodiff force vs. analytic force, `<exp(-ΔH)> = 1`, RNG key threading, `Chain` bookkeeping). There is no CI, linter, or formatter configured.

## Coding philosophy

Four core philosophies of developing `lfox`:

1. __Clean and concise.__  This is a library meant for use by a broader audience.  It is meant specifically for use in theoretical physics research.  As a result, it is essential that the code remains _clean_ and _concise_.  Anyone who is using the code should be able to read it and understand how it functions, so they can verify that the physics outputs will be what they expect.
2. __High-level and ergonomic.__  There are other lattice field theory codes written in lower-level compiled languages already on the market, but they mostly require a good amount of specialized knowledge to use properly.  One of the main reasons `lfox` is written in Python is to enable simple, high-level code.  When implementing user-facing functionality, ergonomics is the guiding principle; functions should be simple and natural to use.
3. __Flexible and extensible.__  Another motivation for working in Python is flexibility and extensibility.  `lfox` code should generally be written to be easily modified or specialized.  Our goal is _not_ to be the fastest code for doing lattice QCD at the physical point; instead, we want to be the _easiest_ code to get up and running with in a new theory or with a new calculation.
4. __Test-driven development.__ This is a code for computational physics research, so correctness is extremely important - we want to get science results that are reproducible and trustworthy out of this code.  When developing for `lfox`, follow test-driven development principles: new code must _always_ be accompanied with tests, ideally with the (initially failing) tests being written first.  Code-path tests are fine, but tests for physics/numerical correctness are better.  


## Testing

Three kinds of tests should be designed as part of the development of `lfox`.

1.  __Regression tests.__  These are automated tests which live in `tests/` and are invoked using `pytest`.  These are standard tests targeting behavior of individual parts of the code to make sure it works as expected and specified.  These should be kept __fast__, so that the regression test suite can be run easily as part of any code-development work.
2. __Integration tests.__  These are _physics_ tests which live in `integration/` and are designed to probe whether the simulation outputs produce known/expected results.  They also serve as demonstrations or examples of how to use `lfox`, appropriate for new users.  Instead of using `pytest`, these are designed as Jupyter notebooks, saved to the repository as Python scripts using `jupytext` (which also makes them accessible to agents if needed.)  These may be much more computationally demanding, but should be kept to a level that a new user on a workstation is able to run them in a reasonable amount of time.  These should _not_ be run automatically with all code changes, but only periodically and typically by a human user directly.
3. __Performance tests.__  These are tests of the efficiency and speed of the `lfox` implementation and live in `benchmarks/`.  Like integration tests these are on a slow duty-cycle, and should only be run periodically or when specifically trying to improve implementation details.  These may contain a mix of tests intended for different targets, in particular some at workstation-scale and some at cluster-scale targeting larger, more distributed workflows.  Obviously, tests should only be run on matching hardware, since not all tests will be appropriate for all machines.
