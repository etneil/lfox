"""The action functional, and the terms it is built from.

Two classes, with one job each:

  * `Term` is what you write.  Subclass it and supply `density` (or `total`,
    for a term too nonlocal to have a per-site density).  A term is not
    evaluable on its own.
  * `Action` is what you evaluate: a flat sum of terms, built by
    `Action(*terms)`, `+`, or `sum(...)`.  It owns the public API.  Anything
    that consumes an action converts its input with
    `eqx.field(converter=Action)`, so a lone term is never seen downstream.

Two vocabularies meet in this file:

  * Author hooks -- `density`, `total`, `draw`, `gradient` -- take fields as
    ROLES, by name.  A term's roles are read from its hook signature, so
    writing the term is what names its fields.
  * The public API -- `S_field`, `S`, `dS`, `sample` -- takes and returns
    dicts keyed by EXTERNAL field name.  `rebind` maps one onto the other;
    with no rebinding the two coincide.  `Term.rebind` is keyed by role and
    `Action.rebind` by external name, since roles are only unique within one
    term class.

Couplings are ordinary dataclass fields, and therefore pytree leaves: changing
one does not retrace, and `jax.grad` can differentiate through it.  Anything
that steers Python control flow inside a hook (a loop count, a flag), or is not
a number, must be declared `eqx.field(static=True)` instead, at the cost of a
retrace when it changes -- and note that a traced coupling cannot be used in an
`if`.

    class Mass(Term):
        m2: float

        def density(self, phi):
            return 0.5 * self.m2 * phi**2

    S = Mass(m2=0.1) + Hopping(kappa=0.18)
    S.S({'phi': phi})

`S` and `dS` are the interface every evolver depends on.  `sampled_fields` /
`evolved_fields` are a structural declaration: a sampled field has an exact
conditional distribution and must not be given a momentum.  `draw` / `sample`
are an opt-in capability -- a term that can supply an exact heatbath for one of
its own fields says so; HMC-family algorithms use it, others ignore it.
"""

import dataclasses
import inspect
from functools import partial
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# The author hooks, in the order they are introspected.  Adding one here plus a
# reader on whichever derived property consumes it is the whole of the work.
_HOOKS = ("density", "total", "draw", "gradient")

# What a traced coupling's leaves may be.  Tracers count as `jax.Array`.
_NUMERIC = (jax.Array, np.ndarray, np.generic, int, float, complex)


class DuplicateTermError(ValueError):
    # Two terms in one action with the same name on the same fields.  Almost
    # always a name collision; distinct `label`s declare a deliberate duplicate.
    pass


def _hook_roles(cls, hook):
    # Roles of one author hook, or None if this class does not supply it.
    fn = getattr(cls, hook)
    if fn is getattr(Term, hook):
        return None

    params = list(inspect.signature(fn).parameters.values())[1:]  # drop self
    if hook == "draw":
        params = params[1:]  # ...and the key

    variadic = [p.name for p in params if p.kind.name.startswith("VAR_")]
    if variadic:
        raise TypeError(
            f"{cls.__name__}.{hook} may not take *args or **kwargs {variadic}; "
            "a term's roles are read from its signature and must be explicit."
        )

    return tuple(p.name for p in params)


def _require_subset(cls, roles, allowed, what):
    if roles is None:
        return

    extra = [r for r in roles if r not in allowed]
    if extra:
        raise TypeError(
            f"{cls.__name__}.{what} names {extra}, which are not roles of this "
            f"term; its roles are {allowed}."
        )


class Term(eqx.Module):
    # One term of an action.  Subclass to write physics; a subclass that
    # supplies neither `density` nor `total` is an abstract base for shared
    # helpers, and can be subclassed but not instantiated.

    # `label` names a term for `S[label]`, defaulting to the class name.
    # `binding` holds the external names, aligned to the class's roles.  Both
    # are kw_only so that a subclass can declare a coupling with no default.
    label: str = eqx.field(static=True, default="", kw_only=True)
    binding: tuple[str, ...] = eqx.field(static=True, default=(), kw_only=True)

    # Resolved once per subclass in __init_subclass__.  ClassVars, so they are
    # neither dataclass fields nor pytree leaves: free at runtime, and already
    # covered by the class identity in the jit key.  `_roles` stays None on an
    # abstract base.
    _roles: ClassVar[tuple[str, ...] | None] = None
    _hooks: ClassVar[dict] = {hook: None for hook in _HOOKS}

    # Roles this term draws from an exact conditional, declared by the author.
    sampled: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # A term's roles are its density's, or its total's if it is nonlocal
        # enough to have no per-site density.  Every other hook draws on a
        # subset of them.
        hooks = {hook: _hook_roles(cls, hook) for hook in _HOOKS}
        cls._hooks = hooks
        cls._roles = hooks["density"]
        if cls._roles is None:
            cls._roles = hooks["total"]
        if cls._roles is None:
            return

        for hook in ("total", "draw", "gradient"):
            _require_subset(cls, hooks[hook], cls._roles, hook)
        _require_subset(cls, cls.sampled, cls._roles, "sampled")
        if cls.sampled and hooks["draw"] is None:
            raise TypeError(
                f"{cls.__name__} samples {list(cls.sampled)} but defines no draw()."
            )

    def __check_init__(self):
        cls = type(self)
        if cls._roles is None:
            raise TypeError(
                f"{cls.__name__} defines neither density nor total, so it can "
                "be subclassed but not instantiated."
            )

        if self.binding and len(self.binding) != len(cls._roles):
            raise ValueError(
                f"{cls.__name__} has roles {list(cls._roles)}, but binding "
                f"{list(self.binding)} names {len(self.binding)} fields."
            )

        for f in dataclasses.fields(self):
            if f.metadata.get("static", False):
                continue
            leaves = jax.tree.leaves(getattr(self, f.name))
            if not all(isinstance(x, _NUMERIC) for x in leaves):
                raise TypeError(
                    f"{cls.__name__}.{f.name} is not numeric, so it cannot be a "
                    "traced coupling; declare it eqx.field(static=True)."
                )

    # --- author hooks ---------------------------------------------------------
    # Overridden by subclasses, which name their fields as ordinary arguments.

    def density(self, **fields):
        # Per-site action density, as a LatticeField.
        raise NotImplementedError(
            f"{type(self).__name__} has no per-site density; use S() for the total."
        )

    def total(self, **fields):
        # Total action, as a scalar.  The default sums the density, which is
        # what a local term wants; a nonlocal term overrides this instead.
        return jnp.sum(self.density(**fields).F)

    def draw(self, key, **fields):
        # Exact heatbath for this term's `sampled` roles, given the others.
        # Returns {role: LatticeField}.
        raise NotImplementedError

    def gradient(self, **fields):
        # dS/dfield for this term, written by hand instead of differentiated.
        # Returns {role: LatticeField}, with the same sign convention as dS,
        # and must cover every field it is asked to differentiate.
        raise NotImplementedError

    # --- structure ------------------------------------------------------------

    @property
    def name(self):
        # The key `S[...]` looks a term up by.
        return self.label or type(self).__name__

    @property
    def field_names(self):
        # External names, in role order.
        return self.binding or self._roles

    @property
    def param_names(self):
        # The couplings: every dataclass field a subclass declared.
        bookkeeping = {f.name for f in dataclasses.fields(Term)}

        return tuple(
            f.name for f in dataclasses.fields(self) if f.name not in bookkeeping
        )

    def _bind(self):
        # role -> external name.
        return dict(zip(self._roles, self.field_names))

    def _args(self, hook, fields):
        # The slice of `fields` one hook wants, keyed by role.  A term takes
        # its own slice, so extra fields are ignored and a missing one raises.
        roles = self._hooks[hook]
        if roles is None:
            roles = self._roles
        bound = self._bind()

        return {role: fields[bound[role]] for role in roles}

    # --- derived terms --------------------------------------------------------

    def rebind(self, **names):
        # Rename fields.  Keys are always ROLES, never the current external
        # name, so chained rebinds do not make the caller track intermediates.
        unknown = [r for r in names if r not in self._roles]
        if unknown:
            raise ValueError(
                f"{type(self).__name__} has no role(s) {unknown}; its roles are "
                f"{list(self._roles)}."
            )

        binding = tuple(
            names.get(role, current)
            for role, current in zip(self._roles, self.field_names)
        )

        return dataclasses.replace(self, binding=binding)

    def with_param(self, name, value):
        # Retune a coupling, returning a new term.
        if name not in self.param_names:
            raise ValueError(
                f"{type(self).__name__} has no parameter {name!r}; its parameters "
                f"are {list(self.param_names)}."
            )

        return dataclasses.replace(self, **{name: value})

    def replicate(self, n, *roles):
        # The many-flavor loop, written once: n copies of this term with the
        # named roles suffixed _0 .. _(n-1).  Everything else stays shared.
        stem = self._bind()

        return Action(
            *(
                self.rebind(**{r: f"{stem.get(r, r)}_{i}" for r in roles})
                for i in range(n)
            )
        )

    def __add__(self, other):
        if not isinstance(other, (Term, Action)):
            return NotImplemented

        return Action(self, other)

    def __radd__(self, other):
        # So that sum(term(i) for i in range(N)) works, which starts from 0.
        if isinstance(other, int) and other == 0:
            return Action(self)

        return NotImplemented


def _check_duplicates(terms):
    # A term is identified by its name and its fields, and nothing subtler.
    # Traced couplings are deliberately excluded -- two mass terms differing
    # only in m2 are the collision this is meant to catch -- and so the check
    # is safe to run under jax.grad.
    seen = set()
    for term in terms:
        key = (term.name, term.field_names)
        if key in seen:
            raise DuplicateTermError(
                f"{term.name} on fields {list(term.field_names)} is already a term "
                "of this action.  If you meant a second field, rebind one of "
                "them; if you meant both terms, give them distinct labels, e.g. "
                "label='light' / label='heavy'."
            )
        seen.add(key)


class Action(eqx.Module):
    # A flat sum of terms, and the only thing that is evaluated.  Final: build
    # one from terms, never subclass it.  A preset theory is a function that
    # returns an Action.
    terms: tuple[Term, ...]

    def __init__(self, *parts):
        # Accepts terms and actions alike, and flattens, so composition never
        # nests and grouping never matters.  `Action(action)` is a copy, which
        # is what lets `Action` serve as a field converter.
        terms = []
        for part in parts:
            if isinstance(part, Action):
                terms.extend(part.terms)
            elif isinstance(part, Term):
                terms.append(part)
            else:
                raise TypeError(
                    f"an Action is built from terms and actions, not "
                    f"{type(part).__name__}."
                )

        if not terms:
            raise ValueError("an Action needs at least one term.")
        _check_duplicates(terms)

        self.terms = tuple(terms)

    # --- structure ------------------------------------------------------------

    @property
    def field_names(self):
        # External names, in first-appearance order across terms.
        return tuple(
            dict.fromkeys(name for t in self.terms for name in t.field_names)
        )

    @property
    def sampled_fields(self):
        # A field sampled by any term is sampled, even if another term merely
        # reads it: otherwise it would get both a heatbath and a momentum.
        drawn = {t._bind()[role] for t in self.terms for role in t.sampled}

        return tuple(name for name in self.field_names if name in drawn)

    @property
    def evolved_fields(self):
        drawn = set(self.sampled_fields)

        return tuple(name for name in self.field_names if name not in drawn)

    def __getitem__(self, name):
        matches = [t for t in self.terms if t.name == name]
        if len(matches) == 1:
            return matches[0]

        raise KeyError(
            f"{len(matches)} terms named {name!r}; this action has "
            f"{[t.name for t in self.terms]}.  Labels make names unique."
        )

    # --- derived actions ------------------------------------------------------

    def rebind(self, **names):
        # Rename fields.  Keys are EXTERNAL names -- the only names every term
        # agrees on -- so a rename reaches every term on that field whatever
        # its role there.  A permutation is fine; a merge is refused.
        unknown = [n for n in names if n not in self.field_names]
        if unknown:
            raise ValueError(
                f"this action has no field(s) {unknown}; its fields are "
                f"{list(self.field_names)}."
            )

        renamed = [names.get(n, n) for n in self.field_names]
        if len(set(renamed)) != len(renamed):
            raise ValueError(
                f"renaming {names} would merge distinct fields of this action."
            )

        return Action(
            *(
                t.rebind(**{r: names[n] for r, n in t._bind().items() if n in names})
                for t in self.terms
            )
        )

    def with_param(self, name, value, term=None):
        # Retune a coupling, returning a new action.  With `term`, only that
        # term.  Without, every term declaring the coupling -- provided they
        # are all one class: the same name on different kinds of term is two
        # couplings, not one knob.
        if term is not None:
            target = self[term]
            retuned = target.with_param(name, value)

            return Action(*(retuned if t is target else t for t in self.terms))

        declaring = [t for t in self.terms if name in t.param_names]
        if not declaring:
            raise ValueError(
                f"no term of this action has a parameter {name!r}; they have "
                f"{sorted({p for t in self.terms for p in t.param_names})}."
            )

        classes = sorted({type(t).__name__ for t in declaring})
        if len(classes) > 1:
            raise ValueError(
                f"{name!r} is declared by different kinds of term {classes}; "
                "pass term=<name> to retune one of them."
            )

        return Action(
            *(
                t.with_param(name, value) if name in t.param_names else t
                for t in self.terms
            )
        )

    def replicate(self, n, *names):
        # The many-flavor loop over a whole block of terms: n copies with the
        # named fields suffixed _0 .. _(n-1).  Everything else stays shared.
        return Action(
            *(self.rebind(**{f: f"{f}_{i}" for f in names}) for i in range(n))
        )

    def __add__(self, other):
        if not isinstance(other, (Term, Action)):
            return NotImplemented

        return Action(self, other)

    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self

        return NotImplemented

    # --- the public API -------------------------------------------------------

    @jax.jit
    def S_field(self, fields):
        # Action density, as a LatticeField.  Raises if any term is nonlocal.
        density = None
        for term in self.terms:
            contrib = term.density(**term._args("density", fields))
            density = contrib if density is None else density + contrib

        return density

    @jax.jit
    def S(self, fields):
        total = 0.0
        for term in self.terms:
            total = total + term.total(**term._args("total", fields))

        return total

    def dS(self, fields, wrt=None):
        # dS/dfield (the gradient, not the force), keyed by external name.
        # Defaults to the evolved fields: a sampled field is drawn, not moved.
        wrt = self.evolved_fields if wrt is None else tuple(wrt)

        return _grad(self, fields, wrt)

    @jax.jit
    def sample(self, fields, key):
        # Draw every sampled field from its term's exact conditional.  Each
        # term gets its own key: splitting once and reusing it would give every
        # pseudofermion the same noise, which is silent and catastrophic.
        drawing = [term for term in self.terms if term.sampled]
        if not drawing:
            return {}

        drawn = {}
        keys = jax.random.split(key, len(drawing))
        for term, subkey in zip(drawing, keys):
            bound = term._bind()
            for role, field in term.draw(subkey, **term._args("draw", fields)).items():
                if role not in term.sampled:
                    raise ValueError(
                        f"{term.name}.draw returned {role!r}, which it does not "
                        f"declare; it samples {list(term.sampled)}."
                    )
                drawn[bound[role]] = field

        return drawn


@partial(jax.jit, static_argnums=(2,))
def _grad(action, fields, wrt):
    # Differentiate the terms that do not supply a gradient, then add in the
    # ones that do.  A term with an analytic gradient -- a pseudofermion force
    # written by hand rather than differentiated through a solver -- is never
    # differentiated as well, or its contribution would be counted twice.
    differentiated = [t for t in action.terms if t._hooks["gradient"] is None]

    def S_auto(sub):
        merged = fields | sub
        total = 0.0
        for term in differentiated:
            total = total + term.total(**term._args("total", merged))

        return total

    grad = jax.grad(S_auto)({name: fields[name] for name in wrt})

    for term in action.terms:
        if term._hooks["gradient"] is None:
            continue

        # Since this term is never differentiated, a requested field it leaves
        # out would silently get no contribution from it.
        bound = term._bind()
        analytic = term.gradient(**term._args("gradient", fields))
        missing = [n for r, n in bound.items() if n in wrt and r not in analytic]
        if missing:
            raise ValueError(
                f"{term.name}.gradient does not return {missing}, which dS was "
                "asked to differentiate."
            )

        for role, contrib in analytic.items():
            name = bound[role]
            if name in grad:
                grad[name] = grad[name] + contrib

    return grad
