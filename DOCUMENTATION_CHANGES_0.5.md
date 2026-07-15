# Envelope 0.5 documentation integration notes

The existing authored documentation has intentionally been left unchanged. These are
the changes that still need to be integrated into it for the 0.5 release.

## README

- Replace the claim that wrappers can be stacked in any order with a link to the
  supported wrapper-order guidance.
- Add installation examples for one adapter extra and for the `adapters` union.
- Mention that Gymnax and Kinetix are temporarily source-backed.
- Link to the 0.5 GitHub release notes rather than adding a migration page to the docs.

## Environment API

- State the exact lifecycle signatures: `init(key)`, `reset(state, key)`, and
  `step(state, action)`.
- Explain that `Info` schemas and leaf shapes must be fixed across traced branches.
- Document `supports_init_pooling`, its default of `False`, and why pooled initialization
  rejects stacks that do not propagate the capability.

## Wrapper API

- Document auto-reset's split semantics: reset state/observation at the top level,
  transition reward and flags at the top level, and the complete terminal emission in
  `info.final`.
- Document the zero-like pre-completion placeholder, `final_valid`, and preservation of
  the latest valid final record.
- Distinguish per-episode `stats` from lifetime `cumulative_stats`.
- Add the supported standard stack and pooled stack from the 0.5 contract.
- Add an explicit order matrix covering auto-reset, vectorization, statistics,
  truncation, state injection, normalization, stateless transforms, and pooling.
- Add `CumulativeStatisticsWrapper` to the API reference.

## Adapter API

- Document `max_episode_steps`: `"default"` uses the captured backend horizon, a
  positive integer overrides it, and `None` disables outer truncation.
- Explain that adapters copy caller mappings and parameter objects, disable native
  horizons and auto-reset, and expose stable metadata as `info.backend`.
- Add installation commands for the published extras.
- Add the exact Gymnax and Kinetix source revisions from `pyproject.toml`, together with
  the regression test that protects each pin.
- Note the Brax and Kinetix configuration rejections introduced in 0.5.

## Spaces and struct APIs

- Document strict structure/shape membership and accepted dtype categories.
- Document finite, one-sided, fully unbounded, and mixed-bound continuous sampling.
- Explain canonical `Container` extra-key ordering and fixed schemas in JAX control flow.
- Explain safe static-field validation and the narrow use of
  `static_field(unsafe=True)` for audited opaque objects.

## Release text

Use `RELEASE_NOTES_0.5.md` as the starting point for the GitHub release description. It
contains the user-facing 0.4-to-0.5 migration information without adding a permanent
migration page to the MkDocs navigation.
