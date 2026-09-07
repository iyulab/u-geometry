# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.1.1 onward; earlier entries list release dates only (see git history).

## [Unreleased]

## [0.1.5] - 2026-09-07

### Changed

- **`getrandom` is now 0.4** on WebAssembly targets, reaching the browser entropy
  source through its `wasm_js` crate feature alone; the
  `RUSTFLAGS --cfg getrandom_backend="wasm_js"` that 0.3 required is no longer
  needed.
- **The minimum supported Rust version is now declared as 1.89** and is verified
  by building on that exact toolchain; 1.88 and below fail. The requirement comes
  from `nalgebra` 0.35's `wide`/`safe_arch` chain. The crate previously declared
  no `rust-version` at all, so this makes an existing requirement explicit rather
  than raising one.
- Bump `nalgebra` dependency from 0.33 to 0.35 (dependency freshness sweep).
  No API changes required — `Point`/`Matrix` usage is unaffected across the
  two minor versions.

## [0.1.4] - 2026-08-15

Recorded retroactively: 0.1.4 was published without a changelog entry, and this
one is reconstructed from the release commit (`cedf349`). Contents come from that
commit, not from a contemporary note.

### Added

- `polygons_intersect` — exact overlap test for concave polygons, replacing the
  convex-only approximation for inputs that are not convex.
- WebAssembly bindings for `polygons_intersect`, `polygon_bounds` and
  `transform_points`, bringing the exposed function count to six.

## [0.1.3] - 2026-07-05

### Fixed

- npm: expose the `./package.json` subpath in the `exports` map so tools
  that `require('<pkg>/package.json')` (license scanners, version
  reporters) keep working alongside the conditional exports introduced in
  the previous release (`ERR_PACKAGE_PATH_NOT_EXPORTED`).

## [0.1.2] - 2026-07-05

### Fixed

- **npm packaging — Node-compatible entry.** The npm package previously
  shipped only the wasm-bindgen *bundler*-target output, whose static
  `.wasm` import fails on Node's CJS path (`tsx`/`ts-node` in non-ESM
  packages) with an opaque `SyntaxError: Invalid or unexpected token`.
  The package now additionally ships the *nodejs*-target CJS glue under
  `node/` and routes Node consumers to it via a conditional `exports`
  map (`node` → CJS with filesystem wasm loading, `default` → bundler
  ESM). `require()`, native ESM `import`, and CJS TS runners all work
  without loader hooks. A pre-publish smoke test (CJS `require` + ESM
  `import`) now guards this path in CI. Rust API unchanged.


## [0.1.1] - 2026-06-10

### Changed

- WASM: dropped legacy `*_json` parameter-name suffixes — exported functions
  take native JS objects/arrays, and JSON-string arguments are now rejected
  early with a descriptive error.
- Dependency: `robust` manifest floor aligned to 1.2.

## Earlier releases

- 0.1.0 — 2026-05-05
