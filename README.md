# u-geometry

**Domain-agnostic computational geometry library**

[![Crates.io](https://img.shields.io/crates/v/u-geometry.svg)](https://crates.io/crates/u-geometry)
[![docs.rs](https://docs.rs/u-geometry/badge.svg)](https://docs.rs/u-geometry)
[![CI](https://github.com/iyulab/u-geometry/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/u-geometry/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

u-geometry provides fundamental geometric primitives, transformations, polygon operations, collision detection, and spatial indexing. It contains no domain-specific concepts — nesting, packing, routing, etc. are defined by consumers.

## Modules

| Module | Description |
|--------|-------------|
| `primitives` | Core types: `Point2`, `Vector2`, `Segment2`, `AABB2`, `Point3`, `Vector3`, `AABB3` |
| `polygon` | Polygon operations: area, centroid, convex hull, winding order, containment |
| `transform` | Rigid transformations: `Transform2D`, `Transform3D` |
| `robust` | Numerically robust geometric predicates (Shewchuk adaptive precision) |
| `collision` | SAT-based collision detection (2D convex polygons), AABB overlap (2D and 3D) |
| `minkowski` | Minkowski sum and No-Fit Polygon (NFP) for convex polygons |
| `spatial_index` | Linear-scan spatial indices for 2D and 3D AABB queries |
| `nalgebra_types` | Re-exports of commonly used `nalgebra` types for version consistency |

## Features

- **`serde`** — Enable serde serialization for geometric types (includes `nalgebra/serde-serialize`)

## Quick Start

```toml
[dependencies]
u-geometry = { git = "https://github.com/iyulab/u-geometry" }

# with serde support
u-geometry = { git = "https://github.com/iyulab/u-geometry", features = ["serde"] }
```

```rust
use u_geometry::primitives::{Point2, AABB2};
use u_geometry::polygon::Polygon2D;
use u_geometry::collision::sat_overlap;

// Points and AABBs
let p = Point2::new(1.0, 2.0);
let aabb = AABB2::new(Point2::new(0.0, 0.0), Point2::new(10.0, 10.0));
assert!(aabb.contains(&p));

// Polygon operations
let polygon = Polygon2D::new(vec![
    Point2::new(0.0, 0.0),
    Point2::new(4.0, 0.0),
    Point2::new(4.0, 3.0),
    Point2::new(0.0, 3.0),
]);
assert!((polygon.area() - 12.0).abs() < 1e-10);
```

## Build & Test

```bash
cargo build
cargo test
cargo bench  # criterion benchmarks
```

## Academic References

- de Berg, Cheong, van Kreveld, Overmars (2008), *Computational Geometry*
- Shewchuk (1997), *Adaptive Precision Floating-Point Arithmetic*
- O'Rourke (1998), *Computational Geometry in C*
- Ericson (2005), *Real-Time Collision Detection*

## Dependencies

- `nalgebra` 0.33 — Linear algebra
- `robust` 1.1 — Robust geometric predicates
- `serde` 1.0 — Serialization (optional)

## License

MIT License — see [LICENSE](LICENSE).

## npm (WebAssembly)

```bash
npm install @iyulab/u-geometry
```

The package resolves per environment via a conditional `exports` map:

| Environment | Entry |
|---|---|
| Bundlers (webpack, Vite, …) | ESM + WebAssembly ESM-integration (`default` condition) |
| Node.js — `require()`, ESM `import`, CJS TS runners (`tsx`, `ts-node`) | CJS glue loading the wasm from the filesystem (`node` condition) — no loader hooks or flags |

### API

All functions take **native JS objects/arrays** (not JSON strings). A point is
`{ "x": number, "y": number }`; a polygon is an array of points.

| Function | Signature | Returns |
|---|---|---|
| `polygon_area(points)` | `Point[] → number` | Unsigned area of a simple polygon |
| `convex_hull(points)` | `Point[] → Point[]` | Convex hull, CCW order |
| `point_in_polygon(point, polygon)` | `(Point, Point[]) → boolean` | Point inside or on boundary |
| `polygons_intersect(polyA, polyB)` | `(Point[], Point[]) → boolean` | Exact overlap for **convex or concave** polygons |
| `polygon_bounds(points)` | `Point[] → AABB` | Axis-aligned bounding box |
| `transform_points(points, tx, ty, angle)` | `(Point[], number, number, number) → Point[]` | Rigid transform (rotation in **radians** about origin, then translate) |

```js
import init, {
  polygons_intersect, polygon_bounds, transform_points,
} from '@iyulab/u-geometry';
await init();

const a = [{x:0,y:0},{x:2,y:0},{x:2,y:2},{x:0,y:2}];
const b = [{x:1,y:1},{x:3,y:1},{x:3,y:3},{x:1,y:3}];

polygons_intersect(a, b);              // true  — interiors overlap
polygon_bounds(a);                     // { min:{x:0,y:0}, max:{x:2,y:2} }
transform_points(a, 10, 5, Math.PI/2); // a rotated 90° CCW, then translated by (10,5)
```

**Semantics of `polygons_intersect`** — exact for simple polygons, convex or
concave. Polygons that merely abut (share an edge or vertex) are **not**
overlapping, and a part nested inside another's concave notch is correctly
reported as *not* overlapping — unlike a convex-hull (SAT) test. Ideal for
nesting/packing placement self-checks. Holes are not considered.

**JSON schema**

```
Point ::= { "x": number, "y": number }
AABB  ::= { "min": Point, "max": Point }
```

## Related

- [u-numflow](https://github.com/iyulab/u-numflow) — Mathematical primitives
- [u-metaheur](https://github.com/iyulab/u-metaheur) — Metaheuristic optimization (GA, SA, ALNS, CP)
- [u-schedule](https://github.com/iyulab/u-schedule) — Scheduling framework
- [u-nesting](https://github.com/iyulab/U-Nesting) — 2D/3D nesting and bin packing
