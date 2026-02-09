# u-geometry

**Domain-agnostic computational geometry library**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

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

## Related

- [u-optim](https://github.com/iyulab/u-optim) — Mathematical primitives
- [u-metaheur](https://github.com/iyulab/u-metaheur) — Metaheuristic optimization (GA, SA, ALNS, CP)
- [u-schedule](https://github.com/iyulab/u-schedule) — Scheduling framework
- [u-nesting](https://github.com/iyulab/U-Nesting) — 2D/3D nesting and bin packing
