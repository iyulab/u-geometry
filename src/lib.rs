//! Domain-agnostic computational geometry library.
//!
//! Provides fundamental geometric primitives, transformations, polygon operations,
//! and numerically robust predicates for the U-Engine ecosystem.
//!
//! # Modules
//!
//! - **`primitives`**: Core types — `Point2`, `Vector2`, `Segment2`, `AABB2`
//! - **`polygon`**: Polygon operations — area, centroid, convex hull, winding
//! - **`transform`**: Rigid transformations — `Transform2D` (rotation + translation)
//! - **`robust`**: Numerically robust geometric predicates (Shewchuk)
//! - **`collision`**: SAT-based collision detection for convex polygons
//! - **`minkowski`**: Minkowski sum and NFP for convex polygons
//!
//! # Architecture
//!
//! This crate sits at Layer 2 (Algorithms) in the U-Engine ecosystem.
//! It contains no domain-specific concepts — nesting, packing, scheduling, etc.
//! are all defined by consumers at higher layers.
//!
//! # References
//!
//! - de Berg, Cheong, van Kreveld, Overmars (2008), "Computational Geometry"
//! - Shewchuk (1997), "Adaptive Precision Floating-Point Arithmetic"
//! - O'Rourke (1998), "Computational Geometry in C"

pub mod collision;
pub mod minkowski;
pub mod polygon;
pub mod primitives;
pub mod robust;
pub mod transform;
