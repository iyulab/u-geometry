//! Domain-agnostic computational geometry library.
//!
//! Provides fundamental geometric primitives (2D and 3D), transformations,
//! polygon operations, collision detection, and spatial indexing for the
//! U-Engine ecosystem.
//!
//! # Modules
//!
//! - **`primitives`**: Core types — `Point2`, `Vector2`, `Segment2`, `AABB2`,
//!   `Point3`, `Vector3`, `AABB3`
//! - **`polygon`**: Polygon operations — area, centroid, convex hull, winding
//! - **`transform`**: Rigid transformations — `Transform2D`, `Transform3D`
//! - **`robust`**: Numerically robust geometric predicates (Shewchuk)
//! - **`collision`**: SAT-based collision detection (2D convex polygons), AABB
//!   overlap (2D and 3D)
//! - **`minkowski`**: Minkowski sum and NFP for convex polygons
//! - **`spatial_index`**: Linear-scan spatial indices for 2D and 3D AABB queries
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
//! - Ericson (2005), "Real-Time Collision Detection"

pub mod collision;
pub mod minkowski;
pub mod polygon;
pub mod primitives;
pub mod robust;
pub mod spatial_index;
pub mod transform;

/// Re-exports of nalgebra types commonly used with this crate.
///
/// Consumers can import these directly instead of adding a separate
/// `nalgebra` dependency, ensuring version consistency across the ecosystem.
pub mod nalgebra_types {
    pub use nalgebra::{
        Isometry2, Isometry3, Point2 as NaPoint2, Point3 as NaPoint3, RealField, Rotation2,
        Rotation3, UnitQuaternion, Vector2 as NaVector2, Vector3 as NaVector3,
    };
}
