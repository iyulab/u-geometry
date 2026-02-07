//! 2D rigid transformations (rotation + translation).
//!
//! Built on nalgebra's `Isometry2` for numerical correctness.
//! Provides a simpler API for common transform operations.

use crate::primitives::Point2;
use nalgebra::{Isometry2, Point2 as NaPoint2, Vector2 as NaVector2};

/// A 2D rigid transformation (rotation + translation).
///
/// Internally uses nalgebra `Isometry2` for correct composition
/// and inversion. The representation stores translation (tx, ty)
/// and rotation angle in radians.
///
/// # Example
///
/// ```
/// use u_geometry::transform::Transform2D;
/// use std::f64::consts::PI;
///
/// let t = Transform2D::new(10.0, 20.0, PI / 2.0);
/// let (x, y) = t.apply(1.0, 0.0);
/// assert!((x - 10.0).abs() < 1e-10);
/// assert!((y - 21.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    /// Translation x.
    pub tx: f64,
    /// Translation y.
    pub ty: f64,
    /// Rotation angle in radians.
    pub angle: f64,
}

impl Transform2D {
    /// Identity transform (no rotation, no translation).
    pub fn identity() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            angle: 0.0,
        }
    }

    /// Creates a translation-only transform.
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self { tx, ty, angle: 0.0 }
    }

    /// Creates a rotation-only transform (about the origin).
    pub fn rotation(angle: f64) -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            angle,
        }
    }

    /// Creates a transform with translation and rotation.
    pub fn new(tx: f64, ty: f64, angle: f64) -> Self {
        Self { tx, ty, angle }
    }

    /// Converts to a nalgebra `Isometry2`.
    #[inline]
    pub fn to_isometry(&self) -> Isometry2<f64> {
        Isometry2::new(NaVector2::new(self.tx, self.ty), self.angle)
    }

    /// Creates from a nalgebra `Isometry2`.
    pub fn from_isometry(iso: &Isometry2<f64>) -> Self {
        Self {
            tx: iso.translation.x,
            ty: iso.translation.y,
            angle: iso.rotation.angle(),
        }
    }

    /// Applies this transform to a point.
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let iso = self.to_isometry();
        let p = iso.transform_point(&NaPoint2::new(x, y));
        (p.x, p.y)
    }

    /// Applies this transform to a `Point2`.
    #[inline]
    pub fn apply_point(&self, p: &Point2) -> Point2 {
        let (x, y) = self.apply(p.x, p.y);
        Point2::new(x, y)
    }

    /// Transforms a slice of points.
    pub fn apply_points(&self, points: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let iso = self.to_isometry();
        points
            .iter()
            .map(|(x, y)| {
                let p = iso.transform_point(&NaPoint2::new(*x, *y));
                (p.x, p.y)
            })
            .collect()
    }

    /// Composes two transforms: applies `self` first, then `other`.
    pub fn then(&self, other: &Self) -> Self {
        let iso1 = self.to_isometry();
        let iso2 = other.to_isometry();
        Self::from_isometry(&(iso1 * iso2))
    }

    /// Returns the inverse transform.
    pub fn inverse(&self) -> Self {
        Self::from_isometry(&self.to_isometry().inverse())
    }

    /// Whether this is approximately an identity transform.
    pub fn is_identity(&self, epsilon: f64) -> bool {
        self.tx.abs() < epsilon && self.ty.abs() < epsilon && self.angle.abs() < epsilon
    }
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_identity() {
        let t = Transform2D::identity();
        let (x, y) = t.apply(1.0, 2.0);
        assert!((x - 1.0).abs() < 1e-10);
        assert!((y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_translation() {
        let t = Transform2D::translation(10.0, 20.0);
        let (x, y) = t.apply(1.0, 2.0);
        assert!((x - 11.0).abs() < 1e-10);
        assert!((y - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_90() {
        let t = Transform2D::rotation(PI / 2.0);
        let (x, y) = t.apply(1.0, 0.0);
        assert!((x - 0.0).abs() < 1e-10);
        assert!((y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_180() {
        let t = Transform2D::rotation(PI);
        let (x, y) = t.apply(1.0, 0.0);
        assert!((x - (-1.0)).abs() < 1e-10);
        assert!((y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compose() {
        let t1 = Transform2D::translation(10.0, 0.0);
        let t2 = Transform2D::translation(0.0, 20.0);
        let composed = t1.then(&t2);
        let (x, y) = composed.apply(0.0, 0.0);
        assert!((x - 10.0).abs() < 1e-10);
        assert!((y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse() {
        let t = Transform2D::new(10.0, 20.0, PI / 4.0);
        let inv = t.inverse();
        let composed = t.then(&inv);
        assert!(composed.is_identity(1e-10));
    }

    #[test]
    fn test_apply_point() {
        let t = Transform2D::translation(5.0, 3.0);
        let p = Point2::new(1.0, 2.0);
        let q = t.apply_point(&p);
        assert!((q.x - 6.0).abs() < 1e-10);
        assert!((q.y - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_points() {
        let t = Transform2D::translation(1.0, 1.0);
        let points = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let transformed = t.apply_points(&points);
        assert!((transformed[0].0 - 1.0).abs() < 1e-10);
        assert!((transformed[0].1 - 1.0).abs() < 1e-10);
        assert!((transformed[1].0 - 2.0).abs() < 1e-10);
        assert!((transformed[2].1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_default_is_identity() {
        let t = Transform2D::default();
        assert!(t.is_identity(1e-15));
    }
}
