//! Rigid transformations for 2D and 3D.
//!
//! Built on nalgebra's `Isometry2`/`Isometry3` for numerical correctness.
//! Provides a simpler API for common transform operations.

use crate::primitives::{Point2, Point3};
use nalgebra::{
    Isometry2, Isometry3, Point2 as NaPoint2, Point3 as NaPoint3, UnitQuaternion,
    Vector2 as NaVector2, Vector3 as NaVector3,
};

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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// A 3D rigid transformation (rotation + translation).
///
/// Internally uses nalgebra `Isometry3` with quaternion rotation
/// for gimbal-lock-free composition and inversion.
/// The representation stores translation (tx, ty, tz) and
/// Euler angles (roll, pitch, yaw) in radians for human readability.
///
/// # Euler Angle Convention
/// - Roll (rx): rotation about X axis
/// - Pitch (ry): rotation about Y axis
/// - Yaw (rz): rotation about Z axis
///
/// Composition order: Rz * Ry * Rx (extrinsic rotations)
///
/// # Reference
/// Diebel (2006), "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors"
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transform3D {
    /// Translation x.
    pub tx: f64,
    /// Translation y.
    pub ty: f64,
    /// Translation z.
    pub tz: f64,
    /// Roll (rotation about X axis) in radians.
    pub rx: f64,
    /// Pitch (rotation about Y axis) in radians.
    pub ry: f64,
    /// Yaw (rotation about Z axis) in radians.
    pub rz: f64,
}

impl Transform3D {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            tz: 0.0,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
        }
    }

    /// Creates a translation-only transform.
    pub fn translation(tx: f64, ty: f64, tz: f64) -> Self {
        Self {
            tx,
            ty,
            tz,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
        }
    }

    /// Creates a transform with translation and Euler angles.
    pub fn new(tx: f64, ty: f64, tz: f64, rx: f64, ry: f64, rz: f64) -> Self {
        Self {
            tx,
            ty,
            tz,
            rx,
            ry,
            rz,
        }
    }

    /// Converts to a nalgebra `Isometry3`.
    ///
    /// Uses the Euler angle convention: Rz * Ry * Rx.
    #[inline]
    pub fn to_isometry(&self) -> Isometry3<f64> {
        let rotation = UnitQuaternion::from_euler_angles(self.rx, self.ry, self.rz);
        Isometry3::from_parts(
            NaVector3::new(self.tx, self.ty, self.tz).into(),
            rotation,
        )
    }

    /// Creates from a nalgebra `Isometry3`.
    pub fn from_isometry(iso: &Isometry3<f64>) -> Self {
        let (rx, ry, rz) = iso.rotation.euler_angles();
        Self {
            tx: iso.translation.x,
            ty: iso.translation.y,
            tz: iso.translation.z,
            rx,
            ry,
            rz,
        }
    }

    /// Applies this transform to coordinates.
    #[inline]
    pub fn apply(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let iso = self.to_isometry();
        let p = iso.transform_point(&NaPoint3::new(x, y, z));
        (p.x, p.y, p.z)
    }

    /// Applies this transform to a `Point3`.
    #[inline]
    pub fn apply_point(&self, p: &Point3) -> Point3 {
        let (x, y, z) = self.apply(p.x, p.y, p.z);
        Point3::new(x, y, z)
    }

    /// Transforms a slice of points.
    pub fn apply_points(&self, points: &[Point3]) -> Vec<Point3> {
        let iso = self.to_isometry();
        points
            .iter()
            .map(|p| {
                let tp = iso.transform_point(&NaPoint3::new(p.x, p.y, p.z));
                Point3::new(tp.x, tp.y, tp.z)
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
        self.tx.abs() < epsilon
            && self.ty.abs() < epsilon
            && self.tz.abs() < epsilon
            && self.rx.abs() < epsilon
            && self.ry.abs() < epsilon
            && self.rz.abs() < epsilon
    }
}

impl Default for Transform3D {
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

    // ======================== 3D Transform Tests ========================

    #[test]
    fn test_3d_identity() {
        let t = Transform3D::identity();
        let (x, y, z) = t.apply(1.0, 2.0, 3.0);
        assert!((x - 1.0).abs() < 1e-10);
        assert!((y - 2.0).abs() < 1e-10);
        assert!((z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_3d_translation() {
        let t = Transform3D::translation(10.0, 20.0, 30.0);
        let (x, y, z) = t.apply(1.0, 2.0, 3.0);
        assert!((x - 11.0).abs() < 1e-10);
        assert!((y - 22.0).abs() < 1e-10);
        assert!((z - 33.0).abs() < 1e-10);
    }

    #[test]
    fn test_3d_rotation_z_90() {
        // Rotate 90 degrees about Z: (1,0,0) → (0,1,0)
        let t = Transform3D::new(0.0, 0.0, 0.0, 0.0, 0.0, PI / 2.0);
        let (x, y, z) = t.apply(1.0, 0.0, 0.0);
        assert!((x - 0.0).abs() < 1e-10);
        assert!((y - 1.0).abs() < 1e-10);
        assert!((z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_3d_rotation_x_90() {
        // Rotate 90 degrees about X: (0,1,0) → (0,0,1)
        let t = Transform3D::new(0.0, 0.0, 0.0, PI / 2.0, 0.0, 0.0);
        let (x, y, z) = t.apply(0.0, 1.0, 0.0);
        assert!((x - 0.0).abs() < 1e-10);
        assert!((y - 0.0).abs() < 1e-10);
        assert!((z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_3d_rotation_y_90() {
        // Rotate 90 degrees about Y: (0,0,1) → (1,0,0)
        let t = Transform3D::new(0.0, 0.0, 0.0, 0.0, PI / 2.0, 0.0);
        let (x, y, z) = t.apply(0.0, 0.0, 1.0);
        assert!((x - 1.0).abs() < 1e-10);
        assert!((y - 0.0).abs() < 1e-10);
        assert!((z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_3d_compose() {
        let t1 = Transform3D::translation(10.0, 0.0, 0.0);
        let t2 = Transform3D::translation(0.0, 20.0, 0.0);
        let composed = t1.then(&t2);
        let (x, y, z) = composed.apply(0.0, 0.0, 0.0);
        assert!((x - 10.0).abs() < 1e-10);
        assert!((y - 20.0).abs() < 1e-10);
        assert!((z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_3d_inverse() {
        let t = Transform3D::new(10.0, 20.0, 30.0, PI / 4.0, PI / 6.0, PI / 3.0);
        let inv = t.inverse();
        let composed = t.then(&inv);
        assert!(composed.is_identity(1e-10));
    }

    #[test]
    fn test_3d_apply_point() {
        let t = Transform3D::translation(5.0, 3.0, 1.0);
        let p = Point3::new(1.0, 2.0, 3.0);
        let q = t.apply_point(&p);
        assert!((q.x - 6.0).abs() < 1e-10);
        assert!((q.y - 5.0).abs() < 1e-10);
        assert!((q.z - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_3d_apply_points() {
        let t = Transform3D::translation(1.0, 1.0, 1.0);
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let transformed = t.apply_points(&points);
        assert!((transformed[0].x - 1.0).abs() < 1e-10);
        assert!((transformed[1].x - 2.0).abs() < 1e-10);
        assert!((transformed[2].y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_3d_default_is_identity() {
        let t = Transform3D::default();
        assert!(t.is_identity(1e-15));
    }

    // ======================== Serde Tests ========================

    #[cfg(feature = "serde")]
    mod serde_tests {
        use super::*;

        #[test]
        fn test_transform2d_roundtrip() {
            let t = Transform2D::new(10.0, 20.0, std::f64::consts::PI / 4.0);
            let json = serde_json::to_string(&t).unwrap();
            let t2: Transform2D = serde_json::from_str(&json).unwrap();
            assert_eq!(t, t2);
        }

        #[test]
        fn test_transform3d_roundtrip() {
            let t = Transform3D::new(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
            let json = serde_json::to_string(&t).unwrap();
            let t2: Transform3D = serde_json::from_str(&json).unwrap();
            assert_eq!(t, t2);
        }
    }
}
