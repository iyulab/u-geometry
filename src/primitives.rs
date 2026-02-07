//! Core geometric primitives.
//!
//! Provides fundamental 2D types used throughout the library.

use std::ops::{Add, Mul, Sub};

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    /// Creates a new point.
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Origin point (0, 0).
    pub const ORIGIN: Self = Self { x: 0.0, y: 0.0 };

    /// Euclidean distance to another point.
    ///
    /// # Complexity
    /// O(1)
    #[inline]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Squared Euclidean distance (avoids sqrt).
    #[inline]
    pub fn distance_sq(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Converts to a tuple.
    #[inline]
    pub fn to_tuple(self) -> (f64, f64) {
        (self.x, self.y)
    }

    /// Creates from a tuple.
    #[inline]
    pub fn from_tuple(t: (f64, f64)) -> Self {
        Self { x: t.0, y: t.1 }
    }
}

impl From<(f64, f64)> for Point2 {
    fn from(t: (f64, f64)) -> Self {
        Self::from_tuple(t)
    }
}

impl From<Point2> for (f64, f64) {
    fn from(p: Point2) -> Self {
        p.to_tuple()
    }
}

impl Sub for Point2 {
    type Output = Vector2;

    fn sub(self, rhs: Self) -> Vector2 {
        Vector2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Add<Vector2> for Point2 {
    type Output = Point2;

    fn add(self, rhs: Vector2) -> Point2 {
        Point2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

/// A 2D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2 {
    pub x: f64,
    pub y: f64,
}

impl Vector2 {
    /// Creates a new vector.
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Zero vector.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Euclidean length.
    #[inline]
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Squared length (avoids sqrt).
    #[inline]
    pub fn length_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Returns a normalized (unit-length) vector, or zero vector if length is ~0.
    #[inline]
    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-15 {
            Self::ZERO
        } else {
            Self::new(self.x / len, self.y / len)
        }
    }

    /// 2D cross product (z-component of the 3D cross product).
    ///
    /// Returns a positive value if `other` is counter-clockwise from `self`,
    /// negative if clockwise, zero if parallel.
    #[inline]
    pub fn cross(&self, other: &Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Dot product.
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Returns the perpendicular vector (rotated 90 degrees CCW).
    #[inline]
    pub fn perp(&self) -> Self {
        Self::new(-self.y, self.x)
    }
}

impl Add for Vector2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vector2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<f64> for Vector2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

/// A 2D line segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment2 {
    pub start: Point2,
    pub end: Point2,
}

impl Segment2 {
    /// Creates a new segment.
    pub fn new(start: Point2, end: Point2) -> Self {
        Self { start, end }
    }

    /// Segment length.
    #[inline]
    pub fn length(&self) -> f64 {
        self.start.distance_to(&self.end)
    }

    /// Squared length.
    #[inline]
    pub fn length_sq(&self) -> f64 {
        self.start.distance_sq(&self.end)
    }

    /// Direction vector (not normalized).
    #[inline]
    pub fn direction(&self) -> Vector2 {
        self.end - self.start
    }

    /// Midpoint.
    #[inline]
    pub fn midpoint(&self) -> Point2 {
        Point2::new(
            (self.start.x + self.end.x) * 0.5,
            (self.start.y + self.end.y) * 0.5,
        )
    }
}

/// A 2D axis-aligned bounding box.
///
/// # Invariant
/// `min.x <= max.x` and `min.y <= max.y`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB2 {
    /// Minimum corner.
    pub min: Point2,
    /// Maximum corner.
    pub max: Point2,
}

impl AABB2 {
    /// Creates an AABB from min/max corners.
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min: Point2::new(min_x, min_y),
            max: Point2::new(max_x, max_y),
        }
    }

    /// Creates an AABB enclosing a set of points.
    ///
    /// Returns `None` if the slice is empty.
    ///
    /// # Complexity
    /// O(n)
    pub fn from_points(points: &[Point2]) -> Option<Self> {
        let first = points.first()?;
        let mut min_x = first.x;
        let mut min_y = first.y;
        let mut max_x = first.x;
        let mut max_y = first.y;

        for p in points.iter().skip(1) {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        Some(Self::new(min_x, min_y, max_x, max_y))
    }

    /// Width (x extent).
    #[inline]
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }

    /// Height (y extent).
    #[inline]
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }

    /// Area.
    #[inline]
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Center point.
    #[inline]
    pub fn center(&self) -> Point2 {
        Point2::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
        )
    }

    /// Whether this AABB contains a point.
    #[inline]
    pub fn contains_point(&self, p: &Point2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// Whether two AABBs intersect.
    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Returns the intersection of two AABBs, or `None` if they don't overlap.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if !self.intersects(other) {
            return None;
        }
        Some(Self::new(
            self.min.x.max(other.min.x),
            self.min.y.max(other.min.y),
            self.max.x.min(other.max.x),
            self.max.y.min(other.max.y),
        ))
    }

    /// Returns the union (bounding box) of two AABBs.
    pub fn union(&self, other: &Self) -> Self {
        Self::new(
            self.min.x.min(other.min.x),
            self.min.y.min(other.min.y),
            self.max.x.max(other.max.x),
            self.max.y.max(other.max.y),
        )
    }

    /// Returns a new AABB expanded by `margin` on all sides.
    pub fn expand(&self, margin: f64) -> Self {
        Self::new(
            self.min.x - margin,
            self.min.y - margin,
            self.max.x + margin,
            self.max.y + margin,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2_distance() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point2_sub_gives_vector() {
        let a = Point2::new(3.0, 5.0);
        let b = Point2::new(1.0, 2.0);
        let v = a - b;
        assert!((v.x - 2.0).abs() < 1e-10);
        assert!((v.y - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_point2_add_vector() {
        let p = Point2::new(1.0, 2.0);
        let v = Vector2::new(3.0, 4.0);
        let q = p + v;
        assert!((q.x - 4.0).abs() < 1e-10);
        assert!((q.y - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_operations() {
        let v = Vector2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-10);
        assert!((v.length_sq() - 25.0).abs() < 1e-10);

        let n = v.normalized();
        assert!((n.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_cross() {
        let a = Vector2::new(1.0, 0.0);
        let b = Vector2::new(0.0, 1.0);
        assert!((a.cross(&b) - 1.0).abs() < 1e-10);
        assert!((b.cross(&a) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_dot() {
        let a = Vector2::new(1.0, 2.0);
        let b = Vector2::new(3.0, 4.0);
        assert!((a.dot(&b) - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_perp() {
        let v = Vector2::new(1.0, 0.0);
        let p = v.perp();
        assert!((p.x - 0.0).abs() < 1e-10);
        assert!((p.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_normalized_zero() {
        let v = Vector2::ZERO;
        let n = v.normalized();
        assert!((n.length()).abs() < 1e-10);
    }

    #[test]
    fn test_segment2() {
        let s = Segment2::new(Point2::new(0.0, 0.0), Point2::new(3.0, 4.0));
        assert!((s.length() - 5.0).abs() < 1e-10);
        let mid = s.midpoint();
        assert!((mid.x - 1.5).abs() < 1e-10);
        assert!((mid.y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb2_from_points() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 5.0),
            Point2::new(3.0, 8.0),
        ];
        let aabb = AABB2::from_points(&points).unwrap();
        assert!((aabb.min.x - 0.0).abs() < 1e-10);
        assert!((aabb.min.y - 0.0).abs() < 1e-10);
        assert!((aabb.max.x - 10.0).abs() < 1e-10);
        assert!((aabb.max.y - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb2_from_points_empty() {
        assert!(AABB2::from_points(&[]).is_none());
    }

    #[test]
    fn test_aabb2_area() {
        let aabb = AABB2::new(0.0, 0.0, 10.0, 20.0);
        assert!((aabb.area() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb2_contains() {
        let aabb = AABB2::new(0.0, 0.0, 10.0, 10.0);
        assert!(aabb.contains_point(&Point2::new(5.0, 5.0)));
        assert!(aabb.contains_point(&Point2::new(0.0, 0.0))); // on boundary
        assert!(!aabb.contains_point(&Point2::new(11.0, 5.0)));
    }

    #[test]
    fn test_aabb2_intersection() {
        let a = AABB2::new(0.0, 0.0, 10.0, 10.0);
        let b = AABB2::new(5.0, 5.0, 15.0, 15.0);
        let int = a.intersection(&b).unwrap();
        assert!((int.min.x - 5.0).abs() < 1e-10);
        assert!((int.min.y - 5.0).abs() < 1e-10);
        assert!((int.max.x - 10.0).abs() < 1e-10);
        assert!((int.max.y - 10.0).abs() < 1e-10);

        let c = AABB2::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersection(&c).is_none());
    }

    #[test]
    fn test_aabb2_union() {
        let a = AABB2::new(0.0, 0.0, 10.0, 10.0);
        let b = AABB2::new(5.0, 5.0, 15.0, 15.0);
        let u = a.union(&b);
        assert!((u.min.x - 0.0).abs() < 1e-10);
        assert!((u.min.y - 0.0).abs() < 1e-10);
        assert!((u.max.x - 15.0).abs() < 1e-10);
        assert!((u.max.y - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb2_expand() {
        let aabb = AABB2::new(5.0, 5.0, 10.0, 10.0);
        let expanded = aabb.expand(1.0);
        assert!((expanded.min.x - 4.0).abs() < 1e-10);
        assert!((expanded.min.y - 4.0).abs() < 1e-10);
        assert!((expanded.max.x - 11.0).abs() < 1e-10);
        assert!((expanded.max.y - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_point2_from_tuple() {
        let p: Point2 = (3.0, 4.0).into();
        assert!((p.x - 3.0).abs() < 1e-10);
        assert!((p.y - 4.0).abs() < 1e-10);

        let t: (f64, f64) = p.into();
        assert!((t.0 - 3.0).abs() < 1e-10);
    }
}
