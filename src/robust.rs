//! Numerically robust geometric predicates.
//!
//! Uses Shewchuk's adaptive precision floating-point arithmetic
//! via the `robust` crate to guarantee correct results even for
//! near-degenerate geometric configurations.
//!
//! # Background
//!
//! Standard floating-point arithmetic can produce incorrect results
//! for geometric predicates when points are nearly collinear or cocircular.
//! This module provides robust alternatives.
//!
//! # References
//!
//! - Shewchuk (1997), "Adaptive Precision Floating-Point Arithmetic
//!   and Fast Robust Predicates for Computational Geometry"
//!   <https://www.cs.cmu.edu/~quake/robust.html>

use robust::{orient2d as shewchuk_orient2d, Coord};

/// Result of an orientation test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Orientation {
    /// Points form a counter-clockwise (left) turn.
    CounterClockwise,
    /// Points form a clockwise (right) turn.
    Clockwise,
    /// Points are collinear.
    Collinear,
}

impl Orientation {
    /// Returns true if counter-clockwise.
    #[inline]
    pub fn is_ccw(self) -> bool {
        matches!(self, Self::CounterClockwise)
    }

    /// Returns true if clockwise.
    #[inline]
    pub fn is_cw(self) -> bool {
        matches!(self, Self::Clockwise)
    }

    /// Returns true if collinear.
    #[inline]
    pub fn is_collinear(self) -> bool {
        matches!(self, Self::Collinear)
    }
}

/// Determines the orientation of three points using exact arithmetic.
///
/// Uses Shewchuk's adaptive precision algorithm. The result is
/// mathematically exact regardless of floating-point rounding.
///
/// # Returns
///
/// - `CounterClockwise` if `pc` is to the left of the directed line `pa → pb`
/// - `Clockwise` if to the right
/// - `Collinear` if on the line
///
/// # Complexity
/// O(1) amortized (fast path resolves ~95% of cases)
///
/// # Example
///
/// ```
/// use u_geometry::robust::{orient2d, Orientation};
///
/// let a = (0.0, 0.0);
/// let b = (1.0, 0.0);
/// let c = (0.5, 1.0);
/// assert_eq!(orient2d(a, b, c), Orientation::CounterClockwise);
/// ```
#[inline]
pub fn orient2d(pa: (f64, f64), pb: (f64, f64), pc: (f64, f64)) -> Orientation {
    let result = shewchuk_orient2d(
        Coord { x: pa.0, y: pa.1 },
        Coord { x: pb.0, y: pb.1 },
        Coord { x: pc.0, y: pc.1 },
    );

    if result > 0.0 {
        Orientation::CounterClockwise
    } else if result < 0.0 {
        Orientation::Clockwise
    } else {
        Orientation::Collinear
    }
}

/// Returns the raw orientation determinant (twice the signed area).
///
/// Positive if CCW, negative if CW, zero if collinear.
/// The magnitude equals twice the signed area of the triangle.
#[inline]
pub fn orient2d_raw(pa: (f64, f64), pb: (f64, f64), pc: (f64, f64)) -> f64 {
    shewchuk_orient2d(
        Coord { x: pa.0, y: pa.1 },
        Coord { x: pb.0, y: pb.1 },
        Coord { x: pc.0, y: pc.1 },
    )
}

/// Epsilon for floating-point filter.
const FILTER_EPSILON: f64 = 1e-12;

/// Fast orientation test with exact fallback.
///
/// Uses a floating-point filter: attempts fast approximate calculation
/// first, falls back to exact arithmetic only when the result is
/// ambiguous (~5% of cases).
#[inline]
pub fn orient2d_filtered(pa: (f64, f64), pb: (f64, f64), pc: (f64, f64)) -> Orientation {
    let acx = pa.0 - pc.0;
    let bcx = pb.0 - pc.0;
    let acy = pa.1 - pc.1;
    let bcy = pb.1 - pc.1;

    let det = acx * bcy - acy * bcx;
    let det_sum = (acx * bcy).abs() + (acy * bcx).abs();

    if det.abs() > FILTER_EPSILON * det_sum {
        return if det > 0.0 {
            Orientation::CounterClockwise
        } else {
            Orientation::Clockwise
        };
    }

    orient2d(pa, pb, pc)
}

/// Checks if a point lies strictly inside a triangle.
///
/// Uses robust orientation tests. Returns false for points
/// on the boundary (edges or vertices).
///
/// # Complexity
/// O(1)
pub fn point_in_triangle(p: (f64, f64), a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    let o1 = orient2d(a, b, p);
    let o2 = orient2d(b, c, p);
    let o3 = orient2d(c, a, p);

    (o1 == Orientation::CounterClockwise
        && o2 == Orientation::CounterClockwise
        && o3 == Orientation::CounterClockwise)
        || (o1 == Orientation::Clockwise
            && o2 == Orientation::Clockwise
            && o3 == Orientation::Clockwise)
}

/// Checks if a point lies inside or on the boundary of a triangle.
///
/// # Complexity
/// O(1)
pub fn point_in_triangle_inclusive(
    p: (f64, f64),
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
) -> bool {
    let o1 = orient2d(a, b, p);
    let o2 = orient2d(b, c, p);
    let o3 = orient2d(c, a, p);

    let has_ccw = o1.is_ccw() || o2.is_ccw() || o3.is_ccw();
    let has_cw = o1.is_cw() || o2.is_cw() || o3.is_cw();

    !(has_ccw && has_cw)
}

/// Checks if a polygon is convex using robust orientation tests.
///
/// # Complexity
/// O(n) where n is the number of vertices
pub fn is_convex(polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut expected: Option<Orientation> = None;

    for i in 0..n {
        let p0 = polygon[i];
        let p1 = polygon[(i + 1) % n];
        let p2 = polygon[(i + 2) % n];

        let o = orient2d(p0, p1, p2);
        if o.is_collinear() {
            continue;
        }

        match expected {
            None => expected = Some(o),
            Some(e) if e != o => return false,
            _ => {}
        }
    }

    true
}

/// Checks if a polygon has counter-clockwise winding order.
///
/// Uses the lowest-leftmost vertex (guaranteed to be convex)
/// to determine winding.
///
/// # Complexity
/// O(n)
pub fn is_ccw(polygon: &[(f64, f64)]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let mut min_idx = 0;
    for (i, &(x, y)) in polygon.iter().enumerate() {
        let (mx, my) = polygon[min_idx];
        if y < my || (y == my && x < mx) {
            min_idx = i;
        }
    }

    let n = polygon.len();
    let prev = polygon[(min_idx + n - 1) % n];
    let curr = polygon[min_idx];
    let next = polygon[(min_idx + 1) % n];

    orient2d(prev, curr, next).is_ccw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orient2d_ccw() {
        assert_eq!(
            orient2d((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
            Orientation::CounterClockwise
        );
    }

    #[test]
    fn test_orient2d_cw() {
        assert_eq!(
            orient2d((0.0, 0.0), (0.5, 1.0), (1.0, 0.0)),
            Orientation::Clockwise
        );
    }

    #[test]
    fn test_orient2d_collinear() {
        assert_eq!(
            orient2d((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
            Orientation::Collinear
        );
    }

    #[test]
    fn test_orient2d_filtered_matches_exact() {
        let cases = [
            ((0.0, 0.0), (10.0, 0.0), (5.0, 10.0)),
            ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
            ((0.0, 0.0), (1.0, 0.0), (0.5, -1.0)),
        ];
        for (a, b, c) in &cases {
            assert_eq!(orient2d(*a, *b, *c), orient2d_filtered(*a, *b, *c));
        }
    }

    #[test]
    fn test_point_in_triangle_inside() {
        let a = (0.0, 0.0);
        let b = (10.0, 0.0);
        let c = (5.0, 10.0);
        assert!(point_in_triangle((5.0, 3.0), a, b, c));
    }

    #[test]
    fn test_point_in_triangle_outside() {
        let a = (0.0, 0.0);
        let b = (10.0, 0.0);
        let c = (5.0, 10.0);
        assert!(!point_in_triangle((20.0, 5.0), a, b, c));
    }

    #[test]
    fn test_point_in_triangle_on_edge() {
        let a = (0.0, 0.0);
        let b = (10.0, 0.0);
        let c = (5.0, 10.0);
        // Strict: on edge is NOT inside
        assert!(!point_in_triangle((5.0, 0.0), a, b, c));
    }

    #[test]
    fn test_point_in_triangle_inclusive() {
        let a = (0.0, 0.0);
        let b = (10.0, 0.0);
        let c = (5.0, 10.0);
        assert!(point_in_triangle_inclusive((5.0, 3.0), a, b, c));
        assert!(point_in_triangle_inclusive((5.0, 0.0), a, b, c)); // on edge
        assert!(point_in_triangle_inclusive((0.0, 0.0), a, b, c)); // on vertex
        assert!(!point_in_triangle_inclusive((20.0, 5.0), a, b, c));
    }

    #[test]
    fn test_is_convex_square() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(is_convex(&square));
    }

    #[test]
    fn test_is_convex_l_shape() {
        let l_shape = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (5.0, 5.0),
            (5.0, 10.0),
            (0.0, 10.0),
        ];
        assert!(!is_convex(&l_shape));
    }

    #[test]
    fn test_is_convex_triangle() {
        assert!(is_convex(&[(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]));
    }

    #[test]
    fn test_is_convex_degenerate() {
        assert!(!is_convex(&[(0.0, 0.0), (1.0, 0.0)])); // too few
    }

    #[test]
    fn test_is_ccw_square() {
        assert!(is_ccw(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0)
        ]));
        assert!(!is_ccw(&[
            (0.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (10.0, 0.0)
        ]));
    }

    #[test]
    fn test_orientation_methods() {
        assert!(Orientation::CounterClockwise.is_ccw());
        assert!(!Orientation::CounterClockwise.is_cw());
        assert!(Orientation::Clockwise.is_cw());
        assert!(Orientation::Collinear.is_collinear());
    }

    #[test]
    fn test_orient2d_extreme_coords() {
        // Large coordinates
        assert_eq!(
            orient2d((1e10, 1e10), (1e10 + 1.0, 1e10), (1e10 + 0.5, 1e10 + 1.0)),
            Orientation::CounterClockwise
        );
    }

    #[test]
    fn test_degenerate_triangle() {
        // Collinear points: not inside
        assert!(!point_in_triangle(
            (5.0, 0.0),
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0)
        ));
    }
}
