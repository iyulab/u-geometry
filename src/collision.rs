//! Collision detection for 2D polygons.
//!
//! # Algorithms
//!
//! - **SAT (Separating Axis Theorem)**: Exact overlap test for convex polygons.
//!   For concave polygons, it tests the convex hull approximation.
//! - **AABB broad phase**: Fast rejection using bounding boxes.
//!
//! # References
//!
//! - Ericson (2005), "Real-Time Collision Detection", Ch. 4.4
//! - Gottschalk, Lin, Manocha (1996), "OBB-Tree: A Hierarchical Structure
//!   for Rapid Interference Detection"

use crate::primitives::{AABB2, AABB3};

/// Checks if two convex polygons overlap using the Separating Axis Theorem.
///
/// Returns `true` if the polygons overlap (not just touching).
/// Uses a tolerance to allow touching without reporting overlap.
///
/// For concave polygons, this tests the convex hull — it may produce
/// false positives but never false negatives.
///
/// # Complexity
/// O(n + m) where n, m are vertex counts
///
/// # Reference
/// Ericson (2005), Real-Time Collision Detection, Ch. 4.4
pub fn polygons_overlap(poly_a: &[(f64, f64)], poly_b: &[(f64, f64)]) -> bool {
    polygons_overlap_with_tolerance(poly_a, poly_b, 1e-6)
}

/// SAT overlap test with configurable tolerance.
///
/// Returns `true` if the polygons overlap by more than `tolerance`.
pub fn polygons_overlap_with_tolerance(
    poly_a: &[(f64, f64)],
    poly_b: &[(f64, f64)],
    tolerance: f64,
) -> bool {
    if poly_a.len() < 3 || poly_b.len() < 3 {
        return false;
    }

    // Broad phase: AABB check
    if let (Some(aabb_a), Some(aabb_b)) = (aabb_from_tuples(poly_a), aabb_from_tuples(poly_b)) {
        let expanded_a = aabb_a.expand(tolerance);
        if !expanded_a.intersects(&aabb_b) {
            return false;
        }
    }

    // SAT: test edges from both polygons
    for polygon in [poly_a, poly_b] {
        let n = polygon.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let edge_x = polygon[j].0 - polygon[i].0;
            let edge_y = polygon[j].1 - polygon[i].1;

            // Axis = normal to edge (perpendicular)
            let len = (edge_x * edge_x + edge_y * edge_y).sqrt();
            if len < 1e-15 {
                continue;
            }
            let axis = (-edge_y / len, edge_x / len);

            // Project both polygons onto axis
            let (min_a, max_a) = project_on_axis(poly_a, axis);
            let (min_b, max_b) = project_on_axis(poly_b, axis);

            // Check for separation gap.
            // Overlap on this axis = min(max_a, max_b) - max(min_a, min_b)
            // If overlap <= tolerance, treat as separated (allows touching).
            let overlap = max_a.min(max_b) - min_a.max(min_b);
            if overlap < tolerance {
                return false; // Separating axis found → no overlap
            }
        }
    }

    true // No separating axis found → overlap
}

/// Computes the overlap depth (penetration) between two convex polygons.
///
/// Returns the minimum translation distance to separate the polygons,
/// or `None` if they don't overlap.
///
/// # Complexity
/// O(n + m)
pub fn overlap_depth(poly_a: &[(f64, f64)], poly_b: &[(f64, f64)]) -> Option<f64> {
    if poly_a.len() < 3 || poly_b.len() < 3 {
        return None;
    }

    let mut min_depth = f64::INFINITY;

    for polygon in [poly_a, poly_b] {
        let n = polygon.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let edge_x = polygon[j].0 - polygon[i].0;
            let edge_y = polygon[j].1 - polygon[i].1;

            let len = (edge_x * edge_x + edge_y * edge_y).sqrt();
            if len < 1e-15 {
                continue;
            }
            let axis = (-edge_y / len, edge_x / len);

            let (min_a, max_a) = project_on_axis(poly_a, axis);
            let (min_b, max_b) = project_on_axis(poly_b, axis);

            let overlap = (max_a.min(max_b) - min_a.max(min_b)).max(0.0);
            if overlap <= 0.0 {
                return None; // Separated
            }
            min_depth = min_depth.min(overlap);
        }
    }

    if min_depth < f64::INFINITY {
        Some(min_depth)
    } else {
        None
    }
}

/// Checks if two AABBs overlap (broad-phase test).
///
/// # Complexity
/// O(1)
#[inline]
pub fn aabb_overlap(a: &AABB2, b: &AABB2) -> bool {
    a.intersects(b)
}

/// Projects a polygon onto an axis and returns (min, max) extent.
#[inline]
fn project_on_axis(polygon: &[(f64, f64)], axis: (f64, f64)) -> (f64, f64) {
    let mut min_proj = f64::INFINITY;
    let mut max_proj = f64::NEG_INFINITY;

    for &(x, y) in polygon {
        let proj = x * axis.0 + y * axis.1;
        min_proj = min_proj.min(proj);
        max_proj = max_proj.max(proj);
    }

    (min_proj, max_proj)
}

// ======================== 3D Collision ========================

/// Checks if two 3D AABBs overlap.
///
/// # Complexity
/// O(1)
#[inline]
pub fn aabb3_overlap(a: &AABB3, b: &AABB3) -> bool {
    a.intersects(b)
}

/// Checks if two 3D AABBs overlap with a tolerance margin.
///
/// Returns `true` if the boxes overlap by more than `tolerance` on all axes.
/// This allows touching (overlap ≤ tolerance) without reporting collision.
///
/// # Complexity
/// O(1)
pub fn aabb3_overlap_with_tolerance(a: &AABB3, b: &AABB3, tolerance: f64) -> bool {
    let overlap_x = a.max.x.min(b.max.x) - a.min.x.max(b.min.x);
    let overlap_y = a.max.y.min(b.max.y) - a.min.y.max(b.min.y);
    let overlap_z = a.max.z.min(b.max.z) - a.min.z.max(b.min.z);
    overlap_x > tolerance && overlap_y > tolerance && overlap_z > tolerance
}

/// Checks if a 3D AABB is fully contained within a boundary AABB.
///
/// Returns `true` if `inner` fits completely inside `boundary`.
///
/// # Complexity
/// O(1)
#[inline]
pub fn aabb3_within(inner: &AABB3, boundary: &AABB3) -> bool {
    boundary.contains(inner)
}

/// Checks if a 3D AABB fits within a boundary with a margin.
///
/// Returns `true` if `inner` fits inside `boundary` shrunk by `margin`.
///
/// # Complexity
/// O(1)
pub fn aabb3_within_with_margin(inner: &AABB3, boundary: &AABB3, margin: f64) -> bool {
    inner.min.x >= boundary.min.x + margin
        && inner.min.y >= boundary.min.y + margin
        && inner.min.z >= boundary.min.z + margin
        && inner.max.x <= boundary.max.x - margin
        && inner.max.y <= boundary.max.y - margin
        && inner.max.z <= boundary.max.z - margin
}

/// Computes AABB from tuple points.
fn aabb_from_tuples(points: &[(f64, f64)]) -> Option<AABB2> {
    let first = points.first()?;
    let mut min_x = first.0;
    let mut min_y = first.1;
    let mut max_x = first.0;
    let mut max_y = first.1;

    for &(x, y) in points.iter().skip(1) {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    Some(AABB2::new(min_x, min_y, max_x, max_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64, size: f64) -> Vec<(f64, f64)> {
        vec![(x, y), (x + size, y), (x + size, y + size), (x, y + size)]
    }

    fn triangle(x: f64, y: f64, size: f64) -> Vec<(f64, f64)> {
        vec![(x, y), (x + size, y), (x + size / 2.0, y + size)]
    }

    #[test]
    fn test_overlapping_squares() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(5.0, 5.0, 10.0);
        assert!(polygons_overlap(&a, &b));
    }

    #[test]
    fn test_separated_squares() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(20.0, 0.0, 10.0);
        assert!(!polygons_overlap(&a, &b));
    }

    #[test]
    fn test_touching_squares() {
        // Touching = not overlapping (within tolerance)
        let a = square(0.0, 0.0, 10.0);
        let b = square(10.0, 0.0, 10.0);
        assert!(!polygons_overlap(&a, &b));
    }

    #[test]
    fn test_contained_square() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(2.0, 2.0, 3.0);
        assert!(polygons_overlap(&a, &b));
    }

    #[test]
    fn test_triangle_overlap() {
        let a = triangle(0.0, 0.0, 10.0);
        let b = triangle(5.0, 0.0, 10.0);
        assert!(polygons_overlap(&a, &b));
    }

    #[test]
    fn test_triangle_no_overlap() {
        let a = triangle(0.0, 0.0, 10.0);
        let b = triangle(20.0, 0.0, 10.0);
        assert!(!polygons_overlap(&a, &b));
    }

    #[test]
    fn test_degenerate_polygons() {
        let a = vec![(0.0, 0.0), (1.0, 0.0)]; // Not a polygon
        let b = square(0.0, 0.0, 10.0);
        assert!(!polygons_overlap(&a, &b));
    }

    #[test]
    fn test_tolerance_effect() {
        let a = square(0.0, 0.0, 10.0);
        // Slightly overlapping by 0.5 units
        let b = square(9.5, 0.0, 10.0);
        // With default tolerance 1e-6, should overlap
        assert!(polygons_overlap(&a, &b));
        // With large tolerance, should NOT overlap
        assert!(!polygons_overlap_with_tolerance(&a, &b, 1.0));
    }

    #[test]
    fn test_overlap_depth_overlapping() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(7.0, 0.0, 10.0);
        let depth = overlap_depth(&a, &b);
        assert!(depth.is_some());
        assert!((depth.unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_overlap_depth_separated() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(20.0, 0.0, 10.0);
        assert!(overlap_depth(&a, &b).is_none());
    }

    #[test]
    fn test_aabb_overlap() {
        let a = AABB2::new(0.0, 0.0, 10.0, 10.0);
        let b = AABB2::new(5.0, 5.0, 15.0, 15.0);
        assert!(aabb_overlap(&a, &b));

        let c = AABB2::new(20.0, 20.0, 30.0, 30.0);
        assert!(!aabb_overlap(&a, &c));
    }

    // ======================== 3D Collision Tests ========================

    fn box3d(x: f64, y: f64, z: f64, w: f64, d: f64, h: f64) -> AABB3 {
        AABB3::new(x, y, z, x + w, y + d, z + h)
    }

    #[test]
    fn test_aabb3_overlap_intersecting() {
        let a = box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        let b = box3d(5.0, 5.0, 5.0, 10.0, 10.0, 10.0);
        assert!(aabb3_overlap(&a, &b));
    }

    #[test]
    fn test_aabb3_overlap_separated() {
        let a = box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        let b = box3d(20.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        assert!(!aabb3_overlap(&a, &b));
    }

    #[test]
    fn test_aabb3_overlap_touching() {
        // Touching on one face — intersects returns true (boundary overlap)
        let a = box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        let b = box3d(10.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        assert!(aabb3_overlap(&a, &b));
    }

    #[test]
    fn test_aabb3_overlap_with_tolerance() {
        let a = box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        let b = box3d(10.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        // Touching: overlap = 0 on x-axis, not > tolerance
        assert!(!aabb3_overlap_with_tolerance(&a, &b, 1e-6));

        // Slightly overlapping
        let c = box3d(9.5, 0.0, 0.0, 10.0, 10.0, 10.0);
        assert!(aabb3_overlap_with_tolerance(&a, &c, 1e-6));
        // With large tolerance, not enough overlap
        assert!(!aabb3_overlap_with_tolerance(&a, &c, 1.0));
    }

    #[test]
    fn test_aabb3_within() {
        let outer = box3d(0.0, 0.0, 0.0, 20.0, 20.0, 20.0);
        let inner = box3d(5.0, 5.0, 5.0, 5.0, 5.0, 5.0);
        assert!(aabb3_within(&inner, &outer));
        assert!(!aabb3_within(&outer, &inner));
    }

    #[test]
    fn test_aabb3_within_with_margin() {
        let boundary = box3d(0.0, 0.0, 0.0, 20.0, 20.0, 20.0);
        let inner = box3d(2.0, 2.0, 2.0, 5.0, 5.0, 5.0);
        assert!(aabb3_within_with_margin(&inner, &boundary, 1.0));
        // With larger margin, inner is too close to boundary
        assert!(!aabb3_within_with_margin(&inner, &boundary, 3.0));
    }

    #[test]
    fn test_aabb3_overlap_z_separated() {
        // Overlap in x,y but separated in z
        let a = box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        let b = box3d(5.0, 5.0, 20.0, 10.0, 10.0, 10.0);
        assert!(!aabb3_overlap(&a, &b));
    }
}
