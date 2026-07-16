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

use crate::polygon::contains_point;
use crate::primitives::{AABB2, AABB3};
use crate::robust::orient2d;

/// Checks if two convex polygons overlap using the Separating Axis Theorem.
///
/// Returns `true` if the polygons overlap (not just touching).
/// Uses a tolerance to allow touching without reporting overlap.
///
/// For concave polygons, this tests the convex hull — it may produce
/// false positives but never false negatives. When exact overlap on
/// **concave** simple polygons is required (e.g. nesting/packing checks
/// where a part nested inside another's notch must *not* be reported),
/// use [`polygons_intersect`] instead.
///
/// # Complexity
/// O(n + m) where n, m are vertex counts
///
/// # Reference
/// Ericson (2005), Real-Time Collision Detection, Ch. 4.4
pub fn polygons_overlap(poly_a: &[(f64, f64)], poly_b: &[(f64, f64)]) -> bool {
    polygons_overlap_with_tolerance(poly_a, poly_b, 1e-6)
}

/// Exact overlap test for two **simple polygons**, convex *or concave*.
///
/// Unlike [`polygons_overlap`] (SAT), which tests each input's *convex hull*
/// and therefore over-reports for concave shapes, this predicate respects the
/// true boundaries. It reports `true` iff the polygons' boundaries properly
/// cross **or** one polygon's interior overlaps the other's.
///
/// Touching without penetrating — shared edges or vertices — is **not**
/// overlap: edge crossings use a strict (proper) intersection test, and the
/// interior test samples points with strict-interior (open) point location, so
/// a boundary-only contact is excluded. This matches the semantics required by
/// nesting/packing self-checks, where parts that merely abut must not be flagged.
///
/// The interior test samples each polygon's vertices *and edge midpoints*; the
/// midpoints catch area overlaps along collinear edges that no vertex reveals
/// (e.g. two axis-aligned rectangles overlapping in a strip). Holes are not
/// considered (outer boundaries only).
///
/// # Limitation
/// Interior sampling is finite: a pathological overlap that contains no sampled
/// point and crosses no edge transversally could be missed. For the polygons
/// that arise in nesting/packing this does not occur; an exact clipping-based
/// area intersection would remove the theoretical gap at higher cost.
///
/// # Complexity
/// O(n · m) where n, m are vertex counts (broad-phase AABB reject first).
///
/// # Reference
/// O'Rourke (1998), "Computational Geometry in C", Ch. 7 — segment
/// intersection and point-in-polygon.
pub fn polygons_intersect(poly_a: &[(f64, f64)], poly_b: &[(f64, f64)]) -> bool {
    if poly_a.len() < 3 || poly_b.len() < 3 {
        return false;
    }

    // Broad phase: disjoint bounding boxes cannot overlap.
    if let (Some(aabb_a), Some(aabb_b)) = (aabb_from_tuples(poly_a), aabb_from_tuples(poly_b)) {
        if !aabb_a.intersects(&aabb_b) {
            return false;
        }
    }

    // Narrow phase 1: any pair of edges properly crosses → overlap.
    let n = poly_a.len();
    let m = poly_b.len();
    for i in 0..n {
        let a0 = poly_a[i];
        let a1 = poly_a[(i + 1) % n];
        for j in 0..m {
            let b0 = poly_b[j];
            let b1 = poly_b[(j + 1) % m];
            if segments_properly_intersect(a0, a1, b0, b1) {
                return true;
            }
        }
    }

    // Narrow phase 2: no boundary crossing → the polygons are either disjoint or
    // one region lies inside the other. A strict-interior sample point of one
    // polygon inside the other proves overlap; a merely-touching point lies on
    // the boundary and is excluded. Sampling vertices *and* edge midpoints lets
    // this catch collinear-edge strip overlaps that no vertex would reveal.
    if any_interior_sample_inside(poly_a, poly_b) || any_interior_sample_inside(poly_b, poly_a) {
        return true;
    }

    false
}

/// Returns `true` if any vertex or edge-midpoint of `probe` lies strictly
/// inside `target` (open interior, boundary excluded).
fn any_interior_sample_inside(probe: &[(f64, f64)], target: &[(f64, f64)]) -> bool {
    let n = probe.len();
    for i in 0..n {
        let a = probe[i];
        if strictly_inside(a, target) {
            return true;
        }
        let b = probe[(i + 1) % n];
        let mid = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
        if strictly_inside(mid, target) {
            return true;
        }
    }
    false
}

/// Proper (strict) segment intersection: `true` only when the two open segments
/// cross transversally. Shared endpoints and collinear overlaps do **not**
/// count. Uses exact orientation predicates (Shewchuk) for robustness.
fn segments_properly_intersect(
    a0: (f64, f64),
    a1: (f64, f64),
    b0: (f64, f64),
    b1: (f64, f64),
) -> bool {
    let d1 = orient2d(b0, b1, a0);
    let d2 = orient2d(b0, b1, a1);
    let d3 = orient2d(a0, a1, b0);
    let d4 = orient2d(a0, a1, b1);

    let a_straddles_b = (d1.is_ccw() && d2.is_cw()) || (d1.is_cw() && d2.is_ccw());
    let b_straddles_a = (d3.is_ccw() && d4.is_cw()) || (d3.is_cw() && d4.is_ccw());
    a_straddles_b && b_straddles_a
}

/// Tests whether `point` lies in the **open interior** of a simple polygon —
/// strictly inside, excluding the boundary. Combines the boundary-inclusive
/// winding test with an explicit on-boundary rejection.
fn strictly_inside(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    contains_point(polygon, point) && !point_on_boundary(point, polygon)
}

/// Tests whether `point` lies on any edge of the polygon (collinear with the
/// edge and within its extent).
fn point_on_boundary(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        if orient2d(a, b, point).is_collinear()
            && point.0 >= a.0.min(b.0)
            && point.0 <= a.0.max(b.0)
            && point.1 >= a.1.min(b.1)
            && point.1 <= a.1.max(b.1)
        {
            return true;
        }
    }
    false
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

    // ============ Exact (concave-correct) polygon intersection ============

    /// L-shaped concave polygon: bottom row (y 0–1) plus left column (x 0–1),
    /// leaving a concave notch over x∈(1,3), y∈(1,3).
    fn l_shape() -> Vec<(f64, f64)> {
        vec![
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
        ]
    }

    #[test]
    fn test_intersect_crossing_squares() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(5.0, 5.0, 10.0);
        assert!(polygons_intersect(&a, &b));
    }

    #[test]
    fn test_intersect_separated() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(20.0, 0.0, 10.0);
        assert!(!polygons_intersect(&a, &b));
    }

    #[test]
    fn test_intersect_abutting_shared_edge_is_not_overlap() {
        // Squares sharing the edge x=10 abut but do not penetrate.
        let a = square(0.0, 0.0, 10.0);
        let b = square(10.0, 0.0, 10.0);
        assert!(!polygons_intersect(&a, &b));
    }

    #[test]
    fn test_intersect_fully_contained() {
        let a = square(0.0, 0.0, 10.0);
        let b = square(2.0, 2.0, 3.0);
        assert!(polygons_intersect(&a, &b));
        assert!(polygons_intersect(&b, &a)); // symmetric
    }

    #[test]
    fn test_intersect_collinear_strip_overlap() {
        // Two axis-aligned rectangles overlapping in the strip x∈[1,2]; the
        // vertical edges are collinear so no vertex is strictly interior — an
        // edge-midpoint sample is what reveals the overlap.
        let a = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let b = vec![(1.0, 0.0), (3.0, 0.0), (3.0, 2.0), (1.0, 2.0)];
        assert!(polygons_intersect(&a, &b));
    }

    #[test]
    fn test_intersect_concave_notch_not_overlap() {
        // A square resting in the L's concave notch does NOT overlap it — but
        // SAT (convex-hull) *does* report overlap. This is the raison d'être of
        // polygons_intersect over polygons_overlap.
        let l = l_shape();
        let peg = square(1.2, 1.2, 0.5); // fully inside the notch, outside the L
        assert!(polygons_overlap(&l, &peg), "SAT over-reports (convex hull)");
        assert!(
            !polygons_intersect(&l, &peg),
            "exact test must not flag a part nested in the notch"
        );
        assert!(!polygons_intersect(&peg, &l));
    }

    #[test]
    fn test_intersect_concave_real_overlap() {
        // A square straddling the L's inner corner genuinely overlaps.
        let l = l_shape();
        let peg = square(0.5, 0.5, 1.0);
        assert!(polygons_intersect(&l, &peg));
    }

    #[test]
    fn test_intersect_degenerate() {
        let a = vec![(0.0, 0.0), (1.0, 0.0)]; // not a polygon
        let b = square(0.0, 0.0, 10.0);
        assert!(!polygons_intersect(&a, &b));
    }

    proptest::proptest! {
        /// Overlap is a symmetric relation for any two polygons.
        #[test]
        fn prop_intersect_symmetric(
            ax in -5.0f64..5.0, ay in -5.0f64..5.0, asz in 0.5f64..5.0,
            bx in -5.0f64..5.0, by in -5.0f64..5.0, bsz in 0.5f64..5.0,
        ) {
            let a = square(ax, ay, asz);
            let b = square(bx, by, bsz);
            proptest::prop_assert_eq!(polygons_intersect(&a, &b), polygons_intersect(&b, &a));
        }

        /// Polygons pushed far apart never overlap.
        #[test]
        fn prop_intersect_far_apart_disjoint(sz in 0.5f64..5.0, gap in 100.0f64..200.0) {
            let a = square(0.0, 0.0, sz);
            let b = square(gap, 0.0, sz);
            proptest::prop_assert!(!polygons_intersect(&a, &b));
        }
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

    // ---- SAT collision: invariants ----
    //
    // The Separating Axis Theorem guarantees:
    //   - Separated polygons  → no collision
    //   - Overlapping polygons → collision
    //   - Fully contained     → collision
    //   - Boundary touching   → no collision (within tolerance 1e-6)
    //
    // Reference: Ericson (2005), Real-Time Collision Detection, Ch. 4.4

    #[test]
    fn test_sat_separated_no_collision() {
        // Two squares with a clear gap → never overlapping
        let a = square(0.0, 0.0, 5.0);
        let b = square(10.0, 0.0, 5.0); // gap of 5 units on x
        assert!(
            !polygons_overlap(&a, &b),
            "clearly separated polygons must not overlap"
        );
    }

    #[test]
    fn test_sat_overlapping_collision() {
        // Squares that share a 2×2 area
        let a = square(0.0, 0.0, 6.0);
        let b = square(4.0, 0.0, 6.0); // 2 units overlap in x
        assert!(
            polygons_overlap(&a, &b),
            "overlapping polygons must report collision"
        );
    }

    #[test]
    fn test_sat_fully_contained_collision() {
        // Small square fully inside large square
        let outer = square(0.0, 0.0, 10.0);
        let inner = square(3.0, 3.0, 3.0);
        assert!(
            polygons_overlap(&outer, &inner),
            "fully contained polygon must report collision"
        );
        // Symmetric
        assert!(
            polygons_overlap(&inner, &outer),
            "collision must be symmetric"
        );
    }

    #[test]
    fn test_sat_touching_boundary_no_collision() {
        // Squares that share only an edge (no area overlap)
        let a = square(0.0, 0.0, 5.0);
        let b = square(5.0, 0.0, 5.0); // touching at x=5
        assert!(
            !polygons_overlap(&a, &b),
            "boundary-touching polygons must not report collision (tolerance=1e-6)"
        );
    }

    #[test]
    fn test_sat_symmetry() {
        // polygons_overlap(A, B) == polygons_overlap(B, A)
        let a = square(0.0, 0.0, 7.0);
        let b = square(5.0, 3.0, 7.0);
        assert_eq!(
            polygons_overlap(&a, &b),
            polygons_overlap(&b, &a),
            "collision detection must be symmetric"
        );
    }

    #[test]
    fn test_sat_self_overlap() {
        // A polygon always overlaps with itself
        let a = square(0.0, 0.0, 5.0);
        assert!(polygons_overlap(&a, &a), "polygon must overlap with itself");
    }
}
