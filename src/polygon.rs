//! Polygon operations: area, centroid, convex hull, winding.
//!
//! All functions operate on slices of `(f64, f64)` tuples for
//! maximum compatibility. Use `Point2::to_tuple()` for interop.
//!
//! # References
//!
//! - O'Rourke (1998), "Computational Geometry in C", Ch.1 (polygon area)
//! - Graham (1972), "An efficient algorithm for determining the convex hull"
//! - de Berg et al. (2008), "Computational Geometry", Ch.1

use crate::robust::orient2d;

/// Computes the signed area of a simple polygon using the Shoelace formula.
///
/// Positive for counter-clockwise winding, negative for clockwise.
/// Uses Kahan summation for numerical stability.
///
/// # Complexity
/// O(n)
///
/// # Reference
/// Meister (1769), Shoelace formula
pub fn signed_area(polygon: &[(f64, f64)]) -> f64 {
    let n = polygon.len();
    if n < 3 {
        return 0.0;
    }

    // Kahan compensated summation
    let mut sum = 0.0;
    let mut c = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        let term = polygon[i].0 * polygon[j].1 - polygon[j].0 * polygon[i].1;

        let y = term - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    sum * 0.5
}

/// Computes the unsigned area of a simple polygon.
///
/// # Complexity
/// O(n)
pub fn area(polygon: &[(f64, f64)]) -> f64 {
    signed_area(polygon).abs()
}

/// Computes the centroid (center of mass) of a simple polygon.
///
/// Assumes the polygon is non-degenerate (area > 0).
/// Returns `None` if the polygon has fewer than 3 vertices or zero area.
///
/// # Complexity
/// O(n)
///
/// # Reference
/// O'Rourke (1998), Eq. 1.6
pub fn centroid(polygon: &[(f64, f64)]) -> Option<(f64, f64)> {
    let n = polygon.len();
    if n < 3 {
        return None;
    }

    let a = signed_area(polygon);
    if a.abs() < 1e-15 {
        return None;
    }

    let mut cx = 0.0;
    let mut cy = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        let cross = polygon[i].0 * polygon[j].1 - polygon[j].0 * polygon[i].1;
        cx += (polygon[i].0 + polygon[j].0) * cross;
        cy += (polygon[i].1 + polygon[j].1) * cross;
    }

    let inv = 1.0 / (6.0 * a);
    Some((cx * inv, cy * inv))
}

/// Computes the unsigned area of a polygon with holes.
///
/// The total area is the exterior area minus the sum of hole areas.
///
/// # Complexity
/// O(n) where n is the total number of vertices across all rings
pub fn area_with_holes(exterior: &[(f64, f64)], holes: &[&[(f64, f64)]]) -> f64 {
    let mut total = area(exterior);
    for hole in holes {
        total -= area(hole);
    }
    total.max(0.0)
}

/// Computes the centroid of a polygon with holes.
///
/// Uses area-weighted centroid: each ring's centroid is weighted by its
/// signed area, then combined. Returns `None` if the net area is zero.
///
/// # Complexity
/// O(n) where n is the total number of vertices across all rings
///
/// # Reference
/// Extension of O'Rourke (1998), Eq. 1.6 to multiply-connected polygons
pub fn centroid_with_holes(
    exterior: &[(f64, f64)],
    holes: &[&[(f64, f64)]],
) -> Option<(f64, f64)> {
    let ext_area = signed_area(exterior);
    let ext_centroid = centroid(exterior)?;

    if holes.is_empty() {
        return Some(ext_centroid);
    }

    let mut weighted_cx = ext_area * ext_centroid.0;
    let mut weighted_cy = ext_area * ext_centroid.1;
    let mut total_area = ext_area;

    for hole in holes {
        let h_area = signed_area(hole);
        if let Some((hcx, hcy)) = centroid(hole) {
            weighted_cx -= h_area.abs() * hcx;
            weighted_cy -= h_area.abs() * hcy;
            total_area -= h_area.abs();
        }
    }

    if total_area.abs() < 1e-15 {
        return None;
    }

    Some((weighted_cx / total_area, weighted_cy / total_area))
}

/// Computes the perimeter of a polygon.
///
/// # Complexity
/// O(n)
pub fn perimeter(polygon: &[(f64, f64)]) -> f64 {
    let n = polygon.len();
    if n < 2 {
        return 0.0;
    }

    let mut p = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = polygon[j].0 - polygon[i].0;
        let dy = polygon[j].1 - polygon[i].1;
        p += (dx * dx + dy * dy).sqrt();
    }
    p
}

/// Computes the convex hull of a set of points using Graham scan.
///
/// Returns the hull vertices in counter-clockwise order.
/// Uses robust orientation tests for correctness.
///
/// # Complexity
/// O(n log n) time, O(n) space
///
/// # Reference
/// Graham (1972), "An efficient algorithm for determining the convex hull
/// of a finite planar set"
pub fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }

    // Find the lowest-leftmost point (pivot)
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    let mut pivot_idx = 0;
    for (i, &(x, y)) in pts.iter().enumerate() {
        let (px, py) = pts[pivot_idx];
        if y < py || (y == py && x < px) {
            pivot_idx = i;
        }
    }
    pts.swap(0, pivot_idx);
    let pivot = pts[0];

    // Sort by polar angle from pivot
    pts[1..].sort_by(|a, b| {
        let o = orient2d(pivot, *a, *b);
        match o {
            crate::robust::Orientation::CounterClockwise => std::cmp::Ordering::Less,
            crate::robust::Orientation::Clockwise => std::cmp::Ordering::Greater,
            crate::robust::Orientation::Collinear => {
                // Closer point first
                let da = (a.0 - pivot.0).powi(2) + (a.1 - pivot.1).powi(2);
                let db = (b.0 - pivot.0).powi(2) + (b.1 - pivot.1).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    });

    // Graham scan
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(n);
    for &p in &pts {
        while hull.len() >= 2 {
            let top = hull[hull.len() - 1];
            let second = hull[hull.len() - 2];
            if !orient2d(second, top, p).is_ccw() {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }

    hull
}

/// Checks if a simple polygon is convex.
///
/// Uses robust orientation predicates for numerical stability.
/// A polygon is convex if all consecutive edge turns have the same
/// orientation (all CCW or all CW). Collinear edges are skipped.
/// Returns `false` for polygons with fewer than 3 vertices.
///
/// # Complexity
/// O(n)
pub fn is_convex(polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut expected: Option<crate::robust::Orientation> = None;

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
            Some(ref exp) if *exp != o => return false,
            _ => {}
        }
    }

    true
}

/// Ensures a polygon has counter-clockwise winding order.
///
/// If the polygon is already CCW, returns a clone. Otherwise reverses it.
///
/// # Complexity
/// O(n)
pub fn ensure_ccw(polygon: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if polygon.len() < 3 {
        return polygon.to_vec();
    }
    if crate::robust::is_ccw(polygon) {
        polygon.to_vec()
    } else {
        let mut reversed = polygon.to_vec();
        reversed.reverse();
        reversed
    }
}

/// Checks if a point lies inside a simple polygon using the winding number method.
///
/// Uses robust orientation tests for correctness on edges and vertices.
///
/// # Complexity
/// O(n) where n is the number of polygon vertices
///
/// # Reference
/// O'Rourke (1998), Ch. 7.4 — Winding number algorithm
pub fn contains_point(polygon: &[(f64, f64)], point: (f64, f64)) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut winding = 0i32;

    for i in 0..n {
        let j = (i + 1) % n;
        let (ax, ay) = polygon[i];
        let (bx, by) = polygon[j];

        if ay <= point.1 {
            if by > point.1 {
                // Upward crossing
                if orient2d((ax, ay), (bx, by), point).is_ccw() {
                    winding += 1;
                }
            }
        } else if by <= point.1 {
            // Downward crossing
            if orient2d((ax, ay), (bx, by), point).is_cw() {
                winding -= 1;
            }
        }
    }

    winding != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Area tests ----

    #[test]
    fn test_signed_area_ccw_square() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let a = signed_area(&square);
        assert!((a - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_signed_area_cw_square() {
        let square = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        let a = signed_area(&square);
        assert!((a - (-100.0)).abs() < 1e-10);
    }

    #[test]
    fn test_area_triangle() {
        let tri = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        assert!((area(&tri) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_area_degenerate() {
        assert!((area(&[(0.0, 0.0), (1.0, 0.0)]) - 0.0).abs() < 1e-15);
        assert!((area(&[]) - 0.0).abs() < 1e-15);
    }

    // ---- Centroid tests ----

    #[test]
    fn test_centroid_square() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let (cx, cy) = centroid(&square).unwrap();
        assert!((cx - 5.0).abs() < 1e-10);
        assert!((cy - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_triangle() {
        let tri = [(0.0, 0.0), (6.0, 0.0), (3.0, 6.0)];
        let (cx, cy) = centroid(&tri).unwrap();
        assert!((cx - 3.0).abs() < 1e-10);
        assert!((cy - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_degenerate() {
        assert!(centroid(&[(0.0, 0.0), (1.0, 0.0)]).is_none());
        // Collinear points (zero area)
        assert!(centroid(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]).is_none());
    }

    // ---- Area with holes tests ----

    #[test]
    fn test_area_with_holes_no_holes() {
        let ext = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        assert!((area_with_holes(&ext, &[]) - 10000.0).abs() < 1e-10);
    }

    #[test]
    fn test_area_with_holes_one_hole() {
        let ext = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let hole: &[(f64, f64)] =
            &[(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)];
        // 10000 - 2500 = 7500
        assert!((area_with_holes(&ext, &[hole]) - 7500.0).abs() < 1e-10);
    }

    #[test]
    fn test_area_with_holes_two_holes() {
        let ext = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let h1: &[(f64, f64)] = &[(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)];
        let h2: &[(f64, f64)] = &[(50.0, 50.0), (70.0, 50.0), (70.0, 70.0), (50.0, 70.0)];
        // 10000 - 400 - 400 = 9200
        assert!((area_with_holes(&ext, &[h1, h2]) - 9200.0).abs() < 1e-10);
    }

    // ---- Centroid with holes tests ----

    #[test]
    fn test_centroid_with_holes_no_holes() {
        let ext = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let (cx, cy) = centroid_with_holes(&ext, &[]).unwrap();
        assert!((cx - 5.0).abs() < 1e-10);
        assert!((cy - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_with_holes_symmetric_hole() {
        // Square with centered square hole — centroid stays at center
        let ext = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let hole: &[(f64, f64)] =
            &[(25.0, 25.0), (75.0, 25.0), (75.0, 75.0), (25.0, 75.0)];
        let (cx, cy) = centroid_with_holes(&ext, &[hole]).unwrap();
        assert!((cx - 50.0).abs() < 1e-6);
        assert!((cy - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_centroid_with_holes_asymmetric_hole() {
        // Square with hole in upper-right — centroid shifts toward lower-left
        let ext = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let hole: &[(f64, f64)] =
            &[(50.0, 50.0), (100.0, 50.0), (100.0, 100.0), (50.0, 100.0)];
        let (cx, cy) = centroid_with_holes(&ext, &[hole]).unwrap();
        // Centroid should shift toward lower-left
        assert!(cx < 50.0);
        assert!(cy < 50.0);
    }

    // ---- Perimeter tests ----

    #[test]
    fn test_perimeter_square() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!((perimeter(&square) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_perimeter_empty() {
        assert!((perimeter(&[]) - 0.0).abs() < 1e-15);
    }

    // ---- Convex Hull tests ----

    #[test]
    fn test_convex_hull_square() {
        let points = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (5.0, 5.0), // interior point
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4);
        // All hull points should be corners (not the interior point)
        assert!((area(&hull) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_convex_hull_triangle() {
        let points = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn test_convex_hull_collinear() {
        let points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        let hull = convex_hull(&points);
        // Collinear points produce a degenerate hull
        assert!(hull.len() <= 3);
    }

    #[test]
    fn test_convex_hull_l_shape() {
        let l_shape = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (5.0, 5.0),
            (5.0, 10.0),
            (0.0, 10.0),
        ];
        let hull = convex_hull(&l_shape);
        // L-shape hull: (0,0), (10,0), (10,5), (5,10), (0,10) — 5 vertices
        assert_eq!(hull.len(), 5);
    }

    // ---- is_convex tests ----

    #[test]
    fn test_is_convex_square() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(is_convex(&square));
    }

    #[test]
    fn test_is_convex_triangle() {
        let tri = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        assert!(is_convex(&tri));
    }

    #[test]
    fn test_is_convex_l_shape() {
        let l = [
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 10.0),
            (10.0, 10.0),
            (10.0, 20.0),
            (0.0, 20.0),
        ];
        assert!(!is_convex(&l));
    }

    #[test]
    fn test_is_convex_degenerate() {
        assert!(!is_convex(&[(0.0, 0.0), (1.0, 0.0)]));
        assert!(!is_convex(&[]));
    }

    // ---- ensure_ccw tests ----

    #[test]
    fn test_ensure_ccw_already_ccw() {
        let ccw = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let result = ensure_ccw(&ccw);
        assert_eq!(result, ccw.to_vec());
    }

    #[test]
    fn test_ensure_ccw_from_cw() {
        let cw = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        let result = ensure_ccw(&cw);
        assert!(crate::robust::is_ccw(&result));
    }

    // ---- contains_point tests ----

    #[test]
    fn test_contains_point_inside_square() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(contains_point(&square, (5.0, 5.0)));
    }

    #[test]
    fn test_contains_point_outside_square() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(!contains_point(&square, (15.0, 5.0)));
    }

    #[test]
    fn test_contains_point_concave_polygon() {
        // L-shape polygon
        let l_shape = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (5.0, 5.0),
            (5.0, 10.0),
            (0.0, 10.0),
        ];
        // Inside the L
        assert!(contains_point(&l_shape, (2.0, 2.0)));
        assert!(contains_point(&l_shape, (2.0, 8.0)));
        // In the concave "notch" — outside
        assert!(!contains_point(&l_shape, (8.0, 8.0)));
    }

    #[test]
    fn test_contains_point_triangle() {
        let tri = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        assert!(contains_point(&tri, (5.0, 3.0)));
        assert!(!contains_point(&tri, (20.0, 5.0)));
    }
}
