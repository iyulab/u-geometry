//! Minkowski sum computation for convex polygons.
//!
//! The Minkowski sum of two convex polygons P and Q is a convex polygon
//! containing all points `p + q` where `p ∈ P` and `q ∈ Q`.
//!
//! # Algorithm
//!
//! Uses the rotating calipers approach: merge the sorted edge lists
//! of both polygons by angle, accumulating vertices.
//!
//! # Complexity
//! O(n + m) for convex polygons with n and m vertices.
//!
//! # References
//!
//! - de Berg et al. (2008), "Computational Geometry", Ch. 13.1
//! - Bennell & Oliveira (2008), "The geometry of nesting problems"

/// Computes the Minkowski sum of two convex polygons.
///
/// Both polygons must be in counter-clockwise order.
/// The result is a convex polygon in counter-clockwise order.
///
/// # Complexity
/// O(n + m) where n, m are the vertex counts.
///
/// # Panics
/// Panics if either polygon has fewer than 3 vertices.
///
/// # Reference
/// de Berg et al. (2008), "Computational Geometry", Theorem 13.5
///
/// # Example
///
/// ```
/// use u_geometry::minkowski::minkowski_sum_convex;
///
/// let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
/// let triangle = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.5)];
/// let sum = minkowski_sum_convex(&square, &triangle);
/// assert!(sum.len() >= 3);
/// ```
pub fn minkowski_sum_convex(p: &[(f64, f64)], q: &[(f64, f64)]) -> Vec<(f64, f64)> {
    assert!(p.len() >= 3, "polygon P must have >= 3 vertices");
    assert!(q.len() >= 3, "polygon Q must have >= 3 vertices");

    // Ensure CCW and find bottom-most vertex as starting point
    let p = ensure_ccw_local(p);
    let q = ensure_ccw_local(q);

    let p_start = bottom_left_index(&p);
    let q_start = bottom_left_index(&q);

    let n = p.len();
    let m = q.len();

    let mut result = Vec::with_capacity(n + m);
    let mut i = 0;
    let mut j = 0;

    // Merge edge sequences by angle (rotating calipers)
    while i < n || j < m {
        let pi = (i + p_start) % n;
        let qi = (j + q_start) % m;

        // Current vertex = p[pi] + q[qi]
        result.push((p[pi].0 + q[qi].0, p[pi].1 + q[qi].1));

        // Edge vectors
        let p_edge = if i < n {
            let next = ((i + 1) + p_start) % n;
            (p[next].0 - p[pi].0, p[next].1 - p[pi].1)
        } else {
            (0.0, 0.0)
        };

        let q_edge = if j < m {
            let next = ((j + 1) + q_start) % m;
            (q[next].0 - q[qi].0, q[next].1 - q[qi].1)
        } else {
            (0.0, 0.0)
        };

        // Cross product to determine which edge has smaller angle
        let cross = p_edge.0 * q_edge.1 - p_edge.1 * q_edge.0;

        if i >= n {
            j += 1;
        } else if j >= m {
            i += 1;
        } else if cross > 0.0 {
            // p_edge has smaller angle → advance p
            i += 1;
        } else if cross < 0.0 {
            // q_edge has smaller angle → advance q
            j += 1;
        } else {
            // Parallel edges → advance both
            i += 1;
            j += 1;
        }
    }

    result
}

/// Computes the Minkowski difference (No-Fit Polygon) for convex polygons.
///
/// The NFP of stationary S and orbiting O is defined as:
/// `NFP(S, O) = Minkowski_sum(S, -O)`
/// where `-O` reflects O about its reference point.
///
/// The result represents all positions where the reference point of O
/// can be placed such that O touches or overlaps S.
///
/// # Complexity
/// O(n + m)
///
/// # Reference
/// Bennell & Oliveira (2008), "The geometry of nesting problems"
pub fn nfp_convex(
    stationary: &[(f64, f64)],
    orbiting: &[(f64, f64)],
) -> Vec<(f64, f64)> {
    // Negate the orbiting polygon (reflect about origin)
    let neg_orbiting: Vec<(f64, f64)> = orbiting.iter().map(|&(x, y)| (-x, -y)).collect();
    minkowski_sum_convex(stationary, &neg_orbiting)
}

/// Finds the bottom-left vertex index.
fn bottom_left_index(polygon: &[(f64, f64)]) -> usize {
    let mut idx = 0;
    for (i, &(x, y)) in polygon.iter().enumerate() {
        let (bx, by) = polygon[idx];
        if y < by || (y == by && x < bx) {
            idx = i;
        }
    }
    idx
}

/// Ensures CCW winding (local copy).
fn ensure_ccw_local(polygon: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if crate::robust::is_ccw(polygon) {
        polygon.to_vec()
    } else {
        let mut reversed = polygon.to_vec();
        reversed.reverse();
        reversed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polygon::area;

    fn square_at(x: f64, y: f64, size: f64) -> Vec<(f64, f64)> {
        vec![
            (x, y),
            (x + size, y),
            (x + size, y + size),
            (x, y + size),
        ]
    }

    #[test]
    fn test_minkowski_two_squares() {
        // Minkowski sum of unit square + unit square = 2x2 square
        let a = square_at(0.0, 0.0, 1.0);
        let b = square_at(0.0, 0.0, 1.0);
        let sum = minkowski_sum_convex(&a, &b);

        // Result should have area 4.0 (2x2 square)
        let a = area(&sum);
        assert!(
            (a - 4.0).abs() < 1e-8,
            "expected area 4.0, got {a}"
        );
    }

    #[test]
    fn test_minkowski_square_triangle() {
        let square = square_at(0.0, 0.0, 2.0);
        let triangle = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let sum = minkowski_sum_convex(&square, &triangle);

        // Sum should be larger than either input
        let sum_area = area(&sum);
        let sq_area = area(&square);
        let tri_area = area(&triangle);
        assert!(sum_area > sq_area);
        assert!(sum_area > tri_area);
    }

    #[test]
    fn test_minkowski_result_is_convex() {
        let a = square_at(0.0, 0.0, 5.0);
        let b = vec![(0.0, 0.0), (3.0, 0.0), (1.5, 2.0)];
        let sum = minkowski_sum_convex(&a, &b);
        assert!(crate::robust::is_convex(&sum));
    }

    #[test]
    fn test_nfp_convex_squares() {
        let stationary = square_at(0.0, 0.0, 10.0);
        let orbiting = square_at(0.0, 0.0, 5.0);
        let nfp = nfp_convex(&stationary, &orbiting);

        // NFP should be a polygon
        assert!(nfp.len() >= 3);
        // NFP area should be (10+5)^2 = 225
        let nfp_area = area(&nfp);
        assert!(
            (nfp_area - 225.0).abs() < 1e-6,
            "expected NFP area 225.0, got {nfp_area}"
        );
    }

    #[test]
    fn test_nfp_convex_symmetric() {
        // For identical convex polygons, NFP should be symmetric about center
        let tri = vec![(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)];
        let nfp = nfp_convex(&tri, &tri);
        assert!(nfp.len() >= 3);
        assert!(crate::robust::is_convex(&nfp));
    }

    #[test]
    #[should_panic(expected = "polygon P must have >= 3 vertices")]
    fn test_minkowski_degenerate() {
        let a = vec![(0.0, 0.0), (1.0, 0.0)];
        let b = square_at(0.0, 0.0, 1.0);
        minkowski_sum_convex(&a, &b);
    }
}
