//! Polygon offset via raw edge offset with miter/bevel joins.
//!
//! Offsets a simple polygon by moving each edge along its outward normal,
//! then computing miter (or bevel) joins at vertices. Self-intersections
//! in the raw offset are detected and resolved by splitting into sub-loops
//! and filtering by signed area.
//!
//! # Algorithm
//!
//! 1. Ensure CCW winding
//! 2. Offset each edge by translating along its outward normal
//! 3. Compute miter joins at consecutive offset edges
//! 4. Apply miter limit -- bevel if ratio exceeds threshold
//! 5. Detect and resolve self-intersections
//!
//! # Complexity
//!
//! O(n^2) worst case for self-intersection resolution on concave polygons,
//! O(n) for convex polygons.
//!
//! # References
//!
//! - Held (1998), "On the Computational Geometry of Pocket Machining", Ch. 3
//! - Chen & McMains (2005), "Polygon Offsetting by Computing Winding Numbers"

use crate::polygon;

/// Default miter limit ratio (miter length / offset distance).
const DEFAULT_MITER_LIMIT: f64 = 2.0;

/// Tolerance for detecting parallel edges and degenerate geometry.
const EPS: f64 = 1e-10;

/// Offsets a simple polygon by `distance` using the default miter limit (2.0).
///
/// Positive distance expands the polygon outward (for CCW winding).
/// Negative distance shrinks inward. Zero distance returns the original polygon.
///
/// Returns a `Vec` of polygon rings. The result may be empty if the polygon
/// collapses, or contain multiple rings if the offset causes topology splits.
///
/// # Arguments
///
/// * `polygon` - A simple polygon as a slice of `(f64, f64)` vertices
/// * `distance` - Offset distance (positive = outward, negative = inward)
///
/// # Complexity
///
/// O(n) for convex polygons, O(n^2) worst case for concave polygons
/// due to self-intersection resolution.
///
/// # Example
///
/// ```
/// use u_geometry::offset::offset_polygon;
///
/// let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
/// let result = offset_polygon(&square, 1.0);
/// assert_eq!(result.len(), 1);
/// ```
pub fn offset_polygon(polygon: &[(f64, f64)], distance: f64) -> Vec<Vec<(f64, f64)>> {
    offset_polygon_miter(polygon, distance, DEFAULT_MITER_LIMIT)
}

/// Offsets a simple polygon by `distance` with a configurable miter limit.
///
/// The miter limit controls how far a miter join can extend relative to the
/// offset distance. When the miter ratio at a vertex exceeds `miter_limit`,
/// a bevel join is inserted instead, preventing long spikes at acute angles.
///
/// # Arguments
///
/// * `polygon` - A simple polygon as a slice of `(f64, f64)` vertices
/// * `distance` - Offset distance (positive = outward, negative = inward)
/// * `miter_limit` - Maximum miter ratio before switching to bevel
///
/// # Returns
///
/// `Vec` of polygon rings. May be empty, single, or multiple rings.
pub fn offset_polygon_miter(
    polygon: &[(f64, f64)],
    distance: f64,
    miter_limit: f64,
) -> Vec<Vec<(f64, f64)>> {
    if polygon.len() < 3 {
        return Vec::new();
    }

    if distance.abs() < EPS {
        return vec![polygon.to_vec()];
    }

    let poly = polygon::ensure_ccw(polygon);
    let n = poly.len();

    // Step 1: Compute offset edges
    let offset_edges: Vec<((f64, f64), (f64, f64))> = (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            offset_edge(poly[i], poly[j], distance)
        })
        .collect();

    // Step 2: Compute miter/bevel joins at each vertex
    let mut raw_vertices: Vec<(f64, f64)> = Vec::new();

    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };

        let (ep0, ep1) = offset_edges[prev];
        let (ec0, ec1) = offset_edges[i];

        let dir_prev = (ep1.0 - ep0.0, ep1.1 - ep0.1);
        let dir_curr = (ec1.0 - ec0.0, ec1.1 - ec0.1);

        if let Some(pt) = line_intersection(ep0, dir_prev, ec0, dir_curr) {
            // Add EPS tolerance to miter limit to avoid false bevel triggers
            // from floating-point rounding near the boundary.
            let miter_dist_sq = (pt.0 - poly[i].0).powi(2) + (pt.1 - poly[i].1).powi(2);
            let limit_sq = (miter_limit * distance.abs() * (1.0 + 1e-4)).powi(2);

            if miter_dist_sq > limit_sq {
                raw_vertices.push(ep1);
                raw_vertices.push(ec0);
            } else {
                raw_vertices.push(pt);
            }
        } else {
            raw_vertices.push(ep1);
        }
    }

    let raw_vertices = remove_consecutive_duplicates(&raw_vertices);

    if raw_vertices.len() < 3 {
        return Vec::new();
    }

    let mut result = resolve_self_intersections(&raw_vertices);

    // Filter out "ghost" polygons that appear when inward offset exceeds
    // half the polygon width. These have valid CCW winding but are actually
    // "inverted" results where edges crossed over each other.
    // Detection: every vertex of a valid inward offset must be at least
    // |distance| away from every original edge (within tolerance).
    if distance < 0.0 {
        let abs_d = distance.abs();
        let tol = abs_d * 0.1 + EPS; // 10% tolerance for numerical stability

        result.retain(|ring| {
            // Check that at least one vertex has proper distance from all edges
            ring.iter().all(|&v| {
                let min_dist = min_distance_to_polygon_edges(&poly, v);
                min_dist >= abs_d - tol
            })
        });
    }

    result
}

/// Computes the minimum distance from a point to any edge of a polygon.
fn min_distance_to_polygon_edges(polygon: &[(f64, f64)], point: (f64, f64)) -> f64 {
    let n = polygon.len();
    let mut min_d = f64::MAX;

    for i in 0..n {
        let j = (i + 1) % n;
        let d = point_to_segment_distance(point, polygon[i], polygon[j]);
        if d < min_d {
            min_d = d;
        }
    }

    min_d
}

/// Computes the distance from a point to a line segment.
fn point_to_segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let len_sq = dx * dx + dy * dy;

    if len_sq < EPS {
        return ((p.0 - a.0).powi(2) + (p.1 - a.1).powi(2)).sqrt();
    }

    // Project point onto line, clamped to [0, 1]
    let t = ((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);

    let proj = (a.0 + t * dx, a.1 + t * dy);
    ((p.0 - proj.0).powi(2) + (p.1 - proj.1).powi(2)).sqrt()
}

/// Offsets a single directed edge segment by translating along its outward normal.
///
/// For a CCW-wound polygon, the outward normal of edge `(p0 -> p1)` points to
/// the right: `(dy, -dx)` normalized, then scaled by `distance`.
pub fn offset_edge(
    p0: (f64, f64),
    p1: (f64, f64),
    distance: f64,
) -> ((f64, f64), (f64, f64)) {
    let dx = p1.0 - p0.0;
    let dy = p1.1 - p0.1;
    let len = (dx * dx + dy * dy).sqrt();

    if len < EPS {
        return (p0, p1);
    }

    let nx = dy / len * distance;
    let ny = -dx / len * distance;

    ((p0.0 + nx, p0.1 + ny), (p1.0 + nx, p1.1 + ny))
}

/// Computes the intersection of two lines defined by point + direction.
///
/// Each line is defined as `p + t * d` for scalar `t`.
/// Returns `None` if the lines are parallel (cross product near zero).
///
/// # Complexity
///
/// O(1)
pub fn line_intersection(
    p1: (f64, f64),
    d1: (f64, f64),
    p2: (f64, f64),
    d2: (f64, f64),
) -> Option<(f64, f64)> {
    let cross = d1.0 * d2.1 - d1.1 * d2.0;

    if cross.abs() < EPS {
        return None;
    }

    let dp = (p2.0 - p1.0, p2.1 - p1.1);
    let t = (dp.0 * d2.1 - dp.1 * d2.0) / cross;

    Some((p1.0 + t * d1.0, p1.1 + t * d1.1))
}

fn segment_intersection(
    a0: (f64, f64),
    a1: (f64, f64),
    b0: (f64, f64),
    b1: (f64, f64),
) -> Option<(f64, f64, (f64, f64))> {
    let da = (a1.0 - a0.0, a1.1 - a0.1);
    let db = (b1.0 - b0.0, b1.1 - b0.1);
    let cross = da.0 * db.1 - da.1 * db.0;

    if cross.abs() < EPS {
        return None;
    }

    let dp = (b0.0 - a0.0, b0.1 - a0.1);
    let t = (dp.0 * db.1 - dp.1 * db.0) / cross;
    let u = (dp.0 * da.1 - dp.1 * da.0) / cross;

    if t > EPS && t < 1.0 - EPS && u > EPS && u < 1.0 - EPS {
        let pt = (a0.0 + t * da.0, a0.1 + t * da.1);
        Some((t, u, pt))
    } else {
        None
    }
}

fn remove_consecutive_duplicates(vertices: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if vertices.is_empty() {
        return Vec::new();
    }

    let mut result = vec![vertices[0]];

    for &v in &vertices[1..] {
        let last = result.last().expect("result is non-empty after initial push");
        if (v.0 - last.0).abs() > EPS || (v.1 - last.1).abs() > EPS {
            result.push(v);
        }
    }

    if result.len() > 1 {
        let first = result[0];
        let last = result.last().expect("result has at least 2 elements");
        if (first.0 - last.0).abs() < EPS && (first.1 - last.1).abs() < EPS {
            result.pop();
        }
    }

    result
}

fn resolve_self_intersections(vertices: &[(f64, f64)]) -> Vec<Vec<(f64, f64)>> {
    let n = vertices.len();

    let mut event_edge_i: Vec<usize> = Vec::new();
    let mut event_edge_j: Vec<usize> = Vec::new();
    let mut event_t_i: Vec<f64> = Vec::new();
    let mut event_t_j: Vec<f64> = Vec::new();
    let mut event_pts: Vec<(f64, f64)> = Vec::new();

    for i in 0..n {
        let i_next = (i + 1) % n;
        for j in (i + 2)..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            let j_next = (j + 1) % n;

            if let Some((t, u, pt)) = segment_intersection(
                vertices[i], vertices[i_next],
                vertices[j], vertices[j_next],
            ) {
                event_edge_i.push(i);
                event_edge_j.push(j);
                event_t_i.push(t);
                event_t_j.push(u);
                event_pts.push(pt);
            }
        }
    }

    if event_edge_i.is_empty() {
        let sa = polygon::signed_area(vertices);
        if sa > EPS {
            return vec![vertices.to_vec()];
        }
        return Vec::new();
    }

    let num_events = event_edge_i.len();
    let mut edge_splits: Vec<Vec<(f64, usize)>> = vec![Vec::new(); n];

    for ei in 0..num_events {
        edge_splits[event_edge_i[ei]].push((event_t_i[ei], ei));
        edge_splits[event_edge_j[ei]].push((event_t_j[ei], ei));
    }

    for splits in &mut edge_splits {
        splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut node_points: Vec<(f64, f64)> = Vec::new();
    let mut node_is_isect: Vec<bool> = Vec::new();
    let mut isect_node_on_i: Vec<usize> = vec![0; num_events];
    let mut isect_node_on_j: Vec<usize> = vec![0; num_events];

    for i in 0..n {
        node_points.push(vertices[i]);
        node_is_isect.push(false);

        for &(_, ei) in &edge_splits[i] {
            let node_idx = node_points.len();
            node_points.push(event_pts[ei]);
            node_is_isect.push(true);

            if event_edge_i[ei] == i {
                isect_node_on_i[ei] = node_idx;
            } else {
                isect_node_on_j[ei] = node_idx;
            }
        }
    }

    let total = node_points.len();
    let forward: Vec<usize> = (0..total).map(|i| (i + 1) % total).collect();

    let mut jump: Vec<Option<usize>> = vec![None; total];
    for ei in 0..num_events {
        let ni = isect_node_on_i[ei];
        let nj = isect_node_on_j[ei];
        jump[ni] = Some(forward[nj]);
        jump[nj] = Some(forward[ni]);
    }

    let mut visited = vec![false; total];
    let mut result_loops: Vec<Vec<(f64, f64)>> = Vec::new();

    for start in 0..total {
        if visited[start] || !node_is_isect[start] {
            continue;
        }

        for use_jump_first in [true, false] {
            let first_step = if use_jump_first {
                match jump[start] {
                    Some(j) => j,
                    None => continue,
                }
            } else {
                forward[start]
            };

            let mut loop_pts = vec![node_points[start]];
            let mut curr = first_step;
            let mut steps = 0;
            let max_steps = total + 1;

            while curr != start && steps < max_steps {
                loop_pts.push(node_points[curr]);
                curr = match jump[curr].take() {
                    Some(j) => j,
                    None => forward[curr],
                };
                steps += 1;
            }

            if curr == start && loop_pts.len() >= 3 {
                let sa = polygon::signed_area(&loop_pts);

                if sa > EPS {
                    for &lp in &loop_pts {
                        for (idx, &np) in node_points.iter().enumerate() {
                            if (lp.0 - np.0).abs() < EPS
                                && (lp.1 - np.1).abs() < EPS
                            {
                                visited[idx] = true;
                            }
                        }
                    }
                    result_loops.push(loop_pts);
                }
            }
        }
    }

    if result_loops.is_empty() {
        let sa = polygon::signed_area(vertices);
        if sa > EPS {
            return vec![vertices.to_vec()];
        }
        return Vec::new();
    }

    result_loops
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_area(polygon: &[(f64, f64)]) -> f64 {
        polygon::area(polygon)
    }

    #[test]
    fn test_offset_edge_horizontal() {
        let ((q0x, q0y), (q1x, q1y)) = offset_edge((0.0, 0.0), (10.0, 0.0), 1.0);
        assert!((q0y - (-1.0)).abs() < EPS);
        assert!((q1y - (-1.0)).abs() < EPS);
        assert!((q0x - 0.0).abs() < EPS);
        assert!((q1x - 10.0).abs() < EPS);
    }

    #[test]
    fn test_offset_edge_vertical() {
        let ((q0x, q0y), (q1x, q1y)) = offset_edge((0.0, 0.0), (0.0, 10.0), 1.0);
        assert!((q0x - 1.0).abs() < EPS);
        assert!((q1x - 1.0).abs() < EPS);
        assert!((q0y - 0.0).abs() < EPS);
        assert!((q1y - 10.0).abs() < EPS);
    }

    #[test]
    fn test_offset_edge_zero_length() {
        let (q0, q1) = offset_edge((5.0, 5.0), (5.0, 5.0), 1.0);
        assert!((q0.0 - 5.0).abs() < EPS);
        assert!((q0.1 - 5.0).abs() < EPS);
        assert!((q1.0 - 5.0).abs() < EPS);
        assert!((q1.1 - 5.0).abs() < EPS);
    }

    #[test]
    fn test_line_intersection_perpendicular() {
        let pt = line_intersection((0.0, 0.0), (1.0, 0.0), (5.0, -5.0), (0.0, 1.0));
        assert!(pt.is_some());
        let (x, y) = pt.expect("perpendicular lines must intersect");
        assert!((x - 5.0).abs() < EPS);
        assert!((y - 0.0).abs() < EPS);
    }

    #[test]
    fn test_line_intersection_parallel() {
        let pt = line_intersection((0.0, 0.0), (1.0, 0.0), (0.0, 5.0), (1.0, 0.0));
        assert!(pt.is_none());
    }

    #[test]
    fn test_line_intersection_diagonal() {
        let pt = line_intersection((0.0, 0.0), (1.0, 1.0), (10.0, 0.0), (-1.0, 1.0));
        let (x, y) = pt.expect("diagonal lines must intersect");
        assert!((x - 5.0).abs() < EPS);
        assert!((y - 5.0).abs() < EPS);
    }

    #[test]
    fn test_square_offset_outward() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let result = offset_polygon(&square, 1.0);
        assert!(!result.is_empty());
        let a = approx_area(&result[0]);
        assert!((a - 144.0).abs() < 1.0, "got {a}");
    }

    #[test]
    fn test_square_offset_inward() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let result = offset_polygon(&square, -1.0);
        assert!(!result.is_empty());
        let a = approx_area(&result[0]);
        assert!((a - 64.0).abs() < 1.0, "got {a}");
    }

    #[test]
    fn test_square_offset_area_formula() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let d = 2.0;
        let result = offset_polygon(&square, d);
        assert!(!result.is_empty());
        let a = approx_area(&result[0]);
        let expected = 100.0 + 40.0 * d + 4.0 * d * d;
        assert!((a - expected).abs() < 1.0, "got {a}, expected {expected}");
    }

    #[test]
    fn test_triangle_offset_outward() {
        let tri = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        let result = offset_polygon(&tri, 0.5);
        assert!(!result.is_empty());
        let original_area = approx_area(&tri);
        let offset_area = approx_area(&result[0]);
        assert!(offset_area > original_area, "{offset_area} <= {original_area}");
    }

    #[test]
    fn test_triangle_offset_inward() {
        let tri = [(0.0, 0.0), (20.0, 0.0), (10.0, 17.32)];
        let result = offset_polygon(&tri, -1.0);
        assert!(!result.is_empty());
        let original_area = approx_area(&tri);
        let offset_area = approx_area(&result[0]);
        assert!(offset_area < original_area, "{offset_area} >= {original_area}");
    }

    #[test]
    fn test_convex_polygon_offset_preserves_convexity() {
        let hexagon: Vec<(f64, f64)> = (0..6)
            .map(|i| {
                let angle = std::f64::consts::PI / 3.0 * i as f64;
                (50.0 + 20.0 * angle.cos(), 50.0 + 20.0 * angle.sin())
            })
            .collect();
        let result = offset_polygon(&hexagon, 2.0);
        assert!(!result.is_empty());
        assert!(polygon::is_convex(&result[0]));
    }

    #[test]
    fn test_zero_distance_returns_original() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let result = offset_polygon(&square, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), square.len());
        for (orig, rp) in square.iter().zip(result[0].iter()) {
            assert!((orig.0 - rp.0).abs() < EPS);
            assert!((orig.1 - rp.1).abs() < EPS);
        }
    }

    #[test]
    fn test_large_negative_offset_collapses() {
        let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let result = offset_polygon(&square, -6.0);
        assert!(result.is_empty(), "got {} rings", result.len());
    }

    #[test]
    fn test_l_shape_offset_outward() {
        let l_shape = [
            (0.0, 0.0), (20.0, 0.0), (20.0, 10.0),
            (10.0, 10.0), (10.0, 20.0), (0.0, 20.0),
        ];
        let result = offset_polygon(&l_shape, 1.0);
        assert!(!result.is_empty());
        let original_area = approx_area(&l_shape);
        let total: f64 = result.iter().map(|r| approx_area(r)).sum();
        assert!(total > original_area, "{total} <= {original_area}");
    }

    #[test]
    fn test_l_shape_offset_inward() {
        let l_shape = [
            (0.0, 0.0), (20.0, 0.0), (20.0, 10.0),
            (10.0, 10.0), (10.0, 20.0), (0.0, 20.0),
        ];
        let result = offset_polygon(&l_shape, -1.0);
        assert!(!result.is_empty());
        let original_area = approx_area(&l_shape);
        let total: f64 = result.iter().map(|r| approx_area(r)).sum();
        assert!(total < original_area, "{total} >= {original_area}");
    }

    #[test]
    fn test_degenerate_polygon_too_few_vertices() {
        assert!(offset_polygon(&[(0.0, 0.0), (10.0, 0.0)], 1.0).is_empty());
    }

    #[test]
    fn test_empty_polygon() {
        assert!(offset_polygon(&[], 1.0).is_empty());
    }

    #[test]
    fn test_cw_polygon_normalized_to_ccw() {
        let cw = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        let result = offset_polygon(&cw, 1.0);
        assert!(!result.is_empty());
        let a = approx_area(&result[0]);
        assert!((a - 144.0).abs() < 1.0, "got {a}");
    }

    #[test]
    fn test_small_polygon_small_offset() {
        let tiny = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let result = offset_polygon(&tiny, 0.1);
        assert!(!result.is_empty());
        let a = approx_area(&result[0]);
        assert!((a - 1.44).abs() < 0.1, "got {a}");
    }

    #[test]
    fn test_miter_limit_with_acute_triangle() {
        let acute = [(0.0, 0.0), (100.0, 0.0), (50.0, 1.0)];
        let result = offset_polygon_miter(&acute, 2.0, 1.5);
        for ring in &result {
            for &(x, y) in ring {
                assert!(x.is_finite());
                assert!(y.is_finite());
            }
        }
    }

    #[test]
    fn test_regular_pentagon_offset() {
        let pentagon: Vec<(f64, f64)> = (0..5)
            .map(|i| {
                let angle = std::f64::consts::TAU / 5.0 * i as f64 - std::f64::consts::FRAC_PI_2;
                (50.0 + 30.0 * angle.cos(), 50.0 + 30.0 * angle.sin())
            })
            .collect();
        let result = offset_polygon(&pentagon, 3.0);
        assert!(!result.is_empty());
        assert!(approx_area(&result[0]) > approx_area(&pentagon));
    }

    #[cfg(test)]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        fn convex_polygon_strategy() -> impl Strategy<Value = Vec<(f64, f64)>> {
            (5..=12u32, 10.0..100.0f64).prop_map(|(n, radius)| {
                (0..n)
                    .map(|i| {
                        let angle = std::f64::consts::TAU / n as f64 * i as f64;
                        (50.0 + radius * angle.cos(), 50.0 + radius * angle.sin())
                    })
                    .collect()
            })
        }

        proptest! {
            #[test]
            fn prop_outward_offset_increases_area(
                poly in convex_polygon_strategy(),
                d in 0.1..5.0f64,
            ) {
                let result = offset_polygon(&poly, d);
                if !result.is_empty() {
                    let original = polygon::area(&poly);
                    let offset_a = polygon::area(&result[0]);
                    prop_assert!(offset_a >= original - 1e-6,
                        "original={}, offset={}", original, offset_a);
                }
            }

            #[test]
            fn prop_inward_offset_decreases_area(
                poly in convex_polygon_strategy(),
                d in 0.1..3.0f64,
            ) {
                let result = offset_polygon(&poly, -d);
                if !result.is_empty() {
                    let original = polygon::area(&poly);
                    let offset_a = polygon::area(&result[0]);
                    prop_assert!(offset_a <= original + 1e-6,
                        "original={}, offset={}", original, offset_a);
                }
            }

            #[test]
            fn prop_offset_vertices_are_finite(
                poly in convex_polygon_strategy(),
                d in -10.0..10.0f64,
            ) {
                let result = offset_polygon(&poly, d);
                for ring in &result {
                    for &(x, y) in ring {
                        prop_assert!(x.is_finite());
                        prop_assert!(y.is_finite());
                    }
                }
            }
        }
    }
}
