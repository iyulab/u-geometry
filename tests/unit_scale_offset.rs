//! Polygon offsetting is the same operation in every unit: scaling the polygon
//! and the distance by a power of two (exact in floating point) scales the
//! rings by the same factor, ring for ring.
//!
//! The degeneracy floors in `offset` used to be absolute (`1e-10` on a cross
//! product, an area and a length), so a part drawn in metres at the scale of a
//! micrometre feature was "parallel", "empty" or "a point" and the offset came
//! back wrong or empty.

use u_geometry::offset::{line_intersection, offset_edge, offset_polygon, offset_polygon_miter};

type Ring = Vec<(f64, f64)>;

/// Powers of two from about 1e-99 to 1e99, and the unit itself.
const EXPONENTS: [i32; 13] = [
    -330, -200, -100, -60, -40, -20, -10, 0, 10, 20, 40, 100, 330,
];

fn scaled(ring: &[(f64, f64)], s: f64) -> Ring {
    ring.iter().map(|&(x, y)| (x * s, y * s)).collect()
}

fn unscaled(rings: &[Ring], s: f64) -> Vec<Ring> {
    rings
        .iter()
        .map(|r| r.iter().map(|&(x, y)| (x / s, y / s)).collect())
        .collect()
}

fn shapes() -> Vec<(&'static str, Ring)> {
    vec![
        (
            "square",
            vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        ),
        (
            "l_shape",
            vec![
                (0.0, 0.0),
                (40.0, 0.0),
                (40.0, 10.0),
                (10.0, 10.0),
                (10.0, 30.0),
                (0.0, 30.0),
            ],
        ),
        ("triangle", vec![(0.0, 0.0), (12.0, 0.0), (3.0, 9.0)]),
    ]
}

#[test]
fn offset_polygon_does_not_depend_on_the_unit() {
    for (name, shape) in shapes() {
        for distance in [1.0, -1.0, 0.5, -0.25] {
            let base = offset_polygon(&shape, distance);
            assert!(
                !base.is_empty(),
                "{name} offset by {distance} is empty at the unit scale"
            );
            for e in EXPONENTS {
                let s = 2f64.powi(e);
                let got = offset_polygon(&scaled(&shape, s), distance * s);
                assert_eq!(
                    unscaled(&got, s),
                    base,
                    "{name} offset by {distance} at 2^{e}"
                );
            }
        }
    }
}

#[test]
fn offset_polygon_miter_does_not_depend_on_the_unit() {
    let spike = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (0.0, 0.5)];
    for limit in [1.0, 2.0, 5.0] {
        let base = offset_polygon_miter(&spike, 0.2, limit);
        assert!(!base.is_empty());
        for e in EXPONENTS {
            let s = 2f64.powi(e);
            let got = offset_polygon_miter(&scaled(&spike, s), 0.2 * s, limit);
            assert_eq!(unscaled(&got, s), base, "miter limit {limit} at 2^{e}");
        }
    }
}

#[test]
fn offset_edge_does_not_depend_on_the_unit() {
    let (p0, p1) = ((1.0, 2.0), (4.0, 6.0));
    let base = offset_edge(p0, p1, 0.5);
    assert_ne!(base, (p0, p1), "the edge moves at the unit scale");
    for e in EXPONENTS {
        let s = 2f64.powi(e);
        let ((a, b), (c, d)) = offset_edge((p0.0 * s, p0.1 * s), (p1.0 * s, p1.1 * s), 0.5 * s);
        assert_eq!(
            ((a / s, b / s), (c / s, d / s)),
            base,
            "an edge of length 5 offset by 0.5 at 2^{e}"
        );
    }
}

#[test]
fn line_intersection_does_not_depend_on_the_unit() {
    let base = line_intersection((0.0, 0.0), (1.0, 0.0), (3.0, -2.0), (0.0, 1.0))
        .expect("perpendicular lines meet");
    assert_eq!(base, (3.0, 0.0));
    for e in EXPONENTS {
        let s = 2f64.powi(e);
        // Directions scale with the unit as well: they are edge vectors.
        let got = line_intersection((0.0, 0.0), (s, 0.0), (3.0 * s, -2.0 * s), (0.0, s))
            .unwrap_or_else(|| panic!("perpendicular lines are 'parallel' at 2^{e}"));
        assert_eq!((got.0 / s, got.1 / s), base, "at 2^{e}");
    }
    // Parallel lines never meet, whatever the unit.
    for e in EXPONENTS {
        let s = 2f64.powi(e);
        assert!(
            line_intersection((0.0, 0.0), (s, s), (0.0, s), (2.0 * s, 2.0 * s)).is_none(),
            "parallel lines meet at 2^{e}"
        );
    }
}
