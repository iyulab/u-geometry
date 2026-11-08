//! `polygons_overlap` answers the same in every unit: its default allowance for
//! touching is a fraction of the polygons' size, not a length of 1e-6 -- which
//! called two squares a micrometre wide (in metres) disjoint however far they
//! overlapped.

use u_geometry::collision::polygons_overlap;

/// Powers of two from about 1e-99 to 1e99, and the unit itself.
const EXPONENTS: [i32; 13] = [
    -330, -200, -100, -60, -40, -20, -10, 0, 10, 20, 40, 100, 330,
];

fn square(x: f64, y: f64, side: f64, s: f64) -> Vec<(f64, f64)> {
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        .iter()
        .map(|&(u, v)| ((x + u * side) * s, (y + v * side) * s))
        .collect()
}

#[test]
fn overlap_and_touching_do_not_depend_on_the_unit() {
    for e in EXPONENTS {
        let s = 2f64.powi(e);
        let a = square(0.0, 0.0, 10.0, s);
        assert!(
            polygons_overlap(&a, &square(5.0, 0.0, 10.0, s)),
            "half overlap at 2^{e}"
        );
        assert!(
            polygons_overlap(&a, &square(9.99, 0.0, 10.0, s)),
            "a overlap of a thousandth of the side at 2^{e}"
        );
        assert!(
            !polygons_overlap(&a, &square(10.0, 0.0, 10.0, s)),
            "touching squares at 2^{e}"
        );
        assert!(
            !polygons_overlap(&a, &square(12.0, 0.0, 10.0, s)),
            "apart at 2^{e}"
        );
        assert!(
            polygons_overlap(&a, &square(3.0, 3.0, 2.0, s)),
            "contained at 2^{e}"
        );
    }
}

/// A vector is normalized in every unit: only one with no length is the zero
/// vector (it used to be any vector shorter than 1e-15).
#[test]
fn normalized_does_not_depend_on_the_unit() {
    use u_geometry::primitives::{Vector2, Vector3};
    for e in EXPONENTS {
        let s = 2f64.powi(e);
        let v = Vector2::new(3.0 * s, 4.0 * s).normalized();
        assert_eq!((v.x, v.y), (0.6, 0.8), "Vector2 at 2^{e}");
        if e.abs() <= 100 {
            let w = Vector3::new(2.0 * s, 3.0 * s, 6.0 * s).normalized();
            assert!(
                (w.x - 2.0 / 7.0).abs() < 1e-15
                    && (w.y - 3.0 / 7.0).abs() < 1e-15
                    && (w.z - 6.0 / 7.0).abs() < 1e-15,
                "Vector3 at 2^{e}: {w:?}"
            );
        }
    }
    assert_eq!(Vector2::new(0.0, 0.0).normalized(), Vector2::ZERO);
}
