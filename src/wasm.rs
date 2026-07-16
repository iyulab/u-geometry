//! WASM bindings for u-geometry.
//!
//! Exposes core polygon operations to JavaScript via `wasm-bindgen`.
//! Only compiled when the `wasm` feature is enabled.
//!
//! # Usage (JavaScript)
//! ```js
//! import init, { polygon_area, convex_hull, point_in_polygon } from '@iyulab/u-geometry';
//! await init();
//! const area = polygon_area([{x: 0, y: 0}, {x: 1, y: 0}, {x: 0, y: 1}]);
//! ```

#![cfg(feature = "wasm")]

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
struct Point2D {
    x: f64,
    y: f64,
}

/// Axis-aligned bounding box, serialized as `{"min": {x, y}, "max": {x, y}}`.
#[derive(Serialize, Deserialize)]
struct Aabb {
    min: Point2D,
    max: Point2D,
}

impl Point2D {
    fn to_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    fn from_tuple(t: (f64, f64)) -> Self {
        Self { x: t.0, y: t.1 }
    }
}

/// Deserialize a native JS value, rejecting JSON strings with an actionable
/// message and prefixing the offending parameter name to any serde error.
fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, param: &str) -> Result<T, JsValue> {
    if value.as_string().is_some() {
        return Err(JsValue::from_str(&format!(
            "{param}: expected a native JS object/array, got a string — \
             pass the value directly, not JSON.stringify(...)"
        )));
    }
    serde_wasm_bindgen::from_value(value).map_err(|e| JsValue::from_str(&format!("{param}: {e}")))
}

fn parse_points(js: JsValue, param: &str) -> Result<Vec<Point2D>, JsValue> {
    from_js(js, param)
}

/// Computes the unsigned area of a simple polygon.
///
/// # Arguments
/// - `points`: native array of `{"x": f64, "y": f64}` objects
///
/// # Returns
/// Area as a non-negative `f64`.
#[wasm_bindgen]
pub fn polygon_area(points: JsValue) -> Result<f64, JsValue> {
    let points = parse_points(points, "points")?;
    let tuples: Vec<(f64, f64)> = points.iter().map(|p| p.to_tuple()).collect();
    Ok(crate::polygon::area(&tuples))
}

/// Computes the convex hull of a set of points (Graham scan, CCW order).
///
/// # Arguments
/// - `points`: native array of `{"x": f64, "y": f64}` objects
///
/// # Returns
/// Array of hull points in CCW order, same `{"x", "y"}` format.
#[wasm_bindgen]
pub fn convex_hull(points: JsValue) -> Result<JsValue, JsValue> {
    let points = parse_points(points, "points")?;
    let tuples: Vec<(f64, f64)> = points.iter().map(|p| p.to_tuple()).collect();
    let hull = crate::polygon::convex_hull(&tuples);
    let result: Vec<Point2D> = hull.into_iter().map(Point2D::from_tuple).collect();
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Tests whether a point lies inside (or on the boundary of) a simple polygon.
///
/// Uses the ray-casting (winding-number) test.
///
/// # Arguments
/// - `point`: native `{"x": f64, "y": f64}` object
/// - `polygon`: native array of `{"x": f64, "y": f64}` objects
///
/// # Returns
/// `true` if the point is inside or on the boundary.
#[wasm_bindgen]
pub fn point_in_polygon(point: JsValue, polygon: JsValue) -> Result<bool, JsValue> {
    let pt: Point2D = from_js(point, "point")?;
    let polygon = parse_points(polygon, "polygon")?;
    let tuples: Vec<(f64, f64)> = polygon.iter().map(|p| p.to_tuple()).collect();
    Ok(crate::polygon::contains_point(&tuples, pt.to_tuple()))
}

/// Tests whether two **simple polygons** (convex or concave) overlap.
///
/// Exact for concave shapes: two polygons that merely abut (share an edge or a
/// vertex) do **not** count as overlapping, and a part nested inside another's
/// concave notch is correctly reported as *not* overlapping. Use this rather
/// than a convex-hull (SAT) test for nesting/packing self-checks.
///
/// # Arguments
/// - `poly_a`: native array of `{"x": f64, "y": f64}` objects
/// - `poly_b`: native array of `{"x": f64, "y": f64}` objects
///
/// # Returns
/// `true` if the polygon interiors overlap; `false` when disjoint or merely touching.
#[wasm_bindgen]
pub fn polygons_intersect(poly_a: JsValue, poly_b: JsValue) -> Result<bool, JsValue> {
    let a = parse_points(poly_a, "poly_a")?;
    let b = parse_points(poly_b, "poly_b")?;
    let ta: Vec<(f64, f64)> = a.iter().map(|p| p.to_tuple()).collect();
    let tb: Vec<(f64, f64)> = b.iter().map(|p| p.to_tuple()).collect();
    Ok(crate::collision::polygons_intersect(&ta, &tb))
}

/// Computes the axis-aligned bounding box of a set of points.
///
/// # Arguments
/// - `points`: native array of `{"x": f64, "y": f64}` objects (must be non-empty)
///
/// # Returns
/// `{"min": {x, y}, "max": {x, y}}`, or an error if `points` is empty.
#[wasm_bindgen]
pub fn polygon_bounds(points: JsValue) -> Result<JsValue, JsValue> {
    let points = parse_points(points, "points")?;
    if points.is_empty() {
        return Err(JsValue::from_str("points: expected a non-empty array"));
    }
    let mut min_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_x = points[0].x;
    let mut max_y = points[0].y;
    for p in points.iter().skip(1) {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    let aabb = Aabb {
        min: Point2D { x: min_x, y: min_y },
        max: Point2D { x: max_x, y: max_y },
    };
    serde_wasm_bindgen::to_value(&aabb).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Applies a rigid 2D transform (rotation about the origin, then translation)
/// to a set of points.
///
/// This is a pure isometry: `angle` is in **radians** and there is no reflection.
/// Domain-specific conventions (degrees, mirroring/flip, a different pivot) are
/// the caller's responsibility — compose them before/after this call.
///
/// # Arguments
/// - `points`: native array of `{"x": f64, "y": f64}` objects
/// - `tx`, `ty`: translation applied after rotation
/// - `angle`: rotation angle in radians (counter-clockwise)
///
/// # Returns
/// Array of transformed points in the same `{"x", "y"}` format.
#[wasm_bindgen]
pub fn transform_points(points: JsValue, tx: f64, ty: f64, angle: f64) -> Result<JsValue, JsValue> {
    let points = parse_points(points, "points")?;
    let tuples: Vec<(f64, f64)> = points.iter().map(|p| p.to_tuple()).collect();
    let t = crate::transform::Transform2D::new(tx, ty, angle);
    let out: Vec<Point2D> = t
        .apply_points(&tuples)
        .into_iter()
        .map(Point2D::from_tuple)
        .collect();
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}
