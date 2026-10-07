//! WASM bindings for u-geometry.
//!
//! Exposes core polygon operations to JavaScript via `wasm-bindgen`.
//! Only compiled when the `wasm` feature is enabled.
//!
//! # Usage (JavaScript)
//! ```js
//! import { polygon_area, convex_hull, point_in_polygon } from '@iyulab/u-geometry';
//! const area = polygon_area([{x: 0, y: 0}, {x: 1, y: 0}, {x: 0, y: 1}]);
//! ```
//!
//! Every refusal throws an `Error` whose `message` is readable text and which
//! carries `code` -- a stable reason -- and `parameter`, the argument it is
//! about. See the README's *Errors*.

#![cfg(feature = "wasm")]

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct Point2D {
    x: f64,
    y: f64,
}

/// Axis-aligned bounding box, serialized as `{"min": {x, y}, "max": {x, y}}`.
#[derive(Serialize, Deserialize, tsify::Tsify)]
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

/// A refusal: an `Error` whose `message` is `message`, with a stable `code`
/// and the `parameter` it is about set on it as properties.
fn refuse(code: &str, parameter: &str, message: String) -> JsValue {
    let err = js_sys::Error::new(&message);
    // `Reflect::set` on a freshly created ordinary object cannot fail.
    let _ = js_sys::Reflect::set(&err, &"code".into(), &code.into());
    let _ = js_sys::Reflect::set(&err, &"parameter".into(), &parameter.into());
    err.into()
}

/// `value_not_finite` for a NaN or ±Infinity, with the path to it as
/// `parameter` and its array position as `index` (`null` when not an element).
fn refuse_non_finite(found: &NonFinite) -> JsValue {
    let err = refuse("value_not_finite", &found.parameter, found.message());
    let index = found
        .index
        .map_or(JsValue::NULL, |i| JsValue::from(i as u32));
    let _ = js_sys::Reflect::set(&err, &"index".into(), &index);
    err
}

/// A number argument passed directly (not inside an object or array).
fn finite_arg(value: f64, parameter: &str) -> Result<f64, JsValue> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(refuse_non_finite(&NonFinite {
            parameter: parameter.to_string(),
            index: None,
            value,
        }))
    }
}

/// A NaN or ±Infinity found in a JS argument, and where it sits.
///
/// JSON has no non-finite numbers, so on the way to the wire schema
/// `serde_json` turns one into `null` and the caller would be told a value has
/// the wrong type. [`find_non_finite`] looks before that happens, so the
/// refusal names the real reason and the place.
struct NonFinite {
    /// The argument's name, then `.key` and `[i]` steps down to the array or
    /// field that holds the number.
    parameter: String,
    /// The number's position, when it is an array element.
    index: Option<usize>,
    value: f64,
}

impl NonFinite {
    fn message(&self) -> String {
        let at = match self.index {
            Some(i) => format!("{}[{i}]", self.parameter),
            None => self.parameter.clone(),
        };
        let got = if self.value.is_nan() {
            "NaN"
        } else if self.value > 0.0 {
            "Infinity"
        } else {
            "-Infinity"
        };
        format!("{at}: expected a finite number, got {got}")
    }
}

/// The first NaN or ±Infinity in `value`, searching arrays, iterables and
/// plain objects. `allow_nan` lets NaN through for an input that reads it as a
/// missing value; it then arrives as `null`.
fn find_non_finite(value: &JsValue, parameter: &str, allow_nan: bool) -> Option<NonFinite> {
    let refused = |n: f64| !n.is_finite() && !(allow_nan && n.is_nan());
    let found = |index: Option<usize>, value: f64| NonFinite {
        parameter: parameter.to_string(),
        index,
        value,
    };
    if let Some(n) = value.as_f64() {
        return refused(n).then(|| found(None, n));
    }
    if !value.is_object() {
        return None;
    }
    if let Ok(Some(items)) = js_sys::try_iter(value) {
        for (i, item) in items.enumerate() {
            // An iterator that throws is left for serde to report.
            let item = item.ok()?;
            match item.as_f64() {
                Some(n) if refused(n) => return Some(found(Some(i), n)),
                Some(_) => {}
                None => {
                    let inner = find_non_finite(&item, &format!("{parameter}[{i}]"), allow_nan);
                    if inner.is_some() {
                        return inner;
                    }
                }
            }
        }
        return None;
    }
    let object: &js_sys::Object = wasm_bindgen::JsCast::unchecked_ref(value);
    for entry in js_sys::Object::entries(object).iter() {
        let pair: js_sys::Array = wasm_bindgen::JsCast::unchecked_into(entry);
        let key = pair.get(0).as_string().unwrap_or_default();
        let inner = find_non_finite(&pair.get(1), &format!("{parameter}.{key}"), allow_nan);
        if inner.is_some() {
            return inner;
        }
    }
    None
}

/// Serializes a response; a failure is reported rather than unwrapped.
fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value)
        .map_err(|e| refuse("malformed_input", "result", e.to_string()))
}

/// Deserialize a native JS value, rejecting JSON strings with an actionable
/// message and prefixing the offending parameter name to any serde error.
fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, param: &str) -> Result<T, JsValue> {
    if value.as_string().is_some() {
        return Err(refuse(
            "malformed_input",
            param,
            format!(
                "{param}: expected a native JS object/array, got a string — \
                 pass the value directly, not JSON.stringify(...)"
            ),
        ));
    }
    if let Some(found) = find_non_finite(&value, param, false) {
        return Err(refuse_non_finite(&found));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value = serde_wasm_bindgen::from_value(value)
        .map_err(|e| refuse("malformed_input", param, format!("{param}: {e}")))?;
    serde_path_to_error::deserialize(json).map_err(|e| {
        let (parameter, index) = failure_site(param, e.path());
        let err = refuse(
            "malformed_input",
            &parameter,
            format!("{param}: {}: {}", e.path(), e.inner()),
        );
        let index = index.map_or(JsValue::NULL, |i| JsValue::from(i as u32));
        // `Reflect::set` on a freshly created ordinary object cannot fail.
        let _ = js_sys::Reflect::set(&err, &"index".into(), &index);
        err
    })
}

/// Where in the argument `param` a request stopped deserializing, as the
/// refusal reports it: the field (`design[1]`, `points[0].y`) and, when the
/// failure sits in an array, its position there -- the same `parameter` /
/// `index` a number array read element by element reports. Without it a
/// `null` three levels down was "invalid type: null, expected f64" with no
/// way to say which row (the gap `read_numbers` closed for bare arrays).
fn failure_site(param: &str, path: &serde_path_to_error::Path) -> (String, Option<usize>) {
    use serde_path_to_error::Segment;
    let segments: Vec<&Segment> = path.iter().collect();
    let index = segments.iter().rev().find_map(|s| match s {
        Segment::Seq { index } => Some(*index),
        _ => None,
    });
    // A trailing `[i]` is the index, not part of the name.
    let named = match segments.last() {
        Some(Segment::Seq { .. }) => &segments[..segments.len() - 1],
        _ => &segments[..],
    };
    let mut name = String::new();
    for segment in named {
        match segment {
            Segment::Seq { index } => name.push_str(&format!("[{index}]")),
            Segment::Map { key } | Segment::Enum { variant: key } => {
                if !name.is_empty() {
                    name.push('.');
                }
                name.push_str(key);
            }
            Segment::Unknown => name.push_str(".?"),
        }
    }
    // A top-level array argument (`data[1][2]`) or a failure at the root
    // (a missing field) is named by the argument itself.
    if name.is_empty() || name.starts_with('[') {
        name.insert_str(0, param);
    }
    (name, index)
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
pub fn polygon_area(
    #[wasm_bindgen(unchecked_param_type = "Point2D[]")] points: JsValue,
) -> Result<f64, JsValue> {
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
#[wasm_bindgen(unchecked_return_type = "Point2D[]")]
pub fn convex_hull(
    #[wasm_bindgen(unchecked_param_type = "Point2D[]")] points: JsValue,
) -> Result<JsValue, JsValue> {
    let points = parse_points(points, "points")?;
    let tuples: Vec<(f64, f64)> = points.iter().map(|p| p.to_tuple()).collect();
    let hull = crate::polygon::convex_hull(&tuples);
    let result: Vec<Point2D> = hull.into_iter().map(Point2D::from_tuple).collect();
    to_js(&result)
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
pub fn point_in_polygon(
    #[wasm_bindgen(unchecked_param_type = "Point2D")] point: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Point2D[]")] polygon: JsValue,
) -> Result<bool, JsValue> {
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
pub fn polygons_intersect(
    #[wasm_bindgen(unchecked_param_type = "Point2D[]")] poly_a: JsValue,
    #[wasm_bindgen(unchecked_param_type = "Point2D[]")] poly_b: JsValue,
) -> Result<bool, JsValue> {
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
/// `{"min": {x, y}, "max": {x, y}}`; throws `empty_input` if `points` is empty.
#[wasm_bindgen(unchecked_return_type = "Aabb")]
pub fn polygon_bounds(
    #[wasm_bindgen(unchecked_param_type = "Point2D[]")] points: JsValue,
) -> Result<JsValue, JsValue> {
    let points = parse_points(points, "points")?;
    if points.is_empty() {
        return Err(refuse(
            "empty_input",
            "points",
            "points: expected a non-empty array".to_string(),
        ));
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
    to_js(&aabb)
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
#[wasm_bindgen(unchecked_return_type = "Point2D[]")]
pub fn transform_points(
    #[wasm_bindgen(unchecked_param_type = "Point2D[]")] points: JsValue,
    tx: f64,
    ty: f64,
    angle: f64,
) -> Result<JsValue, JsValue> {
    let points = parse_points(points, "points")?;
    let (tx, ty, angle) = (
        finite_arg(tx, "tx")?,
        finite_arg(ty, "ty")?,
        finite_arg(angle, "angle")?,
    );
    let tuples: Vec<(f64, f64)> = points.iter().map(|p| p.to_tuple()).collect();
    let t = crate::transform::Transform2D::new(tx, ty, angle);
    let out: Vec<Point2D> = t
        .apply_points(&tuples)
        .into_iter()
        .map(Point2D::from_tuple)
        .collect();
    to_js(&out)
}

#[cfg(test)]
mod path_tests {
    //! A value of the wrong type deep inside an argument is refused where it
    //! sits: `parameter` names the field, `index` its array position.

    #[test]
    fn a_bad_coordinate_is_named_at_its_point() {
        let e = serde_path_to_error::deserialize::<_, Vec<super::Point2D>>(serde_json::json!([
            { "x": 0.0, "y": 0.0 },
            { "x": 1.0, "y": null }
        ]))
        .err()
        .expect("null is not a number");
        assert_eq!(
            super::failure_site("points", e.path()),
            ("points[1].y".to_string(), Some(1))
        );
        // A top-level array element and a missing field.
        let e = serde_path_to_error::deserialize::<_, Vec<f64>>(serde_json::json!([1.0, "2"]))
            .expect_err("a string is not a number");
        assert_eq!(
            super::failure_site("values", e.path()),
            ("values".to_string(), Some(1))
        );
        let e =
            serde_path_to_error::deserialize::<_, super::Point2D>(serde_json::json!({ "x": 1.0 }))
                .err()
                .expect("y is required");
        assert_eq!(
            super::failure_site("point", e.path()),
            ("point".to_string(), None)
        );
    }
}
