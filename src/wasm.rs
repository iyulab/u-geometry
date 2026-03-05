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

impl Point2D {
    fn to_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    fn from_tuple(t: (f64, f64)) -> Self {
        Self { x: t.0, y: t.1 }
    }
}

fn parse_points(js: JsValue) -> Result<Vec<Point2D>, JsValue> {
    serde_wasm_bindgen::from_value(js).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Computes the unsigned area of a simple polygon.
///
/// # Arguments
/// - `points_json`: JSON array of `{"x": f64, "y": f64}` objects
///
/// # Returns
/// Area as a non-negative `f64`.
#[wasm_bindgen]
pub fn polygon_area(points_json: JsValue) -> Result<f64, JsValue> {
    let points = parse_points(points_json)?;
    let tuples: Vec<(f64, f64)> = points.iter().map(|p| p.to_tuple()).collect();
    Ok(crate::polygon::area(&tuples))
}

/// Computes the convex hull of a set of points (Graham scan, CCW order).
///
/// # Arguments
/// - `points_json`: JSON array of `{"x": f64, "y": f64}` objects
///
/// # Returns
/// JSON array of hull points in CCW order, same `{"x", "y"}` format.
#[wasm_bindgen]
pub fn convex_hull(points_json: JsValue) -> Result<JsValue, JsValue> {
    let points = parse_points(points_json)?;
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
/// - `point_json`: `{"x": f64, "y": f64}`
/// - `polygon_json`: JSON array of `{"x": f64, "y": f64}` objects
///
/// # Returns
/// `true` if the point is inside or on the boundary.
#[wasm_bindgen]
pub fn point_in_polygon(point_json: JsValue, polygon_json: JsValue) -> Result<bool, JsValue> {
    let pt: Point2D = serde_wasm_bindgen::from_value(point_json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let polygon = parse_points(polygon_json)?;
    let tuples: Vec<(f64, f64)> = polygon.iter().map(|p| p.to_tuple()).collect();
    Ok(crate::polygon::contains_point(&tuples, pt.to_tuple()))
}
