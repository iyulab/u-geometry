//! Criterion benchmarks for u-geometry core operations.
//!
//! Covers polygon operations, collision detection, spatial indexing,
//! Minkowski sum/NFP, and robust predicates.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use u_geometry::collision;
use u_geometry::minkowski;
use u_geometry::polygon;
use u_geometry::primitives::{Point2, AABB2, AABB3};
use u_geometry::robust;
use u_geometry::spatial_index::{SpatialIndex2D, SpatialIndex3D};
use u_geometry::transform::Transform2D;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn regular_polygon(n: usize, radius: f64) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            (radius * angle.cos(), radius * angle.sin())
        })
        .collect()
}

fn random_points(n: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let px = (x >> 33) as f64 / (1u64 << 31) as f64 * 1000.0;
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let py = (x >> 33) as f64 / (1u64 << 31) as f64 * 1000.0;
            (px, py)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Polygon benchmarks
// ---------------------------------------------------------------------------

fn bench_polygon_area(c: &mut Criterion) {
    let mut group = c.benchmark_group("polygon_area");
    for n in [10, 100, 1000] {
        let poly = regular_polygon(n, 100.0);
        group.bench_with_input(BenchmarkId::from_parameter(n), &poly, |b, p| {
            b.iter(|| polygon::area(black_box(p)))
        });
    }
    group.finish();
}

fn bench_polygon_centroid(c: &mut Criterion) {
    let mut group = c.benchmark_group("polygon_centroid");
    for n in [10, 100, 1000] {
        let poly = regular_polygon(n, 100.0);
        group.bench_with_input(BenchmarkId::from_parameter(n), &poly, |b, p| {
            b.iter(|| polygon::centroid(black_box(p)))
        });
    }
    group.finish();
}

fn bench_convex_hull(c: &mut Criterion) {
    let mut group = c.benchmark_group("convex_hull");
    for n in [50, 200, 1000] {
        let points = random_points(n, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &points, |b, pts| {
            b.iter(|| polygon::convex_hull(black_box(pts)))
        });
    }
    group.finish();
}

fn bench_point_in_polygon(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_in_polygon");
    for n in [10, 100, 1000] {
        let poly = regular_polygon(n, 100.0);
        let point = (50.0, 50.0);
        group.bench_with_input(BenchmarkId::from_parameter(n), &poly, |b, p| {
            b.iter(|| polygon::contains_point(black_box(p), black_box(point)))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Collision detection benchmarks
// ---------------------------------------------------------------------------

fn bench_sat_collision(c: &mut Criterion) {
    let mut group = c.benchmark_group("sat_collision");
    for n in [4, 8, 20] {
        let poly_a = regular_polygon(n, 50.0);
        let poly_b: Vec<(f64, f64)> = regular_polygon(n, 30.0)
            .into_iter()
            .map(|(x, y)| (x + 40.0, y))
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(poly_a.clone(), poly_b.clone()),
            |b, (a, bb)| b.iter(|| collision::polygons_overlap(black_box(a), black_box(bb))),
        );
    }
    group.finish();
}

fn bench_overlap_depth(c: &mut Criterion) {
    let poly_a = regular_polygon(8, 50.0);
    let poly_b: Vec<(f64, f64)> = regular_polygon(8, 30.0)
        .into_iter()
        .map(|(x, y)| (x + 40.0, y))
        .collect();
    c.bench_function("overlap_depth_octagon", |b| {
        b.iter(|| collision::overlap_depth(black_box(&poly_a), black_box(&poly_b)))
    });
}

fn bench_aabb_overlap(c: &mut Criterion) {
    let a = AABB2::new(0.0, 0.0, 100.0, 100.0);
    let b = AABB2::new(50.0, 50.0, 150.0, 150.0);
    c.bench_function("aabb2_overlap", |b_iter| {
        b_iter.iter(|| collision::aabb_overlap(black_box(&a), black_box(&b)))
    });
}

// ---------------------------------------------------------------------------
// Spatial index benchmarks
// ---------------------------------------------------------------------------

fn bench_spatial_index_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_index_2d");

    for n in [100, 500, 1000] {
        // Insert
        group.bench_with_input(BenchmarkId::new("insert", n), &n, |b, &n| {
            b.iter(|| {
                let mut idx = SpatialIndex2D::new();
                for i in 0..n {
                    let x = (i * 10) as f64;
                    let y = (i * 7) as f64;
                    idx.insert(i, AABB2::new(x, y, x + 10.0, y + 10.0));
                }
                black_box(idx)
            })
        });

        // Query after insertion
        let mut idx = SpatialIndex2D::new();
        for i in 0..n {
            let x = (i * 10) as f64;
            let y = (i * 7) as f64;
            idx.insert(i, AABB2::new(x, y, x + 10.0, y + 10.0));
        }
        let query_region = AABB2::new(0.0, 0.0, 100.0, 100.0);
        group.bench_with_input(BenchmarkId::new("query", n), &idx, |b, idx| {
            b.iter(|| idx.query(black_box(&query_region)))
        });
    }
    group.finish();
}

fn bench_spatial_index_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_index_3d");
    for n in [100, 500] {
        let mut idx = SpatialIndex3D::new();
        for i in 0..n {
            let x = (i * 10) as f64;
            let y = (i * 7) as f64;
            let z = (i * 5) as f64;
            idx.insert(i, AABB3::new(x, y, z, x + 10.0, y + 10.0, z + 10.0));
        }
        let query_region = AABB3::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0);
        group.bench_with_input(BenchmarkId::new("query", n), &idx, |b, idx| {
            b.iter(|| idx.query(black_box(&query_region)))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Minkowski sum / NFP benchmarks
// ---------------------------------------------------------------------------

fn bench_minkowski_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("minkowski_sum");
    for n in [4, 8, 16] {
        let p = regular_polygon(n, 50.0);
        let q = regular_polygon(n, 30.0);
        group.bench_with_input(BenchmarkId::from_parameter(n), &(p, q), |b, (p, q)| {
            b.iter(|| minkowski::minkowski_sum_convex(black_box(p), black_box(q)))
        });
    }
    group.finish();
}

fn bench_nfp_convex(c: &mut Criterion) {
    let mut group = c.benchmark_group("nfp_convex");
    for n in [4, 8, 16] {
        let stationary = regular_polygon(n, 50.0);
        let orbiting = regular_polygon(n, 30.0);
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(stationary, orbiting),
            |b, (s, o)| b.iter(|| minkowski::nfp_convex(black_box(s), black_box(o))),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Robust predicates benchmarks
// ---------------------------------------------------------------------------

fn bench_orient2d(c: &mut Criterion) {
    c.bench_function("orient2d", |b| {
        b.iter(|| {
            robust::orient2d(
                black_box((0.0, 0.0)),
                black_box((1.0, 0.0)),
                black_box((0.5, 1e-15)),
            )
        })
    });
}

fn bench_point_in_triangle(c: &mut Criterion) {
    c.bench_function("point_in_triangle", |b| {
        b.iter(|| {
            robust::point_in_triangle(
                black_box((0.3, 0.3)),
                black_box((0.0, 0.0)),
                black_box((1.0, 0.0)),
                black_box((0.5, 1.0)),
            )
        })
    });
}

fn bench_is_convex(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_convex");
    for n in [10, 100, 1000] {
        let poly = regular_polygon(n, 100.0);
        group.bench_with_input(BenchmarkId::from_parameter(n), &poly, |b, p| {
            b.iter(|| robust::is_convex(black_box(p)))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Transform benchmarks
// ---------------------------------------------------------------------------

fn bench_transform_apply(c: &mut Criterion) {
    let t = Transform2D::new(10.0, 20.0, std::f64::consts::FRAC_PI_4);
    let point = Point2::new(50.0, 30.0);
    c.bench_function("transform2d_apply_point", |b| {
        b.iter(|| t.apply_point(black_box(&point)))
    });
}

fn bench_transform_batch(c: &mut Criterion) {
    let t = Transform2D::new(10.0, 20.0, std::f64::consts::FRAC_PI_4);
    let points: Vec<(f64, f64)> = (0..100).map(|i| (i as f64, i as f64 * 2.0)).collect();
    c.bench_function("transform2d_apply_100_points", |b| {
        b.iter(|| t.apply_points(black_box(&points)))
    });
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_polygon_area,
    bench_polygon_centroid,
    bench_convex_hull,
    bench_point_in_polygon,
    bench_sat_collision,
    bench_overlap_depth,
    bench_aabb_overlap,
    bench_spatial_index_2d,
    bench_spatial_index_3d,
    bench_minkowski_sum,
    bench_nfp_convex,
    bench_orient2d,
    bench_point_in_triangle,
    bench_is_convex,
    bench_transform_apply,
    bench_transform_batch,
);
criterion_main!(benches);
