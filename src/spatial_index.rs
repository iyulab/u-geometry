//! Spatial indexing for efficient geometric queries.
//!
//! Provides a simple linear-scan spatial index suitable for small to medium
//! object counts (<1000). For larger datasets, consumers should implement
//! tree-based structures (BVH, R-tree, Octree) using the same API patterns.
//!
//! # Design
//!
//! The spatial index is deliberately simple: a flat `Vec` of (key, AABB) pairs
//! with linear-scan queries. This outperforms tree-based structures for small N
//! due to cache locality and zero overhead.
//!
//! For larger N (>1000), tree-based indices amortize O(log n) queries, but
//! for typical bin-packing/nesting instances (10-500 items), linear scan wins.
//!
//! # References
//!
//! - Ericson (2005), "Real-Time Collision Detection", Ch. 6 (BVH)
//! - Akenine-Möller et al. (2018), "Real-Time Rendering", Ch. 25.1 (Spatial Indexing)

use crate::primitives::{AABB2, AABB3};

/// A 2D spatial index entry.
#[derive(Debug, Clone)]
struct Entry2D {
    key: usize,
    bounds: AABB2,
}

/// A linear-scan 2D spatial index.
///
/// Stores (key, AABB2) pairs and answers overlap queries via brute-force scan.
///
/// # Complexity
/// - Insert: O(1) amortized
/// - Query: O(n)
/// - Remove: O(n)
///
/// Optimal for N < 1000 due to cache locality.
#[derive(Debug, Clone, Default)]
pub struct SpatialIndex2D {
    entries: Vec<Entry2D>,
}

impl SpatialIndex2D {
    /// Creates an empty spatial index.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Creates with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Inserts an entry.
    ///
    /// # Complexity
    /// O(1) amortized
    pub fn insert(&mut self, key: usize, bounds: AABB2) {
        self.entries.push(Entry2D { key, bounds });
    }

    /// Removes all entries with the given key.
    ///
    /// # Complexity
    /// O(n)
    pub fn remove(&mut self, key: usize) {
        self.entries.retain(|e| e.key != key);
    }

    /// Returns all keys whose AABB overlaps the query region.
    ///
    /// # Complexity
    /// O(n)
    pub fn query(&self, region: &AABB2) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|e| e.bounds.intersects(region))
            .map(|e| e.key)
            .collect()
    }

    /// Returns all keys whose AABB overlaps the query region, excluding a key.
    ///
    /// Useful for "find all neighbors except self" queries.
    ///
    /// # Complexity
    /// O(n)
    pub fn query_except(&self, region: &AABB2, exclude: usize) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|e| e.key != exclude && e.bounds.intersects(region))
            .map(|e| e.key)
            .collect()
    }

    /// Checks if any entry overlaps the query region.
    ///
    /// # Complexity
    /// O(n) worst case, but short-circuits on first hit.
    pub fn has_collision(&self, region: &AABB2) -> bool {
        self.entries.iter().any(|e| e.bounds.intersects(region))
    }

    /// Checks if any entry (except the excluded key) overlaps the query region.
    pub fn has_collision_except(&self, region: &AABB2, exclude: usize) -> bool {
        self.entries
            .iter()
            .any(|e| e.key != exclude && e.bounds.intersects(region))
    }

    /// Removes all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// A 3D spatial index entry.
#[derive(Debug, Clone)]
struct Entry3D {
    key: usize,
    bounds: AABB3,
}

/// A linear-scan 3D spatial index.
///
/// Stores (key, AABB3) pairs and answers overlap queries via brute-force scan.
///
/// # Complexity
/// - Insert: O(1) amortized
/// - Query: O(n)
/// - Remove: O(n)
///
/// Optimal for N < 1000 due to cache locality.
///
/// # Reference
/// Ericson (2005), "Real-Time Collision Detection", Ch. 6.1
#[derive(Debug, Clone, Default)]
pub struct SpatialIndex3D {
    entries: Vec<Entry3D>,
}

impl SpatialIndex3D {
    /// Creates an empty spatial index.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Creates with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Inserts an entry.
    ///
    /// # Complexity
    /// O(1) amortized
    pub fn insert(&mut self, key: usize, bounds: AABB3) {
        self.entries.push(Entry3D { key, bounds });
    }

    /// Removes all entries with the given key.
    ///
    /// # Complexity
    /// O(n)
    pub fn remove(&mut self, key: usize) {
        self.entries.retain(|e| e.key != key);
    }

    /// Returns all keys whose AABB overlaps the query region.
    ///
    /// # Complexity
    /// O(n)
    pub fn query(&self, region: &AABB3) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|e| e.bounds.intersects(region))
            .map(|e| e.key)
            .collect()
    }

    /// Returns all keys whose AABB overlaps the query region, excluding a key.
    ///
    /// # Complexity
    /// O(n)
    pub fn query_except(&self, region: &AABB3, exclude: usize) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|e| e.key != exclude && e.bounds.intersects(region))
            .map(|e| e.key)
            .collect()
    }

    /// Checks if any entry overlaps the query region.
    ///
    /// # Complexity
    /// O(n) worst case, but short-circuits on first hit.
    pub fn has_collision(&self, region: &AABB3) -> bool {
        self.entries.iter().any(|e| e.bounds.intersects(region))
    }

    /// Checks if any entry (except the excluded key) overlaps the query region.
    pub fn has_collision_except(&self, region: &AABB3, exclude: usize) -> bool {
        self.entries
            .iter()
            .any(|e| e.key != exclude && e.bounds.intersects(region))
    }

    /// Returns the AABB for a given key, if found.
    pub fn get_bounds(&self, key: usize) -> Option<&AABB3> {
        self.entries
            .iter()
            .find(|e| e.key == key)
            .map(|e| &e.bounds)
    }

    /// Removes all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======================== 2D Spatial Index Tests ========================

    #[test]
    fn test_2d_insert_and_query() {
        let mut idx = SpatialIndex2D::new();
        idx.insert(0, AABB2::new(0.0, 0.0, 10.0, 10.0));
        idx.insert(1, AABB2::new(20.0, 20.0, 30.0, 30.0));
        idx.insert(2, AABB2::new(5.0, 5.0, 15.0, 15.0));

        let hits = idx.query(&AABB2::new(8.0, 8.0, 12.0, 12.0));
        assert!(hits.contains(&0)); // overlaps
        assert!(hits.contains(&2)); // overlaps
        assert!(!hits.contains(&1)); // no overlap
    }

    #[test]
    fn test_2d_query_except() {
        let mut idx = SpatialIndex2D::new();
        idx.insert(0, AABB2::new(0.0, 0.0, 10.0, 10.0));
        idx.insert(1, AABB2::new(5.0, 5.0, 15.0, 15.0));

        let hits = idx.query_except(&AABB2::new(8.0, 8.0, 12.0, 12.0), 0);
        assert!(!hits.contains(&0));
        assert!(hits.contains(&1));
    }

    #[test]
    fn test_2d_has_collision() {
        let mut idx = SpatialIndex2D::new();
        idx.insert(0, AABB2::new(0.0, 0.0, 10.0, 10.0));

        assert!(idx.has_collision(&AABB2::new(5.0, 5.0, 15.0, 15.0)));
        assert!(!idx.has_collision(&AABB2::new(20.0, 20.0, 30.0, 30.0)));
    }

    #[test]
    fn test_2d_remove() {
        let mut idx = SpatialIndex2D::new();
        idx.insert(0, AABB2::new(0.0, 0.0, 10.0, 10.0));
        idx.insert(1, AABB2::new(5.0, 5.0, 15.0, 15.0));

        assert_eq!(idx.len(), 2);
        idx.remove(0);
        assert_eq!(idx.len(), 1);
        assert!(!idx.has_collision(&AABB2::new(0.0, 0.0, 4.0, 4.0)));
    }

    #[test]
    fn test_2d_clear() {
        let mut idx = SpatialIndex2D::new();
        idx.insert(0, AABB2::new(0.0, 0.0, 10.0, 10.0));
        idx.clear();
        assert!(idx.is_empty());
    }

    #[test]
    fn test_2d_empty_query() {
        let idx = SpatialIndex2D::new();
        assert!(idx.query(&AABB2::new(0.0, 0.0, 10.0, 10.0)).is_empty());
        assert!(!idx.has_collision(&AABB2::new(0.0, 0.0, 10.0, 10.0)));
    }

    // ======================== 3D Spatial Index Tests ========================

    fn box3d(x: f64, y: f64, z: f64, w: f64, d: f64, h: f64) -> AABB3 {
        AABB3::new(x, y, z, x + w, y + d, z + h)
    }

    #[test]
    fn test_3d_insert_and_query() {
        let mut idx = SpatialIndex3D::new();
        idx.insert(0, box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0));
        idx.insert(1, box3d(20.0, 20.0, 20.0, 10.0, 10.0, 10.0));
        idx.insert(2, box3d(5.0, 5.0, 5.0, 10.0, 10.0, 10.0));

        let hits = idx.query(&box3d(8.0, 8.0, 8.0, 4.0, 4.0, 4.0));
        assert!(hits.contains(&0));
        assert!(hits.contains(&2));
        assert!(!hits.contains(&1));
    }

    #[test]
    fn test_3d_query_except() {
        let mut idx = SpatialIndex3D::new();
        idx.insert(0, box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0));
        idx.insert(1, box3d(5.0, 5.0, 5.0, 10.0, 10.0, 10.0));

        let hits = idx.query_except(&box3d(8.0, 8.0, 8.0, 4.0, 4.0, 4.0), 0);
        assert!(!hits.contains(&0));
        assert!(hits.contains(&1));
    }

    #[test]
    fn test_3d_has_collision() {
        let mut idx = SpatialIndex3D::new();
        idx.insert(0, box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0));

        assert!(idx.has_collision(&box3d(5.0, 5.0, 5.0, 10.0, 10.0, 10.0)));
        assert!(!idx.has_collision(&box3d(20.0, 20.0, 20.0, 10.0, 10.0, 10.0)));
    }

    #[test]
    fn test_3d_has_collision_except() {
        let mut idx = SpatialIndex3D::new();
        idx.insert(0, box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0));
        idx.insert(1, box3d(5.0, 5.0, 5.0, 10.0, 10.0, 10.0));

        // Query overlaps both, but exclude 0 → only 1 counts
        assert!(idx.has_collision_except(&box3d(8.0, 8.0, 8.0, 4.0, 4.0, 4.0), 0));
        // Exclude both matches
        assert!(!idx.has_collision_except(&box3d(0.0, 0.0, 0.0, 2.0, 2.0, 2.0), 0));
    }

    #[test]
    fn test_3d_remove() {
        let mut idx = SpatialIndex3D::new();
        idx.insert(0, box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0));
        idx.insert(1, box3d(5.0, 5.0, 5.0, 10.0, 10.0, 10.0));

        assert_eq!(idx.len(), 2);
        idx.remove(0);
        assert_eq!(idx.len(), 1);
        assert!(!idx.has_collision(&box3d(0.0, 0.0, 0.0, 2.0, 2.0, 2.0)));
    }

    #[test]
    fn test_3d_get_bounds() {
        let mut idx = SpatialIndex3D::new();
        let b = box3d(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        idx.insert(42, b);

        let found = idx.get_bounds(42).unwrap();
        assert!((found.min.x - 1.0).abs() < 1e-10);
        assert!((found.max.z - 9.0).abs() < 1e-10);
        assert!(idx.get_bounds(99).is_none());
    }

    #[test]
    fn test_3d_clear() {
        let mut idx = SpatialIndex3D::new();
        idx.insert(0, box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0));
        idx.clear();
        assert!(idx.is_empty());
    }

    #[test]
    fn test_3d_with_capacity() {
        let idx = SpatialIndex3D::with_capacity(100);
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_3d_z_axis_separation() {
        let mut idx = SpatialIndex3D::new();
        // Stack at same x,y but different z
        idx.insert(0, box3d(0.0, 0.0, 0.0, 10.0, 10.0, 10.0));
        idx.insert(1, box3d(0.0, 0.0, 20.0, 10.0, 10.0, 10.0));

        // Query in the gap between them
        let hits = idx.query(&box3d(0.0, 0.0, 12.0, 10.0, 10.0, 5.0));
        assert!(hits.is_empty());

        // Query overlapping the top box
        let hits = idx.query(&box3d(0.0, 0.0, 18.0, 10.0, 10.0, 5.0));
        assert_eq!(hits.len(), 1);
        assert!(hits.contains(&1));
    }
}
