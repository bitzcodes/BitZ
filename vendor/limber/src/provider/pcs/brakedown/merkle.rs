//! Minimal binary Merkle tree over `[u8; 32]` leaves (BLAKE3), with
//! domain-separated leaf/node hashing and stateless path verification. Used to
//! commit the columns of a Brakedown encoded matrix. BLAKE3 (SIMD,
//! ~5-10x Keccak single-thread throughput) is used only for this
//! column/tree commitment; the Fiat-Shamir transcript remains Keccak.

use rayon::prelude::*;

/// A 32-byte BLAKE3 digest.
pub type Hash = [u8; 32];

/// Hash raw leaf data (domain tag `0`).
pub fn hash_leaf(data: &[u8]) -> Hash {
  let mut h = blake3::Hasher::new();
  h.update(&[0u8]);
  h.update(data);
  *h.finalize().as_bytes()
}

/// Hash two child digests (domain tag `1`).
fn hash_node(l: &Hash, r: &Hash) -> Hash {
  let mut h = blake3::Hasher::new();
  h.update(&[1u8]);
  h.update(l);
  h.update(r);
  *h.finalize().as_bytes()
}

/// A binary Merkle tree; leaves are padded to a power of two with the zero hash.
#[derive(Clone, Debug)]
pub struct MerkleTree {
  /// `layers[0]` are the (padded) leaf hashes; the last layer is the root.
  layers: Vec<Vec<Hash>>,
}

impl MerkleTree {
  /// Build a tree from already-hashed leaves (padded to a power of two).
  pub fn from_leaves(mut level: Vec<Hash>) -> Self {
    assert!(!level.is_empty(), "Merkle tree needs at least one leaf");
    level.resize(level.len().next_power_of_two(), [0u8; 32]);
    let mut layers = vec![level];
    while layers.last().unwrap().len() > 1 {
      let cur = layers.last().unwrap();
      // Each layer's node hashes are independent → parallelize.
      let next: Vec<Hash> = cur.par_chunks(2).map(|p| hash_node(&p[0], &p[1])).collect();
      layers.push(next);
    }
    Self { layers }
  }

  /// The Merkle root.
  pub fn root(&self) -> Hash {
    *self.layers.last().unwrap().first().unwrap()
  }

  /// Authentication path (sibling hashes, bottom-up) for leaf `idx`.
  pub fn path(&self, mut idx: usize) -> Vec<Hash> {
    let mut path = Vec::with_capacity(self.layers.len() - 1);
    for layer in &self.layers[..self.layers.len() - 1] {
      path.push(layer[idx ^ 1]);
      idx >>= 1;
    }
    path
  }

  /// Tree height (number of layers above the leaves).
  pub fn height(&self) -> usize {
    self.layers.len() - 1
  }

  /// Batched authentication path (a Merkle multiproof) for many leaves: the
  /// minimal set of sibling hashes needed to recompute the root, emitted in the
  /// order [`verify_batch_path`] consumes them (bottom-up, left-to-right). A
  /// sibling that is itself an opened/derivable node is omitted. `indices` need
  /// not be sorted or unique.
  pub fn batch_path(&self, indices: &[usize]) -> Vec<Hash> {
    let mut known: Vec<usize> = indices.to_vec();
    known.sort_unstable();
    known.dedup();
    let mut auth = Vec::new();
    for layer in &self.layers[..self.layers.len() - 1] {
      let mut next = Vec::with_capacity(known.len());
      let mut i = 0;
      while i < known.len() {
        let idx = known[i];
        if i + 1 < known.len() && known[i + 1] == idx ^ 1 {
          i += 2; // both children present → parent is derivable
        } else {
          auth.push(layer[idx ^ 1]);
          i += 1;
        }
        next.push(idx >> 1);
      }
      known = next;
    }
    auth
  }
}

/// Verify a batched authentication path. `leaves` are `(index, leaf_hash)` pairs;
/// they are sorted/deduped internally. `height` is the tree height (use
/// [`MerkleTree::height`], or `ceil(log2(n_leaves))`). Recomputes the root from
/// the opened leaves + `auth` and compares to `root`.
pub fn verify_batch_path(
  root: &Hash,
  height: usize,
  leaves: &[(usize, Hash)],
  auth: &[Hash],
) -> bool {
  let mut known: Vec<(usize, Hash)> = leaves.to_vec();
  known.sort_by_key(|(i, _)| *i);
  known.dedup_by_key(|(i, _)| *i);
  let mut ai = 0usize;
  for _ in 0..height {
    let mut next: Vec<(usize, Hash)> = Vec::with_capacity(known.len());
    let mut i = 0;
    while i < known.len() {
      let (idx, h) = known[i];
      let (left, right, step) = if i + 1 < known.len() && known[i + 1].0 == idx ^ 1 {
        (h, known[i + 1].1, 2)
      } else {
        if ai >= auth.len() {
          return false;
        }
        let a = auth[ai];
        ai += 1;
        if idx & 1 == 0 { (h, a, 1) } else { (a, h, 1) }
      };
      next.push((idx >> 1, hash_node(&left, &right)));
      i += step;
    }
    known = next;
  }
  ai == auth.len() && known.len() == 1 && &known[0].1 == root
}

/// Recompute the root from a leaf and its path; compare to `root`.
pub fn verify_path(root: &Hash, leaf_data: &[u8], mut idx: usize, path: &[Hash]) -> bool {
  let mut cur = hash_leaf(leaf_data);
  for sib in path {
    cur = if idx & 1 == 0 {
      hash_node(&cur, sib)
    } else {
      hash_node(sib, &cur)
    };
    idx >>= 1;
  }
  &cur == root
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn paths_verify_and_reject_tampering() {
    let n = 100usize;
    let data: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 7]).collect();
    let leaves: Vec<Hash> = data.iter().map(|d| hash_leaf(d)).collect();
    let tree = MerkleTree::from_leaves(leaves);
    let root = tree.root();
    for (i, di) in data.iter().enumerate() {
      let path = tree.path(i);
      assert!(verify_path(&root, di, i, &path), "path {i} must verify");
      // wrong data rejects
      assert!(!verify_path(&root, b"bogus", i, &path));
      // wrong index rejects
      assert!(!verify_path(&root, di, (i + 1) % n, &path));
    }
  }

  #[test]
  fn batch_paths_verify_and_reject() {
    let n = 100usize;
    let data: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 5]).collect();
    let leaves: Vec<Hash> = data.iter().map(|d| hash_leaf(d)).collect();
    let tree = MerkleTree::from_leaves(leaves);
    let root = tree.root();

    let idxs = [3usize, 3, 17, 42, 99, 0, 64]; // unsorted, with a duplicate + edges
    let auth = tree.batch_path(&idxs);
    let pairs: Vec<(usize, Hash)> = idxs.iter().map(|&i| (i, hash_leaf(&data[i]))).collect();
    assert!(verify_batch_path(&root, tree.height(), &pairs, &auth));

    // tampering a leaf rejects
    let mut bad = pairs.clone();
    bad[2].1 = hash_leaf(b"bogus");
    assert!(!verify_batch_path(&root, tree.height(), &bad, &auth));

    // batched proof is no larger than the sum of individual paths
    let mut uniq = idxs.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    assert!(auth.len() <= uniq.len() * tree.height());
  }
}
