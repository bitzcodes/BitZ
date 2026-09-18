//! GKR grand-product argument over a multilinear field.
//!
//! Proves `∏_{x ∈ {0,1}^d} v(x) = root` for a leaf vector `v` of length
//! `2^d`, via a layer-by-layer product tree verified with sumchecks
//! (Thaler's grand product — equivalently the *denominator* component of
//! the Papini–Häböck GKR fraction tree, specialised to a pure product, so
//! there is no numerator and no `α` batching).
//!
//! This is the reusable engine for the char-2-sound multiplicative lookup
//! `∏_i (δ − a_i) = ∏_t (δ − fp_t)^{m_t}`: the witness side and the
//! binary-multiplicity table side are each one product tree. The argument
//! is field-agnostic and char-2-safe — products never cancel (`(δ−v)² ≠ 0`),
//! unlike the additive LogUp sum.
//!
//! ## Layer layout
//!
//! Layer `k` has `2^k` entries; layer `d` = leaves, layer `0` = root.
//! ```text
//!   v_k[i] = v_{k+1}[i] · v_{k+1}[i + 2^k]
//! ```
//! For `k ≥ 1` the parent claim `ṽ_k(r) = Σ_z eq(r,z)·left(z)·right(z)`
//! (`left = v_{k+1}[..2^k]`, `right = v_{k+1}[2^k..]`) is a degree-3
//! sumcheck; layer `0` is a direct product check `root = left · right`.
//!
//! ## Output / binding
//!
//! [`prove_product_tree`] returns `(proof, point r ∈ F^d, leaf_eval)` where
//! `leaf_eval = ṽ_d(r)` is the claimed evaluation of the leaf MLE at `r`.
//! [`verify_product_tree`] re-derives `(r, leaf_eval)` from the proof. The
//! caller **binds** `leaf_eval` to the actual leaves at `r` (for witness
//! leaves: a `ψ_z` read-off of the committed words; for table leaves: the
//! public / structured fingerprints and the committed multiplicity bits).
//! Leaf count must be a power of two; pad with the multiplicative identity
//! `1` (a no-op factor) otherwise.

use crate::poly::coefficient::PolynomialField;

use crate::poly::utils::eq_eval;
use crate::transcript::traits::{ConstTranscribable, GenTranscribable, Transcript};
use crate::utils::{cfg_into_iter, cfg_iter_mut, inner_transparent_field::InnerTransparentField};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::piop::sumcheck::{MLSumcheck, SumcheckProof};

/// Proof that `∏ leaves = root` for a leaf vector of length `2^d`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductTreeProof<F: PolynomialField> {
    /// The claimed product of all leaves.
    pub root: F,
    /// Per-layer proofs, one per GKR level `k = 0..d`.
    pub layers: Vec<ProductLayerProof<F>>,
}

/// Proof for a single product-tree layer: the two child evaluations at the
/// layer's subclaim point, plus the sumcheck (`None` for layer 0, which has
/// zero variables and is a direct product check).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductLayerProof<F: PolynomialField> {
    /// Sumcheck proof for this layer (`None` for layer `k = 0`).
    pub sumcheck_proof: Option<SumcheckProof<F>>,
    /// Left-child MLE evaluation at the subclaim point.
    pub left: F,
    /// Right-child MLE evaluation at the subclaim point.
    pub right: F,
}

/// Failure modes of [`verify_product_tree`].
#[derive(Debug, thiserror::Error)]
pub enum ProductTreeError {
    #[error("product-tree layer count != num_vars")]
    LayerCountMismatch,
    #[error("product-tree layer-0 product check failed (root != left·right)")]
    Layer0Mismatch,
    #[error("product-tree layer missing its sumcheck proof")]
    MissingSumcheck,
    #[error("product-tree layer claimed-sum != parent claim")]
    ClaimedSumMismatch,
    #[error("product-tree layer final-eval != eq·left·right")]
    FinalEvalMismatch,
    #[error("product-tree inner sumcheck/eq error")]
    Sumcheck,
}

/// A product-tree layer stored as its two contiguous halves
/// `(left = layer[..len/2], right = layer[len/2..])`. A GKR layer of size
/// `2^{k+1}` is exactly `[left children | right children]` (parent `i` =
/// `layer[i] · layer[i + 2^k]`), and the merged layer sumcheck consumes it
/// as an `(L, R)` pair — storing the halves separately from birth makes the
/// per-layer group handoff a pure move instead of a `split_off`
/// (alloc + memcpy of half of every layer, ~N cells per prove across the
/// forest). The root layer (size 1) is `(vec![root], vec![])`.
type LayerHalves<F> = (Vec<F>, Vec<F>);

/// Parent layer of a child stored as halves: `parent[i] = cl[i] · cr[i]`
/// (`i < 2^level`), written directly into its own halves (no split copy).
#[allow(clippy::arithmetic_side_effects)]
fn build_parent_level<F>(cl: &[F], cr: &[F], level: usize) -> LayerHalves<F>
where
    F: InnerTransparentField,
{
    debug_assert_eq!(cl.len(), 1usize << level);
    debug_assert_eq!(cr.len(), 1usize << level);
    let prod = |i: usize| -> F { cl[i].clone() * &cr[i] };
    if level == 0 {
        return (vec![prod(0)], Vec::new());
    }
    let h = 1usize << (level - 1);
    let left: Vec<F> = (0..h).map(prod).collect();
    let right: Vec<F> = (h..1usize << level).map(prod).collect();
    (left, right)
}

/// Build the product tree bottom-up: returns `[leaves, …, root]` where
/// entry `k` is GKR layer `d − k` (so `[0]` = leaves of `2^d`, last = root),
/// every layer as [`LayerHalves`]. The one `split_off` here (the leaf layer)
/// is the eager path's only half-copy; all upper layers are built pre-split.
#[allow(clippy::arithmetic_side_effects)]
fn build_product_tree<F>(leaves: Vec<F>) -> Vec<LayerHalves<F>>
where
    F: InnerTransparentField + Send + Sync,
{
    let d = crate::utils::log2(leaves.len()) as usize;
    debug_assert_eq!(
        leaves.len(),
        1usize << d,
        "leaf count must be a power of two"
    );
    debug_assert!(d >= 1, "product trees have depth ≥ 1");
    let mut left = leaves;
    let right = left.split_off(1usize << (d - 1));
    let mut layers: Vec<LayerHalves<F>> = Vec::with_capacity(d + 1);
    layers.push((left, right));
    for level in (0..d).rev() {
        // Sequential within a tree; the forest parallelises *across* trees
        // (see `prove_product_forest`), so nesting tiny per-level jobs here
        // would only add rayon-spawn overhead.
        let parent = {
            let (cl, cr) = layers.last().expect("tree non-empty during construction");
            build_parent_level(cl, cr, level)
        };
        layers.push(parent);
    }
    layers
}

/// Build a product tree's UPPER layers from its (caller-built) layer 1,
/// returning `[⟨empty leaf placeholder⟩, layer1, …, root]`. Used by
/// [`prove_product_forest_lazy`]: the leaf layer (`2^d`, the largest) is
/// never materialised at build — the caller fuses leaf generation into its
/// `gen_layer1` — and gets regenerated just-in-time for the final GKR
/// round, keeping the forest peak at `~N`.
#[allow(clippy::arithmetic_side_effects)]
fn build_product_tree_from_layer1<F>(layer1: LayerHalves<F>) -> Vec<LayerHalves<F>>
where
    F: InnerTransparentField + Send + Sync,
{
    let half1 = layer1.0.len() + layer1.1.len();
    let dm1 = crate::utils::log2(half1) as usize; // = d − 1
    debug_assert_eq!(half1, 1usize << dm1, "layer-1 size must be a power of two");
    debug_assert_eq!(
        layer1.0.len(),
        layer1.1.len(),
        "layer 1 must come pre-split"
    );
    let mut layers: Vec<LayerHalves<F>> = Vec::with_capacity(dm1 + 2);
    layers.push((Vec::new(), Vec::new())); // slot 0: leaf placeholder, refilled lazily
    layers.push(layer1);
    for level in (0..dm1).rev() {
        let parent = {
            let (cl, cr) = layers.last().expect("non-empty during construction");
            build_parent_level(cl, cr, level)
        };
        layers.push(parent);
    }
    layers
}

/// Absorb a slice of field elements into the transcript with a SINGLE Blake3
/// update — one domain tag, then every element's inner bytes concatenated —
/// instead of [`Transcript::absorb_random_field`]'s 4 updates *per element*.
/// Used for the forest's `2^s`-scale absorbs (the roots and the per-layer
/// child / final evals). Prover and verifier MUST call it with the same element
/// order; it defines the Fiat–Shamir bytes for those steps.
#[allow(clippy::arithmetic_side_effects)] // index math over a freshly-sized buffer
pub(crate) fn absorb_field_slice<F>(transcript: &mut impl Transcript, xs: &[F])
where
    F: PolynomialField,
    F::Inner: ConstTranscribable,
{
    let nb = F::Inner::NUM_BYTES;
    let mut buf = vec![0u8; xs.len() * nb];
    for (i, x) in xs.iter().enumerate() {
        x.inner()
            .write_transcription_bytes_exact(&mut buf[i * nb..(i + 1) * nb]);
    }
    transcript.absorb_inner(&[0x9]); // domain tag: batched field-element slice
    transcript.absorb_inner(&buf);
}

/// Proof for a **forest** of product trees with their layer sumchecks
/// BATCHED: same-index layers of all active trees run as ONE multi-group
/// eq-factored sumcheck, the per-tree claims combined under a fresh
/// per-layer challenge ρ (group `t` enters with scale `ρ^t`) and reduced
/// at a SHARED point with one line-challenge λ. Trees must come in
/// non-increasing depth order (the active set is always a prefix). A
/// single-tree forest is transcript-identical to the pre-forest
/// [`prove_product_tree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductForestProof<F: PolynomialField> {
    /// Per-tree claimed products, tree-major order.
    pub roots: Vec<F>,
    /// Per merged layer `k = 0..max_depth−1`.
    pub layers: Vec<ForestLayerProof<F>>,
}

/// One merged forest layer: the shared sumcheck (`None` for layer 0, the
/// direct per-tree `root = l·r` check) and the per-ACTIVE-tree
/// `(left, right)` child evaluations at the layer's shared subclaim point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForestLayerProof<F: PolynomialField> {
    pub sumcheck_proof: Option<SumcheckProof<F>>,
    pub evals: Vec<(F, F)>,
}

/// Batched-forest GKR prover: proves `∏ leaves_t = root_t` for every tree
/// at once, with same-index layers merged (see [`ProductForestProof`]).
/// Returns the proof and, per tree, the leaf-layer point and claimed
/// leaf-MLE evaluation `(r_t ∈ F^{d_t}, ṽ_{d_t}(r_t))` the caller binds.
/// GKR grand-product forest prover (eager): builds every tree fully, then proves.
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_product_forest<F>(
    transcript: &mut impl Transcript,
    leaves_per_tree: Vec<Vec<F>>,
    field_cfg: &F::Config,
) -> (ProductForestProof<F>, Vec<(Vec<F>, F)>)
where
    F: InnerTransparentField + crate::utils::wide_mul::WideMulAcc + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let trees: Vec<Vec<LayerHalves<F>>> = {
        let _g = tracing::info_span!("gkr:build").entered();
        // Trees are independent → parallel across trees (each built sequentially).
        cfg_into_iter!(leaves_per_tree)
            .map(build_product_tree)
            .collect()
    };
    forest_rounds(transcript, trees, None, field_cfg)
}

/// Bit-affine leaf provider for the lazy forest: tree `t`'s leaf entry `i`
/// is `1 + bit_i·tau[i]`, with the packed bit halves supplied per tree by
/// `bits_of(t)` (64 bits per `u64`, position `i` at word `i/64`, bit `i%64`)
/// and the public per-position coefficients shared through
/// `tau_sets[tau_set_of(t)]` (`(tau_l, tau_r)` = the L/R leaf halves'
/// coefficients). The final GKR round consumes this directly — leaf values
/// are never materialised, the round-1 message runs as branchless
/// bit-selected subset-sums, and its fold produces the dense next-round
/// buffers (see `eq_factored::GroupBufs::LeafBits`).
pub struct ForestLeafBits<'a, F> {
    pub bits_of: &'a (dyn Fn(usize) -> (Vec<u64>, Vec<u64>) + Sync),
    pub tau_sets: &'a [(Vec<F>, Vec<F>)],
    pub tau_set_of: &'a (dyn Fn(usize) -> usize + Sync),
}

/// Lazy-leaf forest prover: builds each tree's UPPER layers from
/// `gen_layer1(t)` (the leaf layer — the largest — is never materialised at
/// build) and consumes the leaf layer **straight from the committed bits**
/// in the final GKR round (see [`ForestLeafBits`]) — no leaf regeneration,
/// no materialised leaf values at all. `gen_layer1` must be consistent with
/// the bit-affine leaves (`layer1[i] = leaf(i)·leaf(i + 2^{d−1})` with
/// `leaf(i) = 1 + bit_i·tau[i]`); the proof is byte-identical to
/// [`prove_product_forest`] over the materialised leaves — every step is an
/// exact char-2 algebraic identity. All trees must have equal depth ≥ 3
/// (materialise tiny forests eagerly instead). (A layer1-less variant —
/// drop the stored level 1 and regenerate it at its round — was measured
/// and REVERTED: +3.5–17% prove for only ~−11% peak RSS; see the ledger's
/// "layer1-less forest" entry for the numbers and the LeafPairBits case-LUT
/// design that would do it without duplicate work.)
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_product_forest_lazy<F, G1>(
    transcript: &mut impl Transcript,
    num_trees: usize,
    gen_layer1: G1,
    leaf: ForestLeafBits<'_, F>,
    field_cfg: &F::Config,
) -> (ProductForestProof<F>, Vec<(Vec<F>, F)>)
where
    F: InnerTransparentField + crate::utils::wide_mul::WideMulAcc + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
    G1: Fn(usize) -> (Vec<F>, Vec<F>) + Sync,
{
    let mut trees: Vec<Vec<LayerHalves<F>>> = {
        let _g = tracing::info_span!("gkr:build").entered();
        cfg_into_iter!(0..num_trees)
            .map(|t| build_product_tree_from_layer1(gen_layer1(t)))
            .collect()
    };
    if trees.iter().any(|t| t.len() < 4) {
        // Tiny forest (depth < 3): the leaf round would have k < 2, which
        // the bit-affine driver path rejects — materialise the leaves
        // (`1 + bit·τ`, the identical values) and run the eager rounds.
        let one = F::one_with_cfg(field_cfg);
        for (t, tree) in trees.iter_mut().enumerate() {
            let (lbits, rbits) = (leaf.bits_of)(t);
            let (tau_l, tau_r) = &leaf.tau_sets[(leaf.tau_set_of)(t)];
            let build = |bits: &[u64], tau: &[F]| -> Vec<F> {
                tau.iter()
                    .enumerate()
                    .map(|(i, tv)| {
                        let mut v = one.clone();
                        F::add_assign_masked(&mut v, tv, (bits[i >> 6] >> (i & 63)) & 1 == 1);
                        v
                    })
                    .collect()
            };
            tree[0] = (build(&lbits, tau_l), build(&rbits, tau_r));
        }
        return forest_rounds(transcript, trees, None, field_cfg);
    }
    forest_rounds(transcript, trees, Some(&leaf), field_cfg)
}

/// Shared round-loop for both forest provers. When `leaf` is `Some`, trees'
/// empty leaf slots (`trees[t][0]`) are never filled — the round that would
/// consume them gets bit-affine `LeafBits` groups instead (the lazy path);
/// `None` is the eager path (leaf slot already populated).
#[allow(clippy::arithmetic_side_effects)]
fn forest_rounds<F>(
    transcript: &mut impl Transcript,
    mut trees: Vec<Vec<LayerHalves<F>>>,
    leaf: Option<&ForestLeafBits<'_, F>>,
    field_cfg: &F::Config,
) -> (ProductForestProof<F>, Vec<(Vec<F>, F)>)
where
    F: InnerTransparentField + crate::utils::wide_mul::WideMulAcc + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    let depths: Vec<usize> = trees.iter().map(|t| t.len() - 1).collect();
    debug_assert!(!depths.is_empty() && depths.iter().all(|&d| d >= 1));
    debug_assert!(
        depths.windows(2).all(|w| w[0] >= w[1]),
        "non-increasing depths"
    );
    let max_d = depths[0];
    let num_trees = trees.len();
    let one = F::one_with_cfg(field_cfg);

    let roots: Vec<F> = trees
        .iter()
        .map(|t| t.last().expect("built").0[0].clone())
        .collect();
    absorb_field_slice(transcript, &roots);

    // Per-tree state: the running claim v_t at the running point r_t.
    let mut v: Vec<F> = roots.clone();
    let mut r: Vec<Vec<F>> = vec![Vec::new(); num_trees];
    let mut layer_proofs = Vec::with_capacity(max_d);

    for k in 0..max_d {
        // Active trees at this layer (a prefix, by the depth ordering).
        let active = depths.iter().filter(|&&d| d > k).count();
        let child = |t: usize| -> &LayerHalves<F> { &trees[t][depths[t] - (k + 1)] };

        if k == 0 {
            let mut evals = Vec::with_capacity(active);
            for t in 0..active {
                let c = child(t);
                let (l, rgt) = (c.0[0].clone(), c.1[0].clone());
                debug_assert_eq!(v[t], l.clone() * &rgt, "root must equal left·right");
                evals.push((l, rgt));
            }
            let flat: Vec<F> = evals
                .iter()
                .flat_map(|(l, rgt)| [l.clone(), rgt.clone()])
                .collect();
            absorb_field_slice(transcript, &flat);
            let lambda: F = transcript.get_field_challenge(field_cfg);
            for (t, (l, rgt)) in evals.iter().enumerate() {
                v[t] = (one.clone() - &lambda) * l + &(lambda.clone() * rgt);
                r[t] = vec![lambda.clone()];
            }
            layer_proofs.push(ForestLayerProof {
                sumcheck_proof: None,
                evals,
            });
        } else {
            let _g = tracing::info_span!("gkr:round").entered();
            // Fresh per-layer batching challenge (only when ≥ 2 trees are
            // active); group t enters the merged sumcheck with scale ρ^t,
            // so the claimed sum is Σ_t ρ^t·v_t and the final evaluation
            // pins each tree's contribution by Schwartz–Zippel.
            let rho: Option<F> = (active > 1).then(|| transcript.get_field_challenge(field_cfg));
            // Own callback-produced bits for the entire sumcheck; groups only
            // borrow these immutable inputs.
            let leaf_bits: Vec<_> = if leaf.is_some() && depths[..active].contains(&(k + 1)) {
                cfg_into_iter!(0..active)
                    .map(|t| {
                        leaf.filter(|_| depths[t] == k + 1)
                            .map(|leaf| (leaf.bits_of)(t))
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let groups = {
                let _g = tracing::info_span!("gkr:groups").entered();
                // Prefix scales ρ^t (sequential prefix product — `active` muls),
                // then build the per-tree groups in parallel. Each borrows
                // its point and either moves pre-split dense halves
                // ([`LayerHalves`]) or borrows callback-owned leaf bits, so
                // handing off these buffers needs no allocation or memcpy.
                let mut scales = Vec::with_capacity(active);
                let mut scale = one.clone();
                for _ in 0..active {
                    scales.push(scale.clone());
                    if let Some(rho) = &rho {
                        scale = scale * rho;
                    }
                }
                cfg_iter_mut!(trees[..active])
                    .enumerate()
                    .map(|(t, tree)| {
                        let layer_idx = depths[t] - (k + 1);
                        let bufs = match (layer_idx, leaf) {
                            // Leaf layer on the lazy path: hand the driver
                            // the committed bits — the leaf values are never
                            // materialised.
                            (0, Some(leaf)) => {
                                let (lbits, rbits) =
                                    leaf_bits[t].as_ref().expect("leaf bits present");
                                crate::piop::sumcheck::eq_factored::GroupBufs::LeafBits {
                                    lbits,
                                    rbits,
                                    tau_set: (leaf.tau_set_of)(t),
                                }
                            }
                            _ => {
                                let (left, right) = std::mem::take(&mut tree[layer_idx]);
                                debug_assert_eq!(left.len(), 1usize << k);
                                debug_assert_eq!(right.len(), 1usize << k);
                                crate::piop::sumcheck::eq_factored::GroupBufs::Dense(vec![(
                                    left, right,
                                )])
                            }
                        };
                        crate::piop::sumcheck::eq_factored::EqInnerGroupMixed {
                            q: r[t].as_slice().into(),
                            scale: scales[t].clone(),
                            bufs,
                        }
                    })
                    .collect()
            };
            let (sumcheck_proof, s, final_evals) =
                crate::piop::sumcheck::eq_factored::prove_eq_inner_sumcheck_mixed(
                    transcript,
                    groups,
                    leaf.map_or(&[][..], |l| l.tau_sets),
                    &[],
                    &[],
                    field_cfg,
                );

            let evals = {
                let _g = tracing::info_span!("gkr:absorb").entered();
                let evals: Vec<(F, F)> = final_evals.iter().map(|fe| fe[0].clone()).collect();
                let flat: Vec<F> = evals
                    .iter()
                    .flat_map(|(l, r)| [l.clone(), r.clone()])
                    .collect();
                absorb_field_slice(transcript, &flat);
                evals
            };
            let lambda: F = transcript.get_field_challenge(field_cfg);
            {
                let _g = tracing::info_span!("gkr:vr").entered();
                for (t, (l_at, r_at)) in evals.iter().enumerate() {
                    v[t] = (one.clone() - &lambda) * l_at + &(lambda.clone() * r_at);
                    r[t] = s.clone();
                    r[t].push(lambda.clone());
                }
            }
            layer_proofs.push(ForestLayerProof {
                sumcheck_proof: Some(sumcheck_proof),
                evals,
            });
        }
    }

    let claims: Vec<(Vec<F>, F)> = r.into_iter().zip(v).collect();
    (
        ProductForestProof {
            roots,
            layers: layer_proofs,
        },
        claims,
    )
}

/// GKR grand-product prover (single tree).
///
/// Returns the proof, the leaf-layer evaluation point `r ∈ F^d`, and the
/// claimed leaf-MLE evaluation `ṽ_d(r)` that the caller must bind to the
/// actual leaves. Thin wrapper over the single-tree
/// [`prove_product_forest`] (transcript-identical to the pre-forest form).
#[allow(clippy::arithmetic_side_effects)]
pub fn prove_product_tree<F>(
    transcript: &mut impl Transcript,
    leaves: Vec<F>,
    field_cfg: &F::Config,
) -> (ProductTreeProof<F>, Vec<F>, F)
where
    F: InnerTransparentField + crate::utils::wide_mul::WideMulAcc + Send + Sync,
    F::Inner: ConstTranscribable + Zero + Default + Send + Sync,
    F::Modulus: ConstTranscribable,
    F::Config: Sync,
{
    if leaves.len() == 1 {
        let root = leaves[0].clone();
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        transcript.absorb_random_field(&root, &mut buf);
        return (
            ProductTreeProof {
                root: root.clone(),
                layers: vec![],
            },
            vec![],
            root,
        );
    }
    let (forest, mut claims) = prove_product_forest(transcript, vec![leaves], field_cfg);
    let (point, eval) = claims.pop().expect("one tree");
    let layers = forest
        .layers
        .into_iter()
        .map(|fl| {
            let (left, right) = fl.evals.into_iter().next().expect("one tree");
            ProductLayerProof {
                sumcheck_proof: fl.sumcheck_proof,
                left,
                right,
            }
        })
        .collect();
    (
        ProductTreeProof {
            root: forest.roots.into_iter().next().expect("one tree"),
            layers,
        },
        point,
        eval,
    )
}

/// GKR grand-product verifier.
///
/// Re-derives the leaf-layer point `r ∈ F^d` and the claimed leaf-MLE
/// evaluation `ṽ_d(r)`. The caller binds the latter to the actual leaves.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_product_tree<F>(
    transcript: &mut impl Transcript,
    proof: &ProductTreeProof<F>,
    num_vars: usize,
    field_cfg: &F::Config,
) -> Result<(Vec<F>, F), ProductTreeError>
where
    F: InnerTransparentField + Send + Sync,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    if num_vars == 0 {
        let mut buf = vec![0u8; F::Inner::NUM_BYTES];
        transcript.absorb_random_field(&proof.root, &mut buf);
        return Ok((vec![], proof.root.clone()));
    }
    let forest = ProductForestProof {
        roots: vec![proof.root.clone()],
        layers: proof
            .layers
            .iter()
            .map(|lp| ForestLayerProof {
                sumcheck_proof: lp.sumcheck_proof.clone(),
                evals: vec![(lp.left.clone(), lp.right.clone())],
            })
            .collect(),
    };
    let mut claims = verify_product_forest(transcript, &forest, &[num_vars], field_cfg)?;
    Ok(claims.pop().expect("one tree"))
}

/// Batched-forest GKR verifier (see [`ProductForestProof`]): re-derives,
/// per tree, the leaf-layer point and claimed leaf-MLE evaluation. The
/// caller supplies the per-tree depths (non-increasing) and binds the
/// returned claims to the actual leaves.
#[allow(clippy::arithmetic_side_effects)]
pub fn verify_product_forest<F>(
    transcript: &mut impl Transcript,
    proof: &ProductForestProof<F>,
    depths: &[usize],
    field_cfg: &F::Config,
) -> Result<Vec<(Vec<F>, F)>, ProductTreeError>
where
    F: InnerTransparentField + Send + Sync,
    F::Inner: ConstTranscribable + Zero,
    F::Modulus: ConstTranscribable,
{
    let num_trees = depths.len();
    if num_trees == 0
        || depths.iter().any(|&d| d < 1)
        || depths.windows(2).any(|w| w[0] < w[1])
        || proof.roots.len() != num_trees
    {
        return Err(ProductTreeError::LayerCountMismatch);
    }
    let max_d = depths[0];
    if proof.layers.len() != max_d {
        return Err(ProductTreeError::LayerCountMismatch);
    }
    let one = F::one_with_cfg(field_cfg);

    absorb_field_slice(transcript, &proof.roots);

    let mut v: Vec<F> = proof.roots.clone();
    let mut r: Vec<Vec<F>> = vec![Vec::new(); num_trees];

    for k in 0..max_d {
        let active = depths.iter().filter(|&&d| d > k).count();
        let lp = &proof.layers[k];
        if lp.evals.len() != active {
            return Err(ProductTreeError::LayerCountMismatch);
        }

        if k == 0 {
            let flat: Vec<F> = lp
                .evals
                .iter()
                .flat_map(|(l, rgt)| [l.clone(), rgt.clone()])
                .collect();
            absorb_field_slice(transcript, &flat);
            for (t, (l, rgt)) in lp.evals.iter().enumerate() {
                if v[t] != l.clone() * rgt {
                    return Err(ProductTreeError::Layer0Mismatch);
                }
            }
            let lambda: F = transcript.get_field_challenge(field_cfg);
            for (t, (l, rgt)) in lp.evals.iter().enumerate() {
                v[t] = (one.clone() - &lambda) * l + &(lambda.clone() * rgt);
                r[t] = vec![lambda.clone()];
            }
        } else {
            let sc = lp
                .sumcheck_proof
                .as_ref()
                .ok_or(ProductTreeError::MissingSumcheck)?;
            // ρ batches the active trees' claims; the claimed sum must be
            // Σ_t ρ^t·v_t, and the final evaluation pins each tree's
            // eq_t(s)·l_t·r_t contribution under the same ρ-powers.
            let rho: Option<F> = (active > 1).then(|| transcript.get_field_challenge(field_cfg));
            let mut claimed = F::zero_with_cfg(field_cfg);
            let mut scale = one.clone();
            for vt in v.iter().take(active) {
                claimed = claimed + &(scale.clone() * vt);
                if let Some(rho) = &rho {
                    scale = scale * rho;
                }
            }
            if sc.claimed_sum != claimed {
                return Err(ProductTreeError::ClaimedSumMismatch);
            }
            let subclaim = MLSumcheck::verify_as_subprotocol(transcript, k, 3, sc, field_cfg)
                .map_err(|_| ProductTreeError::Sumcheck)?;
            let s = &subclaim.point;

            let flat: Vec<F> = lp
                .evals
                .iter()
                .flat_map(|(l, rgt)| [l.clone(), rgt.clone()])
                .collect();
            absorb_field_slice(transcript, &flat);
            let mut expected = F::zero_with_cfg(field_cfg);
            let mut scale = one.clone();
            for (t, (l, rgt)) in lp.evals.iter().enumerate() {
                let eq_val =
                    eq_eval(s, &r[t], one.clone()).map_err(|_| ProductTreeError::Sumcheck)?;
                expected = expected + &(scale.clone() * &(eq_val * &(l.clone() * rgt)));
                if let Some(rho) = &rho {
                    scale = scale * rho;
                }
            }
            if expected != subclaim.expected_evaluation {
                return Err(ProductTreeError::FinalEvalMismatch);
            }

            let lambda: F = transcript.get_field_challenge(field_cfg);
            for (t, (l, rgt)) in lp.evals.iter().enumerate() {
                v[t] = (one.clone() - &lambda) * l + &(lambda.clone() * rgt);
                r[t] = s.clone();
                r[t].push(lambda.clone());
            }
        }
    }

    Ok(r.into_iter().zip(v).collect())
}
