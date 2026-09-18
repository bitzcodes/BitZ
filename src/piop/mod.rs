pub mod lookup;
pub mod spartan;
pub mod sumcheck;

/// A combination function `f(evals) -> value` over a per-hypercube-point value
/// array — the generic contraction the (multi-degree) sumcheck provers invoke.
pub type CombFn<F> = Box<dyn Fn(&[F]) -> F + Send + Sync>;
