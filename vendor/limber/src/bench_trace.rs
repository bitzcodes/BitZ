//! Exact, opt-in interval capture for benchmark campaigns.
//!
//! This module is intentionally small and inert unless [`set_enabled`] is
//! called.  Production protocol paths keep using their existing tracing spans;
//! [`crate::start_span!`] mirrors those spans into this recorder only while a
//! benchmark trial is active.

use std::{
  cell::{Cell, RefCell},
  time::Instant,
};

#[derive(Clone, Copy)]
struct Frame {
  label: &'static str,
  start: Instant,
  start_ns: u64,
  depth: usize,
  order: u64,
  parent_order: Option<u64>,
}

/// One observed half-open interval in a benchmark trial.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfileInterval {
  /// Static operation label supplied by the instrumentation site.
  pub label: &'static str,
  /// Start offset from the trial's monotonic epoch.
  pub start_ns: u64,
  /// End offset from the trial's monotonic epoch.
  pub end_ns: u64,
  /// Nesting depth on the control thread.
  pub depth: usize,
  /// Monotonic entry order, unique within one drained trial.
  pub order: u64,
  /// Entry order of the enclosing interval, if any.
  pub parent_order: Option<u64>,
}

thread_local! {
  static ENABLED: Cell<bool> = const { Cell::new(false) };
  static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
  static INTERVALS: RefCell<Vec<ProfileInterval>> = const { RefCell::new(Vec::new()) };
  static EPOCH: RefCell<Option<Instant>> = const { RefCell::new(None) };
  static ORDER: Cell<u64> = const { Cell::new(0) };
}

/// Enable or disable exact interval capture on the current control thread.
///
/// Worker-thread spans remain available to the normal tracing subscriber, but
/// are deliberately excluded here because this recorder has one thread-local
/// clock, stack, and output lane per benchmark trial.
pub fn set_enabled(enabled: bool) {
  ENABLED.with(|flag| flag.set(enabled));
}

/// RAII interval guard.  Dropping it records the scope's exact nanosecond
/// bounds when capture is enabled.
#[must_use = "the interval is recorded only while this guard is alive"]
pub struct Scope {
  order: Option<u64>,
}

impl Drop for Scope {
  fn drop(&mut self) {
    let Some(expected_order) = self.order.take() else {
      return;
    };
    let frame = STACK
      .with(|stack| stack.borrow_mut().pop())
      .expect("benchmark interval guard dropped with an empty scope stack");
    assert_eq!(
      frame.order, expected_order,
      "benchmark interval guards must close in LIFO order"
    );
    let elapsed_ns = u64::try_from(frame.start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    INTERVALS.with(|intervals| {
      intervals.borrow_mut().push(ProfileInterval {
        label: frame.label,
        start_ns: frame.start_ns,
        end_ns: frame.start_ns.saturating_add(elapsed_ns),
        depth: frame.depth,
        order: frame.order,
        parent_order: frame.parent_order,
      });
    });
  }
}

/// Open an exact interval with `label`.
#[inline]
pub fn scope(label: &'static str) -> Scope {
  if !ENABLED.with(Cell::get) {
    return Scope { order: None };
  }
  let start = Instant::now();
  let start_ns = EPOCH.with(|epoch| {
    let mut epoch = epoch.borrow_mut();
    let origin = epoch.get_or_insert(start);
    u64::try_from(start.duration_since(*origin).as_nanos()).unwrap_or(u64::MAX)
  });
  let order = ORDER.with(|order| {
    let current = order.get();
    order.set(current + 1);
    current
  });
  STACK.with(|stack| {
    let mut stack = stack.borrow_mut();
    let depth = stack.len();
    let parent_order = stack.last().map(|frame| frame.order);
    stack.push(Frame {
      label,
      start,
      start_ns,
      depth,
      order,
      parent_order,
    });
  });
  Scope { order: Some(order) }
}

/// Drain completed intervals in entry order and reset the trial epoch.
///
/// Call this only after the root trial scope has dropped.
pub fn take_intervals() -> Vec<ProfileInterval> {
  let stack_empty = STACK.with(|stack| stack.borrow().is_empty());
  assert!(
    stack_empty,
    "cannot drain benchmark intervals while scopes are open"
  );
  let mut intervals = INTERVALS.with(|items| std::mem::take(&mut *items.borrow_mut()));
  intervals.sort_by_key(|interval| interval.order);
  ORDER.with(|order| order.set(0));
  EPOCH.with(|epoch| *epoch.borrow_mut() = None);
  intervals
}

/// Guard that keeps a normal `tracing` span and its matching exact interval
/// alive for the same lexical region.
#[must_use = "dropping the guard closes both tracing spans"]
pub(crate) struct SpanGuard {
  _tracing: tracing::span::EnteredSpan,
  _interval: Scope,
}

impl SpanGuard {
  pub(crate) fn new(tracing: tracing::span::EnteredSpan, interval: Scope) -> Self {
    Self {
      _tracing: tracing,
      _interval: interval,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn captures_nested_half_open_intervals() {
    set_enabled(true);
    {
      let _outer = scope("outer");
      let _inner = scope("inner");
    }
    let intervals = take_intervals();
    set_enabled(false);
    assert_eq!(intervals.len(), 2);
    assert_eq!(intervals[0].label, "outer");
    assert_eq!(intervals[1].parent_order, Some(intervals[0].order));
    assert!(intervals[0].start_ns <= intervals[1].start_ns);
    assert!(intervals[1].end_ns <= intervals[0].end_ns);
  }

  #[test]
  fn explicitly_closed_phases_are_siblings() {
    set_enabled(true);
    {
      let _root = scope("root");
      let (projection, _) = crate::start_span!("test_projection");
      drop(projection);
      let (piop, _) = crate::start_span!("test_piop");
      drop(piop);
      let (opening, _) = crate::start_span!("test_opening");
      drop(opening);
    }
    let intervals = take_intervals();
    set_enabled(false);
    assert_eq!(intervals.len(), 4);
    let root = intervals[0];
    for phase in &intervals[1..] {
      assert_eq!(phase.parent_order, Some(root.order));
    }
    assert!(intervals[1].end_ns <= intervals[2].start_ns);
    assert!(intervals[2].end_ns <= intervals[3].start_ns);
  }
}
