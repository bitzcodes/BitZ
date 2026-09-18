//! Opt-in phase timings for the benchmark driver, excluded from normal builds.
use std::{cell::Cell, sync::OnceLock, time::Instant};
thread_local! { static NANOS: Cell<[u64;4]> = const { Cell::new([0;4]) }; }
fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("OUTER_PHASES").is_some())
}
pub(crate) fn reset() {
    NANOS.set([0; 4]);
}
pub(crate) fn take() -> Option<[u64; 4]> {
    enabled().then(|| NANOS.replace([0; 4]))
}
pub(super) struct Phase {
    index: usize,
    start: Option<Instant>,
}
impl Phase {
    pub(super) fn start(index: usize) -> Self {
        Self {
            index,
            start: enabled().then(Instant::now),
        }
    }
}
impl Drop for Phase {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            NANOS.with(|cell| {
                let mut values = cell.get();
                values[self.index] += start.elapsed().as_nanos() as u64;
                cell.set(values);
            });
        }
    }
}
