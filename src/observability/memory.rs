//! Optional peak-RSS observations. This layer never reads a clock or measures time.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
    thread::ThreadId,
};
use tracing::{
    Subscriber,
    field::{Field, Visit},
    span::{Attributes, Id},
};
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PeakGrowth {
    pub bytes: u64,
    pub peak_bytes: u64,
}

#[derive(Clone)]
pub struct MemoryLayer {
    probe: fn() -> u64,
    completed: Arc<Mutex<BTreeMap<String, PeakGrowth>>>,
}

impl MemoryLayer {
    /// The probe returns the process peak RSS in bytes, not current live memory.
    /// Nested regions include their children's growth; these values are not additive.
    pub fn new(probe: fn() -> u64) -> Self {
        Self {
            probe,
            completed: Arc::default(),
        }
    }

    pub fn take(&self) -> BTreeMap<String, PeakGrowth> {
        std::mem::take(&mut *self.completed.lock().expect("memory observations poisoned"))
    }
}

struct Entries {
    label: String,
    starts: HashMap<ThreadId, Vec<u64>>,
}

struct Component(Option<String>);
impl Visit for Component {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "component" {
            self.0 = Some(value.into());
        }
    }
    fn record_debug(&mut self, _: &Field, _: &dyn std::fmt::Debug) {}
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for MemoryLayer {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut component = Component(None);
        attrs.record(&mut component);
        ctx.span(id)
            .expect("new span")
            .extensions_mut()
            .insert(Entries {
                label: component
                    .0
                    .unwrap_or_else(|| attrs.metadata().name().into()),
                starts: HashMap::new(),
            });
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let peak = (self.probe)();
        ctx.span(id)
            .expect("entered span")
            .extensions_mut()
            .get_mut::<Entries>()
            .expect("memory span state")
            .starts
            .entry(std::thread::current().id())
            .or_default()
            .push(peak);
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let peak = (self.probe)();
        let span = ctx.span(id).expect("exited span");
        let mut extensions = span.extensions_mut();
        let entries = extensions.get_mut::<Entries>().expect("memory span state");
        let start = entries
            .starts
            .get_mut(&std::thread::current().id())
            .and_then(Vec::pop)
            .expect("balanced memory span entry");
        let mut completed = self.completed.lock().expect("memory observations poisoned");
        let row = completed.entry(entries.label.clone()).or_default();
        row.bytes += peak.saturating_sub(start);
        row.peak_bytes = row.peak_bytes.max(peak);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tracing_subscriber::prelude::*;

    #[test]
    fn nested_and_reentered_spans_preserve_peak_growth_without_clocks() {
        static PEAK: AtomicU64 = AtomicU64::new(100);
        let memory = MemoryLayer::new(|| PEAK.load(Ordering::Relaxed));
        tracing::subscriber::with_default(
            tracing_subscriber::registry().with(memory.clone()),
            || {
                let outer = tracing::info_span!("outer").entered();
                PEAK.store(110, Ordering::Relaxed);
                let child = tracing::info_span!("child", component = "child.work");
                child.in_scope(|| PEAK.store(120, Ordering::Relaxed));
                child.in_scope(|| PEAK.store(125, Ordering::Relaxed));
                drop(outer);
            },
        );
        let rows = memory.take();
        assert_eq!(
            rows["outer"],
            PeakGrowth {
                bytes: 25,
                peak_bytes: 125
            }
        );
        assert_eq!(
            rows["child.work"],
            PeakGrowth {
                bytes: 15,
                peak_bytes: 125
            }
        );
        assert!(memory.take().is_empty());
    }
}
