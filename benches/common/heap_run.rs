//! Whole-process heap accounting for separate, instrumented benchmark runs.
//! These numbers include preparation and trace collection; they are not timings.
pub struct Report;
impl Report {
    pub fn start() -> Self {
        super::peak_memory::reset_peak();
        Self
    }
}
impl Drop for Report {
    fn drop(&mut self) {
        eprintln!("HEAP_RUN peak_bytes={} live_bytes={}", super::peak_memory::peak_bytes(), super::peak_memory::live_bytes());
    }
}
