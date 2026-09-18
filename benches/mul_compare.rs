mod common;
mod mul;
#[cfg(feature = "bench-peak-memory")]
#[global_allocator]
static ALLOCATOR: common::peak_memory::PeakAlloc = common::peak_memory::PeakAlloc;
fn main() -> anyhow::Result<()> {
    mul::main(true)
}
