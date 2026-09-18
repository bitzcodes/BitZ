#![cfg(feature = "bench-peak-memory")]

#[path = "../benches/common/peak_memory.rs"]
mod peak_memory;

use std::{
    alloc::{GlobalAlloc, Layout},
    sync::Barrier,
};

#[test]
fn tracks_zeroed_reallocated_and_parallel_allocations_across_peak_resets() {
    let allocator = peak_memory::PeakAlloc;
    let baseline = peak_memory::live_bytes();
    peak_memory::reset_peak();
    let small = Layout::from_size_align(64, 16).unwrap();
    // SAFETY: these allocations remain owned here and are released with the
    // layout matching the most recent successful allocation/reallocation.
    unsafe {
        let pointer = allocator.alloc_zeroed(small);
        assert!(!pointer.is_null());
        assert!(
            std::slice::from_raw_parts(pointer, 64)
                .iter()
                .all(|&b| b == 0)
        );
        pointer.write_bytes(0xa5, 64);
        assert_eq!(peak_memory::live_bytes(), baseline + 64);

        let pointer = allocator.realloc(pointer, small, 256);
        assert!(!pointer.is_null());
        assert!(
            std::slice::from_raw_parts(pointer, 64)
                .iter()
                .all(|&b| b == 0xa5)
        );
        assert_eq!(peak_memory::live_bytes(), baseline + 256);
        assert_eq!(peak_memory::peak_bytes(), baseline + 256);

        let pointer = allocator.realloc(pointer, Layout::from_size_align(256, 16).unwrap(), 128);
        assert!(!pointer.is_null());
        assert_eq!(peak_memory::live_bytes(), baseline + 128);
        assert_eq!(peak_memory::peak_bytes(), baseline + 256);
        peak_memory::reset_peak();
        assert_eq!(peak_memory::peak_bytes(), baseline + 128);
        allocator.dealloc(pointer, Layout::from_size_align(128, 16).unwrap());
    }
    assert_eq!(peak_memory::live_bytes(), baseline);

    peak_memory::reset_peak();
    let barrier = Barrier::new(8);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                let layout = Layout::from_size_align(4096, 16).unwrap();
                // SAFETY: each thread exclusively owns its allocation.
                unsafe {
                    let pointer = allocator.alloc(layout);
                    assert!(!pointer.is_null());
                    barrier.wait();
                    allocator.dealloc(pointer, layout);
                }
            });
        }
    });
    assert_eq!(peak_memory::live_bytes(), baseline);
    assert_eq!(peak_memory::peak_bytes(), baseline + 8 * 4096);
    peak_memory::reset_peak();
    assert_eq!(peak_memory::peak_bytes(), baseline);
}
