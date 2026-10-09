use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::Buffer;

/// Base of the shim-owned virtual address space. The values are opaque handles;
/// only uniqueness, 16 KiB alignment and interval lookup matter.
const VA_BASE: u64 = 0x0000_4000_0000;
/// Metal contract: address + offset stays inside one buffer, so every buffer
/// gets its own 16 KiB-aligned region, never reused while the buffer lives.
pub const VA_ALIGN: u64 = 16 * 1024;

struct AddrEntry {
    buffer: Weak<Buffer>,
    len: u64,
}

/// Interval map from shim-owned VAs to live buffers. Pure logic, unit tested
/// without a GPU; `Device` owns one behind an Arc.
pub(crate) struct AddressMap {
    ranges: Mutex<BTreeMap<u64, AddrEntry>>,
    next_va: AtomicU64,
}

impl AddressMap {
    pub(crate) fn new() -> Self {
        AddressMap {
            ranges: Mutex::new(BTreeMap::new()),
            next_va: AtomicU64::new(VA_BASE),
        }
    }

    /// Reserve an aligned region; returns its base address. Call
    /// [`AddressMap::register`] once the buffer Arc exists (weak references
    /// would make `Arc::get_mut` fail, so reserve before publishing the Arc).
    pub(crate) fn reserve(&self, len: u64) -> u64 {
        let len = len.max(1);
        self.next_va.fetch_add(Self::va_span(len), Ordering::Relaxed)
    }

    pub(crate) fn register(self: &Arc<Self>, va: u64, buffer: &Arc<Buffer>, len: u64) {
        self.ranges
            .lock()
            .insert(va, AddrEntry { buffer: Arc::downgrade(buffer), len: len.max(1) });
    }

    fn va_span(len: u64) -> u64 {
        len.div_ceil(VA_ALIGN).max(1) * VA_ALIGN
    }

    pub(crate) fn remove(&self, va: u64) {
        self.ranges.lock().remove(&va);
    }

    /// Which live buffer owns `addr`, and the offset into it.
    pub(crate) fn resolve(&self, addr: u64) -> Option<(Arc<Buffer>, u64)> {
        let ranges = self.ranges.lock();
        let (&va, entry) = ranges.range(..=addr).next_back()?;
        if addr >= va + entry.len {
            return None;
        }
        let buffer = entry.buffer.upgrade()?;
        Some((buffer, addr - va))
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf() -> Arc<Buffer> {
        // AddressMap only holds Weak<Buffer>; contents are never touched here.
        Arc::new(Buffer { inner: crate::buffer::Buffer::test_probe() })
    }

    #[test]
    fn va_alignment_and_uniqueness() {
        let map = Arc::new(AddressMap::new());
        let a = buf();
        let b = buf();
        let va_a = map.reserve(1); // rounds up to one 16 KiB block
        map.register(va_a, &a, 1);
        let va_b = map.reserve(VA_ALIGN * 3);
        map.register(va_b, &b, VA_ALIGN * 3);
        assert_eq!(va_a % VA_ALIGN, 0);
        assert_eq!(va_b % VA_ALIGN, 0);
        assert_eq!(va_b - va_a, VA_ALIGN); // 1-byte buffer occupies one block
        let (got, off) = map.resolve(va_a).unwrap();
        assert!(Arc::ptr_eq(&got, &a));
        assert_eq!(off, 0);
        let (got, off) = map.resolve(va_b + VA_ALIGN * 2 + 3).unwrap();
        assert!(Arc::ptr_eq(&got, &b));
        assert_eq!(off, VA_ALIGN * 2 + 3);
        assert!(
            map.resolve(va_a + 1).is_none(),
            "offset inside the reserved block but outside the 1-byte buffer"
        );
    }

    #[test]
    fn resolve_misses_outside_ranges() {
        let map = Arc::new(AddressMap::new());
        let a = buf();
        let va = map.reserve(1024);
        map.register(va, &a, 1024);
        assert!(map.resolve(va + 1024).is_none(), "one past the end");
        assert!(map.resolve(va - 1).is_none(), "before any range");
        assert!(map.resolve(1).is_none());
    }

    #[test]
    fn dropped_buffers_do_not_resolve_and_vas_are_not_reused() {
        let map = Arc::new(AddressMap::new());
        let va;
        {
            let a = buf();
            va = map.reserve(4096);
            map.register(va, &a, 4096);
        }
        assert!(map.resolve(va).is_none(), "dropped buffer must not resolve");
        let b = buf();
        let va2 = map.reserve(4096);
        map.register(va2, &b, 4096);
        assert!(va2 > va, "address space never rewinds, regions never reused");
        let (got, off) = map.resolve(va2 + 4095).unwrap();
        assert!(Arc::ptr_eq(&got, &b));
        assert_eq!(off, 4095);
    }

    #[test]
    fn adjacent_ranges_resolve_to_their_owner() {
        let map = Arc::new(AddressMap::new());
        let a = buf();
        let b = buf();
        let va_a = map.reserve(VA_ALIGN);
        map.register(va_a, &a, VA_ALIGN);
        let va_b = map.reserve(VA_ALIGN);
        map.register(va_b, &b, VA_ALIGN);
        assert_eq!(va_b, va_a + VA_ALIGN);
        let (got, _) = map.resolve(va_a + VA_ALIGN - 1).unwrap();
        assert!(Arc::ptr_eq(&got, &a));
        let (got, _) = map.resolve(va_b).unwrap();
        assert!(Arc::ptr_eq(&got, &b));
    }
}

#[allow(dead_code)]
fn _assert_send_sync() {
    fn assert<T: Send + Sync>() {}
    assert::<AddressMap>();
}
