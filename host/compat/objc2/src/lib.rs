//! Linux stand-in for the `objc2` surface that uzu's Metal backend consumes.
//!
//! `Retained<T>` is an `Arc<T>` clone; `ProtocolObject<P>` is a `#[repr(transparent)]`
//! wrapper whose last field is `P`, so `Arc<ProtocolObject<Concrete>>` unsizes to
//! `Arc<ProtocolObject<dyn Trait>>` exactly like the Objective-C original.
//! Everything is safe Rust: there is no Objective-C runtime here.
//!
//! uzu's `metal_extensions/device_extensions.rs` (the raw `objc_msgSend` path for
//! obfuscated device selectors) is Apple-only; on Linux the `metal` crate's
//! `MTLDeviceExt` answers those queries from the host device instead.

pub mod rc;
pub mod runtime;

/// Marker for the `Message` bound uzu's Apple-side code carries. No runtime semantics.
pub unsafe trait Message {}

/// Run `f` immediately. No autorelease pool exists outside Objective-C.
pub fn autoreleasepool<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}

pub use rc::Retained;
pub use runtime::{AnyObject, AnyThread, NSObjectProtocol, ProtocolObject};
