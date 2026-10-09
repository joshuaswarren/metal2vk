//! `ProtocolObject<P>` / `AnyObject` / `AnyThread` / `NSObjectProtocol` stand-ins.

use std::fmt;
use std::marker::PhantomData;

/// Transparent wrapper whose only field is `P`, so `ProtocolObject<Concrete>`
/// unsizes to `ProtocolObject<dyn Trait>` whenever `Concrete: Trait`.
///
/// Values are only ever created by the backing implementation (`mtl-rs` shims);
/// uzu only ever sees `ProtocolObject<dyn ...>` behind a [`Retained`](crate::Retained).
#[repr(transparent)]
pub struct ProtocolObject<P: ?Sized> {
    inner: P,
}

impl<P: ?Sized> ProtocolObject<P> {
    /// Wrap a concrete value into the protocol object. `ProtocolObject<Concrete>`
    /// coerces to `ProtocolObject<dyn Trait>` at the call site.
    pub fn from_inner(inner: P) -> Self
    where
        P: Sized,
    {
        Self { inner }
    }

    pub fn as_inner(&self) -> &P {
        &self.inner
    }
}

impl<P: ?Sized> fmt::Debug for ProtocolObject<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ProtocolObject")
    }
}

/// Untyped object reference (capture descriptors hold one of these).
#[repr(transparent)]
pub struct AnyObject {
    _priv: (),
}

/// Marker proving the object is safe to touch from any thread. Zero-sized;
/// `Retained<T>` is already `Send + Sync` whenever `T` is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AnyThread(pub PhantomData<()>);

/// Protocol conformance marker. The shim's traits require it like upstream.
pub unsafe trait NSObjectProtocol {}

unsafe impl<P: ?Sized> NSObjectProtocol for ProtocolObject<P> {}

unsafe impl<P: ?Sized> super::Message for ProtocolObject<P> {}

impl<P: ?Sized> ProtocolObject<P> {
    /// Address of the wrapped object, for pointer-identity maps. In this
    /// compat layer `ProtocolObject<dyn Trait>` pointers are wide; the object
    /// address is their data word.
    pub fn host_addr(this: &Self) -> usize {
        // A cast from a wide pointer to a thin one drops the metadata and keeps the data address.
        // No assumption about the layout of the wide pointer.
        this as *const ProtocolObject<P> as *const () as usize
    }
}
