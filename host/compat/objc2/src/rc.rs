//! `Retained<T>`: an `Arc<T>` clone, plus the small API uzu touches.

use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

#[repr(transparent)]
pub struct Retained<T: ?Sized>(Arc<T>);

impl<T: ?Sized> Retained<T> {
    /// Address of the shared object (uzu uses it as a stable identity key).
    pub fn as_ptr(this: &Self) -> *const T {
        Arc::as_ptr(&this.0)
    }

    /// The shared-ownership handle itself.
    pub fn as_arc(&self) -> &Arc<T> {
        &self.0
    }

    /// Build from an existing shared handle.
    pub fn from_arc(arc: Arc<T>) -> Self {
        Self(arc)
    }

    /// Recover the shared handle (the built-in unsizing coercion applies at
    /// `from_arc`'s argument position when the target is a trait object).
    pub fn into_arc(self) -> Arc<T> {
        self.0
    }

    /// Raw pointer of the shared object.
    #[allow(dead_code)]
    pub fn as_raw(&self) -> *const T {
        Arc::as_ptr(&self.0)
    }
}

impl<T> Retained<T> {
    pub fn new(value: T) -> Self {
        Self(Arc::new(value))
    }

    /// Shared-ownership cast across protocol objects (unused on Linux today;
    /// kept for API parity with `Retained::cast`).
    #[allow(dead_code)]
    pub fn cast<U: From<Arc<T>>>(self) -> U {
        U::from(Arc::clone(&self.0))
    }
}

impl<T: ?Sized> Clone for Retained<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T: ?Sized> Deref for Retained<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: ?Sized> AsRef<T> for Retained<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T: ?Sized> Borrow<T> for Retained<T> {
    fn borrow(&self) -> &T {
        &self.0
    }
}

impl<T: ?Sized> fmt::Debug for Retained<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Retained").field("ptr", &Arc::as_ptr(&self.0)).finish()
    }
}

unsafe impl<T: ?Sized + Send + Sync> Send for Retained<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for Retained<T> {}
