use std::fmt;

/// Error type for every fallible operation in this crate.
#[derive(Debug)]
pub enum Error {
    /// Requested capability is not implemented or not offered by the device.
    /// No silent no-ops: every unimplemented API surface returns this.
    Unsupported(String),
    /// Bad argument or malformed input (e.g. a corrupt `.m2vlib`).
    Invalid(String),
    /// The routing policy refused to run a kernel: it has no trusted translation and no fallback. Names the kernel.
    Refused(String),
    /// Underlying Vulkan call failed.
    Vulkan(ash::vk::Result),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Invalid(m) => write!(f, "invalid: {m}"),
            Error::Refused(m) => write!(f, "refused: {m}"),
            Error::Vulkan(r) => write!(f, "vulkan error: {r:?}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<ash::vk::Result> for Error {
    fn from(r: ash::vk::Result) -> Self {
        Error::Vulkan(r)
    }
}
