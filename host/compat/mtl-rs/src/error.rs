//! Plain Rust error replacing `NSError`. uzu only ever stringifies it.

use std::fmt;

#[derive(Debug, Clone)]
pub struct MetalError {
    pub message: String,
}

impl MetalError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MetalError {}

impl From<metal2vk::Error> for MetalError {
    fn from(error: metal2vk::Error) -> Self {
        Self::new(error.to_string())
    }
}
