use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{Device, Error, Result};

/// `.m2vlib` container magic ("M2VL") and format version.
pub const M2VLIB_MAGIC: [u8; 4] = *b"M2VL";
pub const M2VLIB_VERSION: u32 = 1;

/// Where one function's SPIR-V blob lives in the container (bytes from the start).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpvRange {
    pub offset: u64,
    pub len: u64,
}

/// Metal argument-table slot -> descriptor binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferBindingJson {
    pub arg: u32,
    pub binding: u32,
    pub set: u32,
}

/// POD argument -> push constant offset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushWordJson {
    pub arg: u32,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PushJson {
    pub size: u32,
    #[serde(default)]
    pub words: Vec<PushWordJson>,
}

/// Metal function constant index -> SpecId.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantJson {
    pub index: u32,
    pub spec_id: u32,
    #[serde(rename = "type")]
    pub ty: ConstantType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstantType {
    #[serde(rename = "bool")]
    Bool,
    #[serde(rename = "u32")]
    U32,
    #[serde(rename = "i32")]
    I32,
    #[serde(rename = "f32")]
    F32,
}

/// Dynamic threadgroup memory: OpenCL local pointer args become spec-sized arrays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadgroupJson {
    pub arg: u32,
    pub spec_id: u32,
}

/// One function in the `.m2vlib` JSON section. Layout contract in `host/DESIGN.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionJson {
    pub name: String,
    pub entry: String,
    pub spv: SpvRange,
    pub workgroup_size: [u32; 3],
    #[serde(default)]
    pub workgroup_size_spec_ids: Option<[u32; 3]>,
    #[serde(default)]
    pub buffers: Vec<BufferBindingJson>,
    #[serde(default)]
    pub push: PushJson,
    #[serde(default)]
    pub constants: Vec<ConstantJson>,
    #[serde(default)]
    pub threadgroup: Vec<ThreadgroupJson>,
    #[serde(default)]
    pub uses_coopmat: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M2vLibJson {
    pub functions: Vec<FunctionJson>,
}

/// A Metal function constant value bound at pipeline creation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstantValue {
    Bool(bool),
    U32(u32),
    I32(i32),
    F32(f32),
}

/// A parsed `.m2vlib`: JSON description plus the SPIR-V blobs it points at.
pub struct Library {
    _priv: (),
}

impl Library {
    /// Parse a `.m2vlib` (magic `M2VL`, version 1; JSON section, then aligned SPIR-V blobs).
    pub fn from_bytes(device: &Arc<Device>, bytes: &[u8]) -> Result<Arc<Library>> {
        let _ = (device, bytes);
        Err(Error::Unsupported("Library::from_bytes".into()))
    }

    /// Metal function names available for [`Device::create_pipeline`].
    pub fn function_names(&self) -> Vec<String> {
        unimplemented_stub("Library::function_names")
    }
}

fn unimplemented_stub<T>(what: &str) -> T {
    // temporary: bodies are filled in by the implementation commit
    let _ = what;
    unreachable!()
}
