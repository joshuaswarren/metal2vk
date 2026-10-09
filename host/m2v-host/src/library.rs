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

/// One function in the `.m2vlib` JSON section. Layout contract in `host/DESIGN.md`;
/// `host/tools/make-m2vlib.py` is the reference emitter.
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
    pub(crate) desc: M2vLibJson,
    /// Whole container; SPIR-V ranges index into it.
    pub(crate) bytes: Arc<[u8]>,
}

impl Library {
    /// Parse a `.m2vlib` (magic `M2VL`, version 1; JSON section, then aligned SPIR-V blobs).
    pub fn from_bytes(_device: &Arc<Device>, bytes: &[u8]) -> Result<Arc<Library>> {
        Self::from_bytes_inner(bytes)
    }

    pub(crate) fn from_bytes_inner(bytes: &[u8]) -> Result<Arc<Library>> {
        let desc = parse_container(bytes)?;
        Ok(Arc::new(Library { desc, bytes: bytes.into() }))
    }

    /// Metal function names available for [`Device::create_pipeline`].
    pub fn function_names(&self) -> Vec<String> {
        self.desc.functions.iter().map(|f| f.name.clone()).collect()
    }

    pub(crate) fn function(&self, name: &str) -> Result<&FunctionJson> {
        self.desc
            .functions
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "no function {name:?} in library (have: {})",
                    self.function_names().join(", ")
                ))
            })
    }

    pub(crate) fn spv_blob(&self, f: &FunctionJson) -> Result<&[u8]> {
        let start = f.spv.offset as usize;
        let end = start.checked_add(f.spv.len as usize).ok_or_else(|| {
            Error::Invalid(format!("function {:?}: SPIR-V range overflows", f.name))
        })?;
        self.bytes
            .get(start..end)
            .ok_or_else(|| Error::Invalid(format!("function {:?}: SPIR-V range outside container", f.name)))
    }
}

/// Parse the container, checking structure only (no device needed).
/// Split out for GPU-free tests and for the C ABI's pre-validation.
pub(crate) fn parse_container(bytes: &[u8]) -> Result<M2vLibJson> {
    if bytes.len() < 12 {
        return Err(Error::Invalid(format!(
            "m2vlib: {} bytes is shorter than the 12-byte header",
            bytes.len()
        )));
    }
    if bytes[0..4] != M2VLIB_MAGIC {
        return Err(Error::Invalid(format!(
            "m2vlib: bad magic {:?} (expected {:?})",
            &bytes[0..4], M2VLIB_MAGIC
        )));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != M2VLIB_VERSION {
        return Err(Error::Invalid(format!(
            "m2vlib: unsupported version {version} (expected {})",
            M2VLIB_VERSION
        )));
    }
    let json_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let json_end = 12 + json_len;
    if bytes.len() < json_end {
        return Err(Error::Invalid(format!(
            "m2vlib: JSON section truncated (header says {json_len} bytes, container has {})",
            bytes.len() - 12
        )));
    }
    let json: M2vLibJson = serde_json::from_slice(&bytes[12..json_end])
        .map_err(|e| Error::Invalid(format!("m2vlib: JSON section: {e}")))?;
    if json.functions.is_empty() {
        return Err(Error::Invalid("m2vlib: no functions".into()));
    }
    // Validate every SPIR-V range now, so dispatch never sees a bad blob.
    for f in &json.functions {
        let start = f.spv.offset as usize;
        let end = start
            .checked_add(f.spv.len as usize)
            .ok_or_else(|| Error::Invalid(format!("m2vlib: function {:?}: SPIR-V range overflows", f.name)))?;
        if end > bytes.len() {
            return Err(Error::Invalid(format!(
                "m2vlib: function {:?}: SPIR-V range [{start},{end}) outside container (len {})",
                f.name,
                bytes.len()
            )));
        }
        if f.spv.len % 4 != 0 || f.spv.offset % 4 != 0 {
            return Err(Error::Invalid(format!(
                "m2vlib: function {:?}: SPIR-V offset/len must be 4-byte aligned",
                f.name
            )));
        }
    }
    Ok(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_spv(len: u32) -> Vec<u8> {
        // plausible enough for container tests; magic + version like real SPIR-V
        let mut v = vec![0x03, 0x02, 0x23, 0x07];
        v.resize(len as usize, 0xAB);
        v
    }

    fn container(json: &str, spv: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&M2VLIB_MAGIC);
        v.extend_from_slice(&M2VLIB_VERSION.to_le_bytes());
        v.extend_from_slice(&(json.len() as u32).to_le_bytes());
        v.extend_from_slice(json.as_bytes());
        while v.len() % 4 != 0 {
            v.push(0);
        }
        v.extend_from_slice(spv);
        v
    }

    fn sample_json(spv_off: u64, spv_len: u64) -> String {
        r#"{"functions":[{"name":"add","entry":"add","spv":{"offset":OFF,"len":LEN},
            "workgroup_size":[32,1,1],"workgroup_size_spec_ids":null,
            "buffers":[{"arg":0,"binding":0,"set":0},{"arg":1,"binding":1,"set":0}],
            "push":{"size":16,"words":[{"arg":2,"offset":16,"size":4}]},
            "constants":[{"index":0,"spec_id":3,"type":"f32"}],
            "threadgroup":[],"uses_coopmat":false}]}"#
            .replace("OFF", &spv_off.to_string())
            .replace("LEN", &spv_len.to_string())
    }

    #[test]
    fn round_trip_and_lookup() {
        let spv = fake_spv(128);
        // The JSON encodes the SPIR-V offset, which depends on the JSON length:
        // iterate to the fixed point.
        let mut spv_off = 0u64;
        for _ in 0..4 {
            let j = sample_json(spv_off, spv.len() as u64);
            let next = (12 + j.len()).next_multiple_of(4) as u64;
            if next == spv_off {
                break;
            }
            spv_off = next;
        }
        let bytes = container(&sample_json(spv_off, spv.len() as u64), &spv);
        let lib = Library::from_bytes_inner(&bytes).expect("valid container");
        assert_eq!(lib.function_names(), vec!["add".to_string()]);
        let f = lib.function("add").unwrap();
        assert_eq!(f.workgroup_size, [32, 1, 1]);
        assert_eq!(f.push.words[0].offset, 16);
        assert_eq!(f.constants[0].ty, ConstantType::F32);
        assert_eq!(lib.spv_blob(f).unwrap(), spv.as_slice());
        assert_eq!(f.spv.offset, spv_off);
    }

    #[test]
    fn bad_magic_names_the_magic() {
        let mut bytes = container(&sample_json(16, 128), &fake_spv(128));
        bytes[0] = b'X';
        let err = parse_container(&bytes).unwrap_err().to_string();
        assert!(err.contains("magic"), "error should mention magic: {err}");
        assert!(err.contains("bad magic"));
    }

    #[test]
    fn bad_version_names_the_version() {
        let mut bytes = container(&sample_json(16, 128), &fake_spv(128));
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = parse_container(&bytes).unwrap_err().to_string();
        assert!(err.contains("version 99"), "error should name the version: {err}");
    }

    #[test]
    fn truncated_json_and_out_of_range_blob_are_rejected() {
        let bytes = container(&sample_json(16, 128), &fake_spv(128));
        let err = parse_container(&bytes[..20]).unwrap_err().to_string();
        assert!(err.contains("truncated"), "{err}");

        let big = sample_json(16, 1 << 20);
        let err = parse_container(&container(&big, &fake_spv(128)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside container"), "{err}");

        let off_end = sample_json(1 << 20, 128);
        let err = parse_container(&container(&off_end, &fake_spv(128)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside container"), "{err}");
    }

    #[test]
    fn unaligned_spv_rejected() {
        let json = sample_json(17, 128); // odd offset
        let err = parse_container(&container(&json, &fake_spv(128)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("aligned"), "{err}");
    }
}
