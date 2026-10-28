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

/// Whether a translated kernel has passed its correctness check against a reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateState {
    Verified,
    #[default]
    Unverified,
    /// Mismatched its reference: the host never runs it and routes to its fallback.
    Failed,
}

/// Per-function verification gate. `evidence` names the check that justifies `Verified`
/// (a test name, receipt path or commit); it is empty for `Unverified`. Libraries written
/// before the field existed read as unverified.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateJson {
    #[serde(default)]
    pub state: GateState,
    #[serde(default)]
    pub evidence: String,
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
    /// Verification gate; defaults to unverified when the library does not carry it.
    #[serde(default)]
    pub gate: GateJson,
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

/// Parse the container, checking its structure and the reflection metadata the host
/// consumes (no device needed). Split out for GPU-free tests and for the C ABI's
/// pre-validation, so both paths reject the same inputs.
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
        check_reflection(f)?;
    }
    Ok(json)
}

/// Reflection metadata that pipeline creation and the encoder trust: buffers use set 0
/// (the only set the pipeline layout creates) with distinct bindings, push words lie
/// inside the declared push block (the encoder slices the push buffer with them), and
/// no argument slot appears twice in a lookup table (`set_address` / `set_bytes`).
fn check_reflection(f: &FunctionJson) -> Result<()> {
    let name = &f.name;
    for (i, b) in f.buffers.iter().enumerate() {
        if b.set != 0 {
            return Err(Error::Invalid(format!(
                "m2vlib: function {name:?}: buffer argument {} uses descriptor set {} (the host supports set 0 only)",
                b.arg, b.set
            )));
        }
        let earlier = &f.buffers[..i];
        if let Some(prev) = earlier.iter().find(|p| p.arg == b.arg) {
            return Err(Error::Invalid(format!(
                "m2vlib: function {name:?}: buffer argument {} is declared twice (bindings {} and {})",
                b.arg, prev.binding, b.binding
            )));
        }
        if let Some(prev) = earlier.iter().find(|p| (p.set, p.binding) == (b.set, b.binding)) {
            return Err(Error::Invalid(format!(
                "m2vlib: function {name:?}: buffer arguments {} and {} both map to set {} binding {}",
                prev.arg, b.arg, b.set, b.binding
            )));
        }
    }
    let block = f.push.size;
    for (i, w) in f.push.words.iter().enumerate() {
        let Some(end) = w.offset.checked_add(w.size) else {
            return Err(Error::Invalid(format!(
                "m2vlib: function {name:?}: push word for argument {} at offset {}, size {} overflows u32",
                w.arg, w.offset, w.size
            )));
        };
        if end > block {
            return Err(Error::Invalid(format!(
                "m2vlib: function {name:?}: push word for argument {} at offset {}, size {} ends past the {block}-byte push block",
                w.arg, w.offset, w.size
            )));
        }
        if let Some(prev) = f.push.words[..i].iter().find(|p| p.arg == w.arg) {
            return Err(Error::Invalid(format!(
                "m2vlib: function {name:?}: push-constant argument {} is declared twice (offsets {} and {})",
                w.arg, prev.offset, w.offset
            )));
        }
    }
    Ok(())
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
            "push":{"size":20,"words":[{"arg":2,"offset":16,"size":4}]},
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
        assert_eq!(f.push.size, 20);
        assert_eq!(f.push.words[0].offset, 16);
        assert_eq!(f.constants[0].ty, ConstantType::F32);
        assert_eq!(lib.spv_blob(f).unwrap(), spv.as_slice());
        assert_eq!(f.spv.offset, spv_off);
        assert_eq!(f.gate, GateJson::default(), "no gate field reads as unverified");
        assert_eq!(f.gate.state, GateState::Unverified);
    }

    #[test]
    fn gate_field_round_trips() {
        let spv = fake_spv(128);
        let with_gate = |gate: &str| {
            let j = sample_json(0, spv.len() as u64).replace(
                r#""uses_coopmat":false"#,
                &format!(r#""uses_coopmat":false,"gate":{gate}"#),
            );
            let mut off = 0u64;
            for _ in 0..4 {
                let j = j.replace("\"offset\":0,", &format!("\"offset\":{off},"));
                let next = (12 + j.len()).next_multiple_of(4) as u64;
                if next == off {
                    break;
                }
                off = next;
            }
            container(&j.replace("\"offset\":0,", &format!("\"offset\":{off},")), &spv)
        };
        let lib = Library::from_bytes_inner(&with_gate(r#"{"state":"verified","evidence":"gpu_add on G13C"}"#))
            .expect("verified gate parses");
        let g = &lib.function("add").unwrap().gate;
        assert_eq!(g.state, GateState::Verified);
        assert_eq!(g.evidence, "gpu_add on G13C");
        let lib = Library::from_bytes_inner(&with_gate(r#"{"evidence":"x"}"#)).expect("state defaults");
        assert_eq!(lib.function("add").unwrap().gate.state, GateState::Unverified);
        let err = Library::from_bytes_inner(&with_gate(r#"{"state":"maybe"}"#)).err().expect("bad state refused").to_string();
        assert!(err.contains("maybe") || err.contains("variant"), "unknown state must be refused: {err}");
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

    /// One valid function: buffer arguments 0 and 1 at set 0 bindings 0 and 1, a 20-byte
    /// push block with one POD word (argument 2, offset 16, size 4).
    fn fixture() -> FunctionJson {
        FunctionJson {
            name: "fixture".into(),
            entry: "fixture".into(),
            spv: SpvRange { offset: 0, len: 0 },
            workgroup_size: [32, 1, 1],
            workgroup_size_spec_ids: None,
            buffers: vec![
                BufferBindingJson { arg: 0, binding: 0, set: 0 },
                BufferBindingJson { arg: 1, binding: 1, set: 0 },
            ],
            push: PushJson { size: 20, words: vec![PushWordJson { arg: 2, offset: 16, size: 4 }] },
            constants: vec![],
            threadgroup: vec![],
            uses_coopmat: false,
            gate: GateJson::default(),
        }
    }

    /// Container for `f` with the SPIR-V offset recomputed for the final JSON, so the
    /// byte envelope (aligned, in-range blob) stays valid and only the metadata varies.
    fn fixture_container(mut f: FunctionJson) -> Vec<u8> {
        let spv = fake_spv(128);
        f.spv = SpvRange { offset: 0, len: spv.len() as u64 };
        for _ in 0..8 {
            let json = serde_json::to_string(&M2vLibJson { functions: vec![f.clone()] }).unwrap();
            let off = (12 + json.len()).next_multiple_of(4) as u64;
            if off == f.spv.offset {
                return container(&json, &spv);
            }
            f.spv.offset = off;
        }
        panic!("SPIR-V offset did not converge");
    }

    /// Both parse paths must refuse `f` with `Error::Invalid`; returns the diagnostic.
    fn invalid(f: FunctionJson) -> String {
        let bytes = fixture_container(f);
        let msg = match parse_container(&bytes) {
            Err(Error::Invalid(msg)) => msg,
            Err(e) => panic!("expected Error::Invalid, got {e}"),
            Ok(_) => panic!("inconsistent metadata accepted"),
        };
        match Library::from_bytes_inner(&bytes) {
            Err(Error::Invalid(m)) => assert_eq!(m, msg),
            Err(e) => panic!("expected Error::Invalid, got {e}"),
            Ok(_) => panic!("library returned for inconsistent metadata"),
        }
        msg
    }

    fn push_layout(f: &FunctionJson) -> (u32, Vec<(u32, u32, u32)>) {
        (f.push.size, f.push.words.iter().map(|w| (w.arg, w.offset, w.size)).collect())
    }

    fn buffer_layout(f: &FunctionJson) -> Vec<(u32, u32, u32)> {
        f.buffers.iter().map(|b| (b.arg, b.set, b.binding)).collect()
    }

    #[test]
    fn push_word_exactly_at_block_end_is_valid() {
        let lib = Library::from_bytes_inner(&fixture_container(fixture())).expect("word ending at byte 20 is valid");
        let f = lib.function("fixture").unwrap();
        // The word ends exactly at the declared block end; no trailing padding is required.
        assert_eq!(push_layout(f), (20, vec![(2, 16, 4)]));
        assert_eq!(lib.spv_blob(f).unwrap(), fake_spv(128).as_slice());
    }

    #[test]
    fn push_word_outside_block_is_invalid() {
        // (16, 16, 4) is the old round-trip fixture's shape.
        for (block, offset, size) in [(16, 16, 4), (20, 20, 4), (20, 24, 4)] {
            let mut f = fixture();
            f.push = PushJson { size: block, words: vec![PushWordJson { arg: 2, offset, size }] };
            let err = invalid(f);
            for id in [
                "function \"fixture\"".to_string(),
                "push word for argument 2".to_string(),
                format!("offset {offset}, size {size}"),
                format!("{block}-byte push block"),
            ] {
                assert!(err.contains(&id), "({block}, {offset}, {size}): {id:?} missing from {err}");
            }
        }
    }

    #[test]
    fn push_word_extent_overflow_is_invalid() {
        let mut f = fixture();
        f.push.words[0].offset = 4294967292;
        f.push.words[0].size = 8;
        let err = invalid(f);
        for id in ["function \"fixture\"", "push word for argument 2", "offset 4294967292, size 8", "overflows"] {
            assert!(err.contains(id), "{id:?} missing from {err}");
        }
    }

    #[test]
    fn descriptor_set_outside_host_contract_is_invalid() {
        let mut f = fixture();
        f.buffers[1].set = 1;
        let err = invalid(f);
        for id in ["function \"fixture\"", "buffer argument 1", "descriptor set 1"] {
            assert!(err.contains(id), "{id:?} missing from {err}");
        }
    }

    #[test]
    fn conflicting_descriptor_bindings_are_invalid() {
        let lib = Library::from_bytes_inner(&fixture_container(fixture())).expect("bindings 0 and 1 are valid");
        assert_eq!(buffer_layout(lib.function("fixture").unwrap()), vec![(0, 0, 0), (1, 0, 1)]);

        let mut f = fixture();
        f.buffers[1].binding = 0;
        let err = invalid(f);
        for id in ["function \"fixture\"", "buffer arguments 0 and 1", "set 0 binding 0"] {
            assert!(err.contains(id), "{id:?} missing from {err}");
        }
    }

    #[test]
    fn duplicate_argument_slots_are_invalid() {
        // Distinct bindings must not hide a duplicate buffer slot.
        let mut f = fixture();
        f.buffers[1].arg = 0;
        let err = invalid(f);
        for id in ["function \"fixture\"", "buffer argument 0", "declared twice"] {
            assert!(err.contains(id), "{id:?} missing from {err}");
        }

        // Distinct in-block extents must not hide a duplicate push-constant slot.
        let mut f = fixture();
        f.push = PushJson {
            size: 24,
            words: vec![PushWordJson { arg: 2, offset: 16, size: 4 }, PushWordJson { arg: 2, offset: 20, size: 4 }],
        };
        let err = invalid(f);
        for id in ["function \"fixture\"", "push-constant argument 2", "declared twice"] {
            assert!(err.contains(id), "{id:?} missing from {err}");
        }
    }
}
