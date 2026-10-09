//! Kernel routing policy: never run a wrong kernel.
//!
//! For every kernel name the host layer asks [`Policy::route`] before it dispatches. The answer is one of:
//!
//! * [`Route::Translated`]: the translated kernel is verified (it matched its reference in the sweep table or on a
//!   previous first use in this process). Run it.
//! * [`Route::TranslatedChecked`]: the translated kernel exists but has not been verified. Run it into shadow outputs,
//!   compute the reference into the real outputs, compare, and call [`Policy::report_check`]. The caller uses the
//!   reference result for this first dispatch; a mismatch therefore never reaches the model.
//! * [`Route::Fallback`]: the kernel could not be translated, or failed its check. Run the named hand kernel (an
//!   omarchy-mlx kernel that matches) or the CPU reference instead.
//! * `Err(Error::Refused)`: no translation that can be trusted and no fallback. The error names the kernel.
//!
//! Every routing event that is not the plain verified path is logged once per kernel on stderr (`m2v:` prefix) and kept
//! in [`Policy::events`] for tests and receipts. There is no switch that turns the gate off.
//!
//! The gate table is JSON (`M2V_POLICY_FILE`, or [`Policy::from_json`]):
//!
//! ```json
//! { "kernels":   { "softmax_f32": {"state": "verified", "evidence": "sweep run1 row 41"},
//!                   "gemm_f32_t64x64x32": {"state": "unverified"} },
//!   "fallbacks": { "gemm_f32_t64x64x32": {"hand": "matmul_coopmat_direct"},
//!                  "rope_f32": {"cpu": "rope_reference_f32"} } }
//! ```
//!
//! `tools/make-gate.py` builds the `kernels` section from a sweep result: a kernel is `verified` only when its sweep row
//! has `ref` ok (it ran on the device and matched the CPU reference).

use std::collections::{BTreeMap, HashMap, HashSet};
use parking_lot::Mutex;

use serde::{Deserialize, Serialize};

use crate::{Error, GateState, Result};

/// The substitute for a kernel that cannot be used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Fallback {
    /// An omarchy-mlx hand kernel with the same contract, by kernel name.
    Hand(String),
    /// A CPU reference implementation registered by the host application, by name.
    Cpu(String),
}

/// What the caller must do for this dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    Translated,
    /// Run translated into shadow outputs and the reference into the real ones, then call `report_check`.
    TranslatedChecked { reference: Fallback },
    Fallback(Fallback),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateEntry {
    pub state: Option<GateState>,
    #[serde(default)]
    pub evidence: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PolicyJson {
    #[serde(default)]
    pub kernels: BTreeMap<String, GateEntry>,
    #[serde(default)]
    pub fallbacks: BTreeMap<String, Fallback>,
}

/// Routing policy shared by every dispatch of one device.
pub struct Policy {
    state: Mutex<Inner>,
}

struct Inner {
    gate: HashMap<String, (GateState, String)>,
    fallbacks: HashMap<String, Fallback>,
    logged: HashSet<(String, &'static str)>,
    events: Vec<String>,
    quiet: bool,
}

impl Policy {
    /// Empty policy: nothing is verified and nothing has a fallback, so every kernel is refused until it is added.
    pub fn new() -> Policy {
        Policy {
            state: Mutex::new(Inner {
                gate: HashMap::new(),
                fallbacks: HashMap::new(),
                logged: HashSet::new(),
                events: Vec::new(),
                quiet: false,
            }),
        }
    }

    pub fn from_json(text: &str) -> Result<Policy> {
        let parsed: PolicyJson =
            serde_json::from_str(text).map_err(|e| Error::Invalid(format!("policy json: {e}")))?;
        let p = Policy::new();
        {
            let mut s = p.state.lock();
            for (name, e) in parsed.kernels {
                s.gate.insert(name, (e.state.unwrap_or(GateState::Unverified), e.evidence));
            }
            s.fallbacks = parsed.fallbacks.into_iter().collect();
        }
        Ok(p)
    }

    /// Reads `M2V_POLICY_FILE` when set (an unreadable or malformed file is an error, never ignored).
    pub fn from_env() -> Result<Policy> {
        match std::env::var("M2V_POLICY_FILE") {
            Ok(path) => {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| Error::Invalid(format!("M2V_POLICY_FILE {path}: {e}")))?;
                Policy::from_json(&text)
            }
            Err(_) => Ok(Policy::new()),
        }
    }

    /// Silence stderr (events are still recorded). For tests.
    pub fn set_quiet(&self, quiet: bool) {
        self.state.lock().quiet = quiet;
    }

    pub fn set_fallback(&self, kernel: &str, fb: Fallback) {
        self.state.lock().fallbacks.insert(kernel.to_string(), fb);
    }

    pub fn set_state(&self, kernel: &str, state: GateState, evidence: &str) {
        self.state.lock().gate.insert(kernel.to_string(), (state, evidence.to_string()));
    }

    /// True when the table already has an entry for `kernel` (from the policy file or a first-use check).
    pub fn knows(&self, kernel: &str) -> bool {
        self.state.lock().gate.contains_key(kernel)
    }

    pub fn state_of(&self, kernel: &str) -> GateState {
        self.state.lock().gate.get(kernel).map(|g| g.0).unwrap_or(GateState::Unverified)
    }

    /// Everything that was logged, in order (each message once).
    pub fn events(&self) -> Vec<String> {
        self.state.lock().events.clone()
    }

    /// Decide how to run `kernel`. `translated` is whether the loaded library has a translated function of that name
    /// (false: not translated, or the translated pipeline failed to build).
    pub fn route(&self, kernel: &str, translated: bool) -> Result<Route> {
        let mut s = self.state.lock();
        let state = s.gate.get(kernel).map(|g| g.0).unwrap_or(GateState::Unverified);
        let fallback = s.fallbacks.get(kernel).cloned();
        match (translated, state) {
            (true, GateState::Verified) => Ok(Route::Translated),
            (true, GateState::Unverified) => match fallback {
                Some(reference) => {
                    s.log(kernel, "unverified", format!("kernel '{kernel}': translated but not verified, checking it against {reference:?} on first use"));
                    Ok(Route::TranslatedChecked { reference })
                }
                None => {
                    let msg = format!("kernel '{kernel}': translated but not verified and there is no reference to check it against; refusing");
                    s.log(kernel, "refused", msg.clone());
                    Err(Error::Refused(msg))
                }
            },
            (true, GateState::Failed) | (false, _) => {
                let why = if translated { "failed its correctness check" } else { "not translated" };
                match fallback {
                    Some(fb) => {
                        s.log(kernel, "fallback", format!("kernel '{kernel}': {why}, using fallback {fb:?}"));
                        Ok(Route::Fallback(fb))
                    }
                    None => {
                        let msg = format!("kernel '{kernel}': {why} and no fallback is registered; refusing");
                        s.log(kernel, "refused", msg.clone());
                        Err(Error::Refused(msg))
                    }
                }
            }
        }
    }

    /// Result of the first-use check requested by [`Route::TranslatedChecked`]. A match enables the translated kernel
    /// for the rest of the process; a mismatch disables it and routes to the fallback from now on.
    pub fn report_check(&self, kernel: &str, matched: bool, detail: &str) {
        let mut s = self.state.lock();
        if matched {
            s.gate.insert(kernel.to_string(), (GateState::Verified, format!("first use: {detail}")));
            s.log(kernel, "verified", format!("kernel '{kernel}': matched its reference on first use ({detail}), enabled"));
        } else {
            s.gate.insert(kernel.to_string(), (GateState::Failed, format!("first use: {detail}")));
            s.log(kernel, "mismatch", format!("kernel '{kernel}': MISMATCH against its reference ({detail}), disabled; routing to the fallback"));
        }
    }
}

impl Default for Policy {
    fn default() -> Self {
        Policy::new()
    }
}

impl Inner {
    fn log(&mut self, kernel: &str, kind: &'static str, msg: String) {
        if self.logged.insert((kernel.to_string(), kind)) {
            if !self.quiet {
                eprintln!("m2v: {msg}");
            }
            self.events.push(msg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(json: &str) -> Policy {
        let p = Policy::from_json(json).unwrap();
        p.set_quiet(true);
        p
    }

    const T: &str = r#"{
      "kernels": {"ok_k": {"state": "verified", "evidence": "sweep"}, "new_k": {"state": "unverified"}, "bad_k": {"state": "failed"}},
      "fallbacks": {"new_k": {"hand": "hand_new"}, "bad_k": {"cpu": "ref_bad"}, "gone_k": {"cpu": "ref_gone"}}
    }"#;

    #[test]
    fn verified_runs_translated_without_log() {
        let p = policy(T);
        assert_eq!(p.route("ok_k", true).unwrap(), Route::Translated);
        assert!(p.events().is_empty());
    }

    #[test]
    fn unverified_is_checked_then_enabled_on_match() {
        let p = policy(T);
        assert_eq!(p.route("new_k", true).unwrap(), Route::TranslatedChecked { reference: Fallback::Hand("hand_new".into()) });
        p.report_check("new_k", true, "max rel err 1e-7");
        assert_eq!(p.route("new_k", true).unwrap(), Route::Translated);
        assert_eq!(p.state_of("new_k"), GateState::Verified);
    }

    #[test]
    fn mismatch_disables_the_translated_kernel_for_good() {
        let p = policy(T);
        p.route("new_k", true).unwrap();
        p.report_check("new_k", false, "max rel err 0.3");
        for _ in 0..3 {
            assert_eq!(p.route("new_k", true).unwrap(), Route::Fallback(Fallback::Hand("hand_new".into())));
        }
        assert_eq!(p.state_of("new_k"), GateState::Failed);
    }

    #[test]
    fn failed_kernel_never_runs_translated() {
        let p = policy(T);
        assert_eq!(p.route("bad_k", true).unwrap(), Route::Fallback(Fallback::Cpu("ref_bad".into())));
    }

    #[test]
    fn untranslated_uses_fallback_else_refuses_by_name() {
        let p = policy(T);
        assert_eq!(p.route("gone_k", false).unwrap(), Route::Fallback(Fallback::Cpu("ref_gone".into())));
        let e = p.route("nothing_k", false).unwrap_err().to_string();
        assert!(e.contains("nothing_k"), "{e}");
    }

    #[test]
    fn unverified_without_reference_is_refused() {
        let p = policy(T);
        let e = p.route("unknown_k", true).unwrap_err().to_string();
        assert!(e.contains("unknown_k") && e.contains("refusing"), "{e}");
    }

    #[test]
    fn each_event_is_logged_once_per_kernel() {
        let p = policy(T);
        for _ in 0..5 {
            let _ = p.route("gone_k", false);
            let _ = p.route("nothing_k", false);
        }
        assert_eq!(p.events().len(), 2);
    }

    #[test]
    fn malformed_policy_is_an_error() {
        assert!(Policy::from_json("{not json").is_err());
    }
}
