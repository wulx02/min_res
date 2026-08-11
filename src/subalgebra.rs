use std::{
    cmp::Ordering,
    env, fmt, fs,
    path::Path,
    sync::{
        OnceLock,
        atomic::{AtomicU8, Ordering as AtomicOrdering},
    },
};

use crate::fast_hash::FastHashMap as HashMap;
use crate::milnor::{
    CoeffKey, Milnor, PACKED_ENTRY_LIMIT, basis_of_degree, pack_padded_entries,
    pack_padded_entries_unchecked, packed_entry, tau_a, weight,
};
use serde::Deserialize;

const A2_DETAILED_SHAPE: usize = 513;
const A2_DETAILED_MAX_DEGREE: usize = A2_DETAILED_SHAPE - 1;
const A2_DETAILED_D_MAX: usize = 23;
const A2_DETAILED_DEFAULT_PATH: &str = "detailed_subalgebra/Ext_A2.bin";
const A2_DETAILED_EMBEDDED_BITSET: &[u8] = include_bytes!("../detailed_subalgebra/Ext_A2.bin");
const A2_DETAILED_EMBEDDED_METADATA: &[u8] =
    include_bytes!("../detailed_subalgebra/Ext_A2_support_0_512.json");
const B3211_DETAILED_EMBEDDED_BITSET: &[u8] =
    include_bytes!("../detailed_subalgebra/B3211_support_s0-512_t0-512.bin");
const B3211_DETAILED_EMBEDDED_METADATA: &[u8] =
    include_bytes!("../detailed_subalgebra/B3211_support_s0-512_t0-512.json");
const B3221_DETAILED_EMBEDDED_BITSET: &[u8] =
    include_bytes!("../detailed_subalgebra/B3221_support_s0-512_t0-512.bin");
const B3221_DETAILED_EMBEDDED_METADATA: &[u8] =
    include_bytes!("../detailed_subalgebra/B3221_support_s0-512_t0-512.json");
const SUBALGEBRA_SELECTION_ORIGINAL: u8 = 0;
const SUBALGEBRA_SELECTION_DETAILED: u8 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum A2ConditionMode {
    Theorem,
    Paper23,
    Detailed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum A3ConditionMode {
    Theorem,
    Paper72,
}

static A2_CONDITION_MODE: OnceLock<A2ConditionMode> = OnceLock::new();
static A3_CONDITION_MODE: OnceLock<A3ConditionMode> = OnceLock::new();
static A2_DETAILED_SUPPORT: OnceLock<Vec<u8>> = OnceLock::new();
static DETAILED_EXT_TABLES: OnceLock<Vec<DetailedExtTableLoad>> = OnceLock::new();
static SUBALGEBRA_SELECTION_MODE: AtomicU8 = AtomicU8::new(SUBALGEBRA_SELECTION_DETAILED);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubalgebraSelectionMode {
    Original,
    Detailed,
}

impl SubalgebraSelectionMode {
    pub fn parse(input: &str) -> Result<Self, String> {
        match input.trim().to_ascii_lowercase().as_str() {
            "detailed" | "current" | "default" | "detailed-ext" | "detailed_ext" | "ext-table"
            | "ext_table" => Ok(Self::Detailed),
            "original" | "orig" | "old" | "built-in" | "builtin" => Ok(Self::Original),
            other => Err(format!(
                "invalid subalgebra selection mode `{other}`; use original or detailed"
            )),
        }
    }
}

pub fn set_subalgebra_selection_mode(mode: SubalgebraSelectionMode) {
    let value = match mode {
        SubalgebraSelectionMode::Original => SUBALGEBRA_SELECTION_ORIGINAL,
        SubalgebraSelectionMode::Detailed => SUBALGEBRA_SELECTION_DETAILED,
    };
    SUBALGEBRA_SELECTION_MODE.store(value, AtomicOrdering::Relaxed);
}

pub fn subalgebra_selection_mode() -> SubalgebraSelectionMode {
    match SUBALGEBRA_SELECTION_MODE.load(AtomicOrdering::Relaxed) {
        SUBALGEBRA_SELECTION_DETAILED => SubalgebraSelectionMode::Detailed,
        _ => SubalgebraSelectionMode::Original,
    }
}

#[derive(Clone, Copy, Debug)]
struct DetailedFiniteSubalgebraSpec {
    name: &'static str,
    profile: &'static [u32],
    dim: usize,
    tau: usize,
    bitset_env: &'static str,
    metadata_env: &'static str,
    default_bitset_path: &'static str,
    default_metadata_path: &'static str,
    embedded_bitset: Option<&'static [u8]>,
    embedded_metadata: Option<&'static [u8]>,
}

const DETAILED_FINITE_SUBALGEBRAS: &[DetailedFiniteSubalgebraSpec] = &[
    DetailedFiniteSubalgebraSpec {
        name: "A2",
        profile: &[3, 2, 1],
        dim: 64,
        tau: 23,
        bitset_env: "EXT_A2_EXT_BITSET",
        metadata_env: "EXT_A2_EXT_METADATA",
        default_bitset_path: A2_DETAILED_DEFAULT_PATH,
        default_metadata_path: "detailed_subalgebra/Ext_A2_support_0_512.json",
        embedded_bitset: Some(A2_DETAILED_EMBEDDED_BITSET),
        embedded_metadata: Some(A2_DETAILED_EMBEDDED_METADATA),
    },
    DetailedFiniteSubalgebraSpec {
        name: "B3211",
        profile: &[3, 2, 1, 1],
        dim: 128,
        tau: 38,
        bitset_env: "EXT_B3211_EXT_BITSET",
        metadata_env: "EXT_B3211_EXT_METADATA",
        default_bitset_path: "detailed_subalgebra/B3211_support_s0-512_t0-512.bin",
        default_metadata_path: "detailed_subalgebra/B3211_support_s0-512_t0-512.json",
        embedded_bitset: Some(B3211_DETAILED_EMBEDDED_BITSET),
        embedded_metadata: Some(B3211_DETAILED_EMBEDDED_METADATA),
    },
    DetailedFiniteSubalgebraSpec {
        name: "B3221",
        profile: &[3, 2, 2, 1],
        dim: 256,
        tau: 52,
        bitset_env: "EXT_B3221_EXT_BITSET",
        metadata_env: "EXT_B3221_EXT_METADATA",
        default_bitset_path: "detailed_subalgebra/B3221_support_s0-512_t0-512.bin",
        default_metadata_path: "detailed_subalgebra/B3221_support_s0-512_t0-512.json",
        embedded_bitset: Some(B3221_DETAILED_EMBEDDED_BITSET),
        embedded_metadata: Some(B3221_DETAILED_EMBEDDED_METADATA),
    },
    DetailedFiniteSubalgebraSpec {
        name: "B3321",
        profile: &[3, 3, 2, 1],
        dim: 512,
        tau: 64,
        bitset_env: "EXT_B3321_EXT_BITSET",
        metadata_env: "EXT_B3321_EXT_METADATA",
        default_bitset_path: "detailed_subalgebra/B3321_support_s0-512_t0-512.bin",
        default_metadata_path: "detailed_subalgebra/B3321_support_s0-512_t0-512.json",
        embedded_bitset: None,
        embedded_metadata: None,
    },
    DetailedFiniteSubalgebraSpec {
        name: "A3",
        profile: &[4, 3, 2, 1],
        dim: 1024,
        tau: 72,
        bitset_env: "EXT_A3_EXT_BITSET",
        metadata_env: "EXT_A3_EXT_METADATA",
        default_bitset_path: "detailed_subalgebra/A3_support_s0-512_t0-512.bin",
        default_metadata_path: "detailed_subalgebra/A3_support_s0-512_t0-512.json",
        embedded_bitset: None,
        embedded_metadata: None,
    },
];

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct SubalgebraApplicability {
    pub algebra_name: String,
    pub usable: bool,
    pub certification_source: &'static str,
    pub candidate_status: String,
    pub window_lo: usize,
    pub window_hi: Option<usize>,
    pub blocking_reason: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DetailedExtTableCoverage {
    pub algebra_name: String,
    pub profile: Vec<u32>,
    pub dim: usize,
    pub tau: usize,
    pub bitset_path: String,
    pub metadata_path: String,
    pub state: String,
    pub s_min: Option<usize>,
    pub s_max: Option<usize>,
    pub u_min: Option<usize>,
    pub u_max: Option<usize>,
    pub nonzero_entries: Option<usize>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExtQuery {
    KnownZero,
    KnownNonzero(u32),
    Unknown,
}

#[derive(Clone, Debug)]
struct DetailedExtTable {
    algebra_name: &'static str,
    profile: &'static [u32],
    bitset_path: String,
    metadata_path: String,
    s_min: usize,
    s_max: usize,
    u_min: usize,
    u_max: usize,
    u_count: usize,
    nonzero_entries: usize,
    data: Vec<u8>,
}

#[derive(Clone, Debug)]
enum DetailedExtTableLoad {
    Loaded(DetailedExtTable),
    Missing {
        spec: DetailedFiniteSubalgebraSpec,
        bitset_path: String,
        metadata_path: String,
    },
    Invalid {
        spec: DetailedFiniteSubalgebraSpec,
        bitset_path: String,
        metadata_path: String,
        error: String,
    },
}

#[derive(Deserialize)]
struct DetailedExtMetadata {
    bounds: DetailedExtBounds,
    format: String,
    shape: [usize; 2],
    #[serde(default)]
    byte_count: Option<usize>,
}

#[derive(Deserialize)]
struct DetailedExtBounds {
    s_min: usize,
    s_max: usize,
    #[serde(alias = "t_min")]
    u_min: usize,
    #[serde(alias = "t_max")]
    u_max: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Family {
    A,
    B,
    F,
    FPrime,
}

#[derive(Clone, Debug)]
pub struct Subalgebra {
    family: Family,
    n: usize,
    label: Option<String>,
    profile: Vec<u32>,
    signatures: Vec<Milnor>,
    signature_packed_index: HashMap<CoeffKey, usize>,
    bit_order: Vec<(usize, u32)>,
}

impl Subalgebra {
    pub fn a(n: usize) -> Result<Self, String> {
        if n > 5 {
            return Err("A(n) is limited to n <= 5 in this implementation".into());
        }
        let profile = (1..=n + 1).map(|j| (n + 2 - j) as u32).collect::<Vec<_>>();
        let bit_order = compatible_bit_order(&profile);
        Self::from_profile(Family::A, n, None, profile, bit_order, None)
    }

    #[allow(dead_code)]
    pub fn b3211() -> Result<Self, String> {
        Self::b_profile("B3211", vec![3, 2, 1, 1])
    }

    #[allow(dead_code)]
    pub fn b3221() -> Result<Self, String> {
        Self::b_profile("B3221", vec![3, 2, 2, 1])
    }

    #[allow(dead_code)]
    pub fn b3321() -> Result<Self, String> {
        Self::b_profile("B3321", vec![3, 3, 2, 1])
    }

    pub fn b_profile(label: impl Into<String>, profile: Vec<u32>) -> Result<Self, String> {
        validate_finite_profile(&profile)?;
        validate_registered_b_profile(&profile)?;
        let label = label.into();
        let bit_order = compatible_bit_order(&profile);
        Self::from_profile(
            Family::B,
            profile_fingerprint(&profile),
            Some(label),
            profile,
            bit_order,
            None,
        )
    }

    pub fn f(n: usize, max_degree: usize) -> Result<Self, String> {
        if n == 0 {
            return Err("F(n) expects n >= 1".into());
        }
        if n > 3 {
            return Err("F(n) is implemented for n <= 3 in this patch".into());
        }
        let max_index = max_milnor_index(max_degree);
        let profile = (1..=max_index)
            .map(|j| {
                if j <= n {
                    0
                } else {
                    bits_needed((max_degree / weight(j)) as u32)
                }
            })
            .collect::<Vec<_>>();
        let bit_order = compatible_bit_order(&profile);
        Self::from_profile(Family::F, n, None, profile, bit_order, Some(max_degree))
    }

    pub fn fprime(n: usize, max_degree: usize) -> Result<Self, String> {
        if n == 0 {
            return Err("F'(n) expects n >= 1".into());
        }
        if n > 3 {
            return Err("F'(n) is implemented for n <= 3 in this patch".into());
        }
        let max_index = max_milnor_index(max_degree);
        let profile = vec![0; max_index];
        let bit_order = fprime_bit_order(n, max_degree);
        let mut entries = vec![0; max_index];
        let mut signatures = Vec::new();
        generate_fprime_signatures(
            1,
            n,
            max_index,
            max_degree,
            0,
            &mut entries,
            &mut signatures,
        );
        sort_signatures(Family::FPrime, &bit_order, &mut signatures);
        Self::from_signatures(Family::FPrime, n, None, profile, bit_order, signatures)
    }

    fn from_profile(
        family: Family,
        n: usize,
        label: Option<String>,
        profile: Vec<u32>,
        bit_order: Vec<(usize, u32)>,
        max_signature_degree: Option<usize>,
    ) -> Result<Self, String> {
        let mut entries = vec![0; profile.len()];
        let mut signatures = Vec::new();
        generate_signatures(
            0,
            &profile,
            max_signature_degree.unwrap_or(usize::MAX),
            0,
            &mut entries,
            &mut signatures,
        );
        sort_signatures(family, &bit_order, &mut signatures);
        Self::from_signatures(family, n, label, profile, bit_order, signatures)
    }

    fn from_signatures(
        family: Family,
        n: usize,
        label: Option<String>,
        profile: Vec<u32>,
        bit_order: Vec<(usize, u32)>,
        signatures: Vec<Milnor>,
    ) -> Result<Self, String> {
        let signature_packed_index = signatures
            .iter()
            .enumerate()
            .map(|(i, sig)| {
                (
                    sig.packed()
                        .unwrap_or_else(|| panic!("signature {sig} cannot be packed")),
                    i,
                )
            })
            .collect();

        Ok(Self {
            family,
            n,
            label,
            profile,
            signatures,
            signature_packed_index,
            bit_order,
        })
    }

    pub fn parse(input: &str, max_degree: usize) -> Result<Self, String> {
        let trimmed = input.trim();
        if let Some(n_text) = parse_family_index(trimmed, 'A') {
            let n = n_text
                .parse::<usize>()
                .map_err(|_| format!("invalid subalgebra `{input}`; use A0, A1, A2, ..."))?;
            return Self::a(n);
        }
        if let Some((label, profile)) = parse_b_profile(trimmed)? {
            return Self::b_profile(label, profile);
        }
        if let Some(n_text) = parse_fprime_index(trimmed) {
            let n = n_text
                .parse::<usize>()
                .map_err(|_| format!("invalid subalgebra `{input}`; use Fp1, Fp2, Fprime2, ..."))?;
            return Self::fprime(n, max_degree);
        }
        if let Some(n_text) = parse_family_index(trimmed, 'F') {
            let n = n_text
                .parse::<usize>()
                .map_err(|_| format!("invalid subalgebra `{input}`; use F1, F2, ..."))?;
            return Self::f(n, max_degree);
        }
        Err(
            "subalgebra must be A0/A1/A2/..., B3211/B(3,2,1,1), F1/F2/..., or Fp1/Fp2/..."
                .to_string(),
        )
    }

    pub fn name(&self) -> String {
        let prefix = match self.family {
            Family::A => "A",
            Family::B => {
                return self
                    .label
                    .clone()
                    .unwrap_or_else(|| profile_label(&self.profile));
            }
            Family::F => "F",
            Family::FPrime => "Fp",
        };
        format!("{prefix}{}", self.n)
    }

    pub fn cache_key(&self) -> (u8, usize, usize, usize, usize) {
        let family = match self.family {
            Family::A => 0,
            Family::F => 1,
            Family::FPrime => 2,
            Family::B => 3,
        };
        let id = if matches!(self.family, Family::B) {
            profile_fingerprint(&self.profile)
        } else {
            self.n
        };
        (
            family,
            id,
            self.profile.len(),
            self.signatures.len(),
            self.bit_order.len(),
        )
    }

    #[allow(dead_code)]
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn signatures(&self) -> &[Milnor] {
        &self.signatures
    }

    pub fn profile_cache_key(&self) -> Option<(u8, usize)> {
        match self.family {
            Family::A => Some((0, self.n)),
            Family::B => Some((3, profile_fingerprint(&self.profile))),
            Family::F => Some((1, profile_fingerprint(&self.profile))),
            Family::FPrime => None,
        }
    }

    pub fn profile(&self) -> &[u32] {
        &self.profile
    }

    #[allow(dead_code)]
    pub fn profile_dim(&self) -> usize {
        let exponent = self
            .profile
            .iter()
            .map(|&entry| entry as usize)
            .sum::<usize>();
        1usize
            .checked_shl(exponent as u32)
            .expect("profile dimension overflow")
    }

    pub fn profile_tau(&self) -> usize {
        self.profile
            .iter()
            .enumerate()
            .filter(|&(_, &entry)| entry > 0)
            .map(|(index, &entry)| {
                let two_power = 1usize
                    .checked_shl(entry)
                    .expect("profile tau exponent overflow");
                (two_power - 1) * weight(index + 1)
            })
            .sum()
    }

    pub fn profile_d(&self) -> usize {
        self.profile
            .iter()
            .enumerate()
            .filter(|&(_, &entry)| entry > 0)
            .map(|(index, &entry)| {
                let two_power = 1usize
                    .checked_shl(entry - 1)
                    .expect("profile d exponent overflow");
                weight(index + 1) * two_power
            })
            .max()
            .unwrap_or(0)
    }

    pub fn profile_lower_ok(&self, s: usize, t: usize) -> bool {
        let tau = self.profile_tau();
        let d = self.profile_d();
        t > d.saturating_mul(s).saturating_add(tau)
    }

    pub fn selection_priority(&self) -> usize {
        match self.family {
            Family::A => match self.n {
                3 => 0,
                2 => 4,
                1 => 5,
                0 => 6,
                n => 100 + n,
            },
            Family::B => match self.name().as_str() {
                "B3321" => 1,
                "B3221" => 2,
                "B3211" => 3,
                _ => 50,
            },
            Family::FPrime => 1000 + self.n.saturating_sub(1) * 2,
            Family::F => 1001 + self.n.saturating_sub(1) * 2,
        }
    }

    pub(crate) fn f_family_index(&self) -> Option<(usize, bool)> {
        match self.family {
            Family::F => Some((self.n, true)),
            Family::FPrime => Some((self.n, false)),
            _ => None,
        }
    }

    pub fn split_profile_signature_packed(&self, x: CoeffKey) -> Option<(CoeffKey, CoeffKey)> {
        self.profile_cache_key()?;
        let mut sig = [0_u32; PACKED_ENTRY_LIMIT];
        let mut quotient = [0_u32; PACKED_ENTRY_LIMIT];
        for i in 0..PACKED_ENTRY_LIMIT {
            let value = packed_entry(x, i);
            let profile_entry = self.profile.get(i).copied().unwrap_or(0);
            let sig_value = if profile_entry >= u32::BITS {
                value
            } else {
                let mask = (1_u32 << profile_entry).saturating_sub(1);
                value & mask
            };
            let quotient_value = value.checked_sub(sig_value)?;
            sig[i] = sig_value;
            quotient[i] = quotient_value;
        }
        Some((
            pack_padded_entries_unchecked(&sig),
            pack_padded_entries_unchecked(&quotient),
        ))
    }

    #[allow(dead_code)]
    pub fn split_signature_packed(&self, x: CoeffKey) -> Option<(CoeffKey, CoeffKey)> {
        let signature = self.signature_packed(x);
        let quotient = self.quotient_part_packed(x)?;
        Some((signature, quotient))
    }

    #[allow(dead_code)]
    pub fn quotient_part_packed(&self, x: CoeffKey) -> Option<CoeffKey> {
        let signature = self.signature_packed(x);
        let mut quotient = [0_u32; PACKED_ENTRY_LIMIT];
        for (i, quotient_entry) in quotient.iter_mut().enumerate() {
            let value = packed_entry(x, i);
            let sig_value = packed_entry(signature, i);
            *quotient_entry = value.checked_sub(sig_value)?;
        }
        pack_padded_entries(&quotient)
    }

    #[allow(dead_code)]
    pub fn compose_signature_with_quotient_packed(
        &self,
        sig: CoeffKey,
        quotient: CoeffKey,
    ) -> Option<CoeffKey> {
        let mut entries = [0_u32; PACKED_ENTRY_LIMIT];
        for (i, entry) in entries.iter_mut().enumerate() {
            *entry = packed_entry(sig, i).checked_add(packed_entry(quotient, i))?;
        }
        let packed = pack_padded_entries(&entries)?;
        (self.signature_packed(packed) == sig && self.signature_is_zero_packed(quotient))
            .then_some(packed)
    }

    #[allow(dead_code)]
    pub fn signature_is_zero_packed(&self, x: CoeffKey) -> bool {
        self.signature_packed(x) == 0
    }

    #[allow(dead_code)]
    pub fn same_signature_packed(&self, x: CoeffKey, sig: CoeffKey) -> bool {
        self.signature_packed(x) == sig
    }

    #[allow(dead_code)]
    pub fn signature_degree_packed(&self, sig: CoeffKey) -> usize {
        Milnor::from_packed(sig).degree()
    }

    #[allow(dead_code)]
    pub fn quotient_basis(&self, degree: usize) -> Vec<CoeffKey> {
        basis_of_degree(degree)
            .into_iter()
            .filter_map(|coeff| coeff.packed())
            .filter(|&packed| self.signature_is_zero_packed(packed))
            .collect()
    }

    #[allow(dead_code)]
    pub fn quotient_count(&self, degree: usize) -> usize {
        self.quotient_basis(degree).len()
    }

    #[allow(dead_code)]
    pub fn profile_quotient_packed(&self, x: CoeffKey) -> Option<CoeffKey> {
        self.profile_cache_key()?;
        Some(self.profile_quotient_packed_unchecked(x))
    }

    pub fn profile_quotient_packed_unchecked(&self, x: CoeffKey) -> CoeffKey {
        let mut quotient = [0_u32; PACKED_ENTRY_LIMIT];
        for (i, quotient_entry) in quotient.iter_mut().enumerate() {
            let value = packed_entry(x, i);
            let profile_entry = self.profile.get(i).copied().unwrap_or(0);
            *quotient_entry = if profile_entry >= u32::BITS {
                0
            } else {
                let mask = (1_u32 << profile_entry).saturating_sub(1);
                value & !mask
            };
        }
        pack_padded_entries_unchecked(&quotient)
    }

    pub fn profile_signature_is_zero_packed_unchecked(&self, x: CoeffKey) -> bool {
        debug_assert!(self.profile_cache_key().is_some());
        for i in 0..PACKED_ENTRY_LIMIT {
            let value = packed_entry(x, i);
            let profile_entry = self.profile.get(i).copied().unwrap_or(0);
            let mask = if profile_entry >= u32::BITS {
                u32::MAX
            } else {
                (1_u32 << profile_entry).saturating_sub(1)
            };
            if value & mask != 0 {
                return false;
            }
        }
        true
    }

    #[allow(dead_code)]
    pub fn attach_profile_signature_packed(&self, sig: CoeffKey, x: CoeffKey) -> Option<CoeffKey> {
        self.profile_cache_key()?;
        self.attach_profile_signature_packed_unchecked(sig, x)
    }

    pub fn attach_profile_signature_packed_unchecked(
        &self,
        sig: CoeffKey,
        x: CoeffKey,
    ) -> Option<CoeffKey> {
        let mut out = [0_u32; PACKED_ENTRY_LIMIT];
        for (i, out_entry) in out.iter_mut().enumerate() {
            let sig_value = packed_entry(sig, i);
            let value = packed_entry(x, i);
            if sig_value & value != 0 {
                return None;
            }
            *out_entry = sig_value + value;
        }
        pack_padded_entries(&out)
    }

    #[allow(dead_code)]
    pub fn signature(&self, x: &Milnor) -> Milnor {
        let entries = match self.family {
            Family::A | Family::B | Family::F => (0..self.profile.len())
                .map(|i| {
                    let modulus = if self.profile[i] >= u32::BITS {
                        u32::MAX
                    } else {
                        1_u32 << self.profile[i]
                    };
                    if modulus == u32::MAX {
                        x.entries().get(i).copied().unwrap_or(0)
                    } else {
                        x.entries().get(i).copied().unwrap_or(0) % modulus
                    }
                })
                .collect::<Vec<_>>(),
            Family::FPrime => (0..self.profile.len())
                .map(|i| {
                    let j = i + 1;
                    let value = x.entries().get(i).copied().unwrap_or(0);
                    if j < self.n {
                        0
                    } else if j == self.n {
                        value & !1
                    } else {
                        value
                    }
                })
                .collect::<Vec<_>>(),
        };
        Milnor::new(entries)
    }

    #[allow(dead_code)]
    pub fn signature_index(&self, x: &Milnor) -> usize {
        let packed = x
            .packed()
            .unwrap_or_else(|| panic!("Milnor element {x} cannot be packed"));
        self.signature_index_packed(packed)
    }

    pub fn signature_index_packed(&self, x: CoeffKey) -> usize {
        if matches!(self.family, Family::A | Family::B) {
            return self.profile_signature_index_packed(x);
        }
        let sig = self.signature_packed(x);
        self.signature_packed_index
            .get(&sig)
            .copied()
            .unwrap_or_else(|| panic!("missing packed signature {sig} for {}", self.name()))
    }

    fn profile_signature_index_packed(&self, x: CoeffKey) -> usize {
        let mut index = 0_usize;
        for &(j, bit) in &self.bit_order {
            let value = packed_entry(x, j - 1);
            index = (index << 1) | (((value >> bit) & 1) as usize);
        }
        index
    }

    fn signature_packed(&self, x: CoeffKey) -> CoeffKey {
        let mut entries = [0_u32; PACKED_ENTRY_LIMIT];
        for (i, &profile_entry) in self.profile.iter().enumerate() {
            let value = packed_entry(x, i);
            let sig_value = match self.family {
                Family::A | Family::B | Family::F => {
                    let modulus = if profile_entry >= u32::BITS {
                        u32::MAX
                    } else {
                        1_u32 << profile_entry
                    };
                    if modulus == u32::MAX {
                        value
                    } else {
                        value % modulus
                    }
                }
                Family::FPrime => {
                    let j = i + 1;
                    if j < self.n {
                        0
                    } else if j == self.n {
                        value & !1
                    } else {
                        value
                    }
                }
            };
            entries[i] = sig_value;
        }
        pack_padded_entries_unchecked(&entries)
    }

    /// Returns whether this subalgebra is justified for the fixed-t homology
    /// task `s` at internal degree `t`. In fixed-t layer code this `s` is the
    /// task index for `H_s(D^(t))`, not the homological degree `s + 1` of
    /// generators produced by the task.
    pub fn lower_line_applies(&self, s: usize, t: usize) -> bool {
        match self.family {
            Family::A => {
                if self.n == 3 {
                    match a3_condition_mode() {
                        A3ConditionMode::Theorem => {}
                        A3ConditionMode::Paper72 => {
                            return t > 15 * s + tau_a(3);
                        }
                    }
                }
                if self.n == 2 {
                    match a2_condition_mode() {
                        A2ConditionMode::Theorem => {}
                        A2ConditionMode::Paper23 => {
                            return t > 7 * s + tau_a(2);
                        }
                        A2ConditionMode::Detailed => {
                            return a2_detailed_condition_applies(s, t);
                        }
                    }
                }
                let rho = (1usize << (self.n + 1)) - 1;
                t > rho * s + tau_a(self.n)
            }
            Family::B => self.profile_lower_ok(s, t),
            Family::F => {
                let slope = (1usize << (self.n + 1)) - 1;
                t < slope * s
            }
            Family::FPrime => {
                let slope = (1usize << (self.n + 1)) - 2;
                t < slope * s
            }
        }
    }

    /// Uses the same task `s` convention as `lower_line_applies`.
    pub fn selection_condition_applies(&self, s: usize, t: usize) -> bool {
        self.applicability_for_mode(s, t, subalgebra_selection_mode())
            .usable
    }

    /// Uses the same task `s` convention as `lower_line_applies`.
    pub fn selection_applicability(&self, s: usize, t: usize) -> SubalgebraApplicability {
        self.applicability_for_mode(s, t, subalgebra_selection_mode())
    }

    pub fn applicability_for_mode(
        &self,
        s: usize,
        t: usize,
        mode: SubalgebraSelectionMode,
    ) -> SubalgebraApplicability {
        match mode {
            SubalgebraSelectionMode::Original => original_applicability(self, s, t),
            SubalgebraSelectionMode::Detailed => detailed_selection_applicability(self, s, t),
        }
    }

    /// Bound for the fixed-t task index `s`, not for output degree `s + 1`.
    pub fn lower_line_bound(&self, s: usize) -> usize {
        match self.family {
            Family::A => {
                let rho = (1usize << (self.n + 1)) - 1;
                rho * s + tau_a(self.n)
            }
            Family::B => self
                .profile_d()
                .saturating_mul(s)
                .saturating_add(self.profile_tau()),
            Family::F => {
                let slope = (1usize << (self.n + 1)) - 1;
                slope * s
            }
            Family::FPrime => {
                let slope = (1usize << (self.n + 1)) - 2;
                slope * s
            }
        }
    }

    #[allow(dead_code)]
    pub fn debug_order(&self) -> String {
        self.bit_order
            .iter()
            .map(|(j, bit)| format!("P{}_{}", j, bit))
            .collect::<Vec<_>>()
            .join(" > ")
    }
}

fn a2_condition_mode() -> A2ConditionMode {
    *A2_CONDITION_MODE.get_or_init(|| {
        match env::var("EXT_A2_CONDITION")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "" | "current" | "default" | "detailed" | "support" | "ext-a2" | "ext_a2" => {
                A2ConditionMode::Detailed
            }
            "theorem" | "old" => A2ConditionMode::Theorem,
            "paper23" | "paper" | "german" | "t>7s+23" => A2ConditionMode::Paper23,
            other => {
                panic!("invalid EXT_A2_CONDITION={other:?}; use theorem, paper23, or detailed")
            }
        }
    })
}

fn a3_condition_mode() -> A3ConditionMode {
    *A3_CONDITION_MODE.get_or_init(|| {
        match env::var("EXT_A3_CONDITION")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "" | "current" | "default" | "paper72" | "paper" | "german" | "t>15s+72" => {
                A3ConditionMode::Paper72
            }
            "theorem" | "old" => A3ConditionMode::Theorem,
            other => panic!("invalid EXT_A3_CONDITION={other:?}; use theorem or paper72"),
        }
    })
}

pub fn detailed_ext_table_coverages() -> Vec<DetailedExtTableCoverage> {
    detailed_ext_tables()
        .iter()
        .map(|load| match load {
            DetailedExtTableLoad::Loaded(table) => DetailedExtTableCoverage {
                algebra_name: table.algebra_name.to_string(),
                profile: table.profile.to_vec(),
                dim: detailed_spec_by_name(table.algebra_name)
                    .map(|spec| spec.dim)
                    .unwrap_or(0),
                tau: detailed_spec_by_name(table.algebra_name)
                    .map(|spec| spec.tau)
                    .unwrap_or(0),
                bitset_path: table.bitset_path.clone(),
                metadata_path: table.metadata_path.clone(),
                state: "found".to_string(),
                s_min: Some(table.s_min),
                s_max: Some(table.s_max),
                u_min: Some(table.u_min),
                u_max: Some(table.u_max),
                nonzero_entries: Some(table.nonzero_entries),
                error: None,
            },
            DetailedExtTableLoad::Missing {
                spec,
                bitset_path,
                metadata_path,
            } => DetailedExtTableCoverage {
                algebra_name: spec.name.to_string(),
                profile: spec.profile.to_vec(),
                dim: spec.dim,
                tau: spec.tau,
                bitset_path: bitset_path.clone(),
                metadata_path: metadata_path.clone(),
                state: "missing".to_string(),
                s_min: None,
                s_max: None,
                u_min: None,
                u_max: None,
                nonzero_entries: None,
                error: None,
            },
            DetailedExtTableLoad::Invalid {
                spec,
                bitset_path,
                metadata_path,
                error,
            } => DetailedExtTableCoverage {
                algebra_name: spec.name.to_string(),
                profile: spec.profile.to_vec(),
                dim: spec.dim,
                tau: spec.tau,
                bitset_path: bitset_path.clone(),
                metadata_path: metadata_path.clone(),
                state: "invalid".to_string(),
                s_min: None,
                s_max: None,
                u_min: None,
                u_max: None,
                nonzero_entries: None,
                error: Some(error.clone()),
            },
        })
        .collect()
}

fn original_applicability(subalgebra: &Subalgebra, s: usize, t: usize) -> SubalgebraApplicability {
    let usable = subalgebra.lower_line_applies(s, t);
    let source = if usable {
        original_certification_source(subalgebra)
    } else {
        "unavailable_data"
    };
    SubalgebraApplicability {
        algebra_name: subalgebra.name(),
        usable,
        certification_source: source,
        candidate_status: if usable {
            format!("usable:{source}")
        } else {
            "blocked:old_rule_failed".to_string()
        },
        window_lo: 0,
        window_hi: None,
        blocking_reason: (!usable).then_some("old_rule_failed"),
    }
}

fn original_certification_source(subalgebra: &Subalgebra) -> &'static str {
    match subalgebra.family {
        Family::A => {
            if subalgebra.n == 2 {
                match a2_condition_mode() {
                    A2ConditionMode::Theorem => "coarse_window_lemma",
                    A2ConditionMode::Paper23 => "coarse_window_lemma",
                    A2ConditionMode::Detailed => "detailed_ext_table",
                }
            } else {
                "coarse_window_lemma"
            }
        }
        Family::B => "coarse_window_lemma",
        Family::F | Family::FPrime => "original_rule",
    }
}

fn detailed_selection_applicability(
    subalgebra: &Subalgebra,
    s: usize,
    t: usize,
) -> SubalgebraApplicability {
    let Some(spec) = detailed_spec_for_subalgebra(subalgebra) else {
        return original_applicability(subalgebra, s, t);
    };
    detailed_applicability_with_original_fallback(subalgebra, spec, s, t)
}

fn detailed_applicability_with_original_fallback(
    subalgebra: &Subalgebra,
    spec: DetailedFiniteSubalgebraSpec,
    s: usize,
    t: usize,
) -> SubalgebraApplicability {
    let table_result = detailed_ext_table_for(spec.name);
    let detailed = match table_result {
        Some(Ok(table)) => detailed_table_applicability(spec, table, s, t),
        Some(Err(_)) | None => return original_applicability(subalgebra, s, t),
    };

    if detailed.usable
        || matches!(
            detailed.blocking_reason,
            Some("nonzero_Ext_row_s" | "nonzero_Ext_row_s_minus_1")
        )
    {
        return detailed;
    }

    if matches!(detailed.blocking_reason, Some("unknown_out_of_range")) {
        return original_applicability(subalgebra, s, t);
    }

    detailed
}

fn detailed_table_applicability(
    spec: DetailedFiniteSubalgebraSpec,
    table: &DetailedExtTable,
    s: usize,
    t: usize,
) -> SubalgebraApplicability {
    let (window_lo, window_hi) = detailed_applicability_window(t, spec.tau);
    let Some(window_hi_value) = window_hi else {
        return SubalgebraApplicability {
            algebra_name: spec.name.to_string(),
            usable: true,
            certification_source: "detailed_ext_table",
            candidate_status: "usable:detailed_ext_table".to_string(),
            window_lo,
            window_hi,
            blocking_reason: None,
        };
    };

    for u in window_lo..=window_hi_value {
        match table.query(s, u) {
            ExtQuery::KnownZero => {}
            ExtQuery::KnownNonzero(_) => {
                return SubalgebraApplicability {
                    algebra_name: spec.name.to_string(),
                    usable: false,
                    certification_source: "detailed_ext_table",
                    candidate_status: "blocked:nonzero_Ext_row_s".to_string(),
                    window_lo,
                    window_hi,
                    blocking_reason: Some("nonzero_Ext_row_s"),
                };
            }
            ExtQuery::Unknown => {
                return SubalgebraApplicability {
                    algebra_name: spec.name.to_string(),
                    usable: false,
                    certification_source: "unavailable_data",
                    candidate_status: "blocked:unknown_out_of_range".to_string(),
                    window_lo,
                    window_hi,
                    blocking_reason: Some("unknown_out_of_range"),
                };
            }
        }

        if s == 0 {
            continue;
        }
        match table.query(s - 1, u) {
            ExtQuery::KnownZero => {}
            ExtQuery::KnownNonzero(_) => {
                return SubalgebraApplicability {
                    algebra_name: spec.name.to_string(),
                    usable: false,
                    certification_source: "detailed_ext_table",
                    candidate_status: "blocked:nonzero_Ext_row_s_minus_1".to_string(),
                    window_lo,
                    window_hi,
                    blocking_reason: Some("nonzero_Ext_row_s_minus_1"),
                };
            }
            ExtQuery::Unknown => {
                return SubalgebraApplicability {
                    algebra_name: spec.name.to_string(),
                    usable: false,
                    certification_source: "unavailable_data",
                    candidate_status: "blocked:unknown_out_of_range".to_string(),
                    window_lo,
                    window_hi,
                    blocking_reason: Some("unknown_out_of_range"),
                };
            }
        }
    }

    SubalgebraApplicability {
        algebra_name: spec.name.to_string(),
        usable: true,
        certification_source: "detailed_ext_table",
        candidate_status: "usable:detailed_ext_table".to_string(),
        window_lo,
        window_hi,
        blocking_reason: None,
    }
}

fn detailed_applicability_window(t: usize, tau: usize) -> (usize, Option<usize>) {
    if t == 0 {
        return (0, None);
    }
    (t.saturating_sub(tau), Some(t - 1))
}

fn detailed_spec_for_subalgebra(subalgebra: &Subalgebra) -> Option<DetailedFiniteSubalgebraSpec> {
    let name = subalgebra.name();
    DETAILED_FINITE_SUBALGEBRAS
        .iter()
        .copied()
        .find(|spec| spec.name == name && spec.profile == subalgebra.profile())
}

fn detailed_spec_by_name(name: &str) -> Option<DetailedFiniteSubalgebraSpec> {
    DETAILED_FINITE_SUBALGEBRAS
        .iter()
        .copied()
        .find(|spec| spec.name == name)
}

fn detailed_ext_table_for(name: &str) -> Option<Result<&'static DetailedExtTable, &'static str>> {
    detailed_ext_tables().iter().find_map(|load| match load {
        DetailedExtTableLoad::Loaded(table) if table.algebra_name == name => Some(Ok(table)),
        DetailedExtTableLoad::Missing { spec, .. } if spec.name == name => {
            Some(Err("missing_table"))
        }
        DetailedExtTableLoad::Invalid { spec, .. } if spec.name == name => {
            Some(Err("invalid_table"))
        }
        _ => None,
    })
}

fn detailed_ext_tables() -> &'static [DetailedExtTableLoad] {
    DETAILED_EXT_TABLES.get_or_init(|| {
        DETAILED_FINITE_SUBALGEBRAS
            .iter()
            .copied()
            .map(load_detailed_ext_table)
            .collect()
    })
}

fn load_detailed_ext_table(spec: DetailedFiniteSubalgebraSpec) -> DetailedExtTableLoad {
    let bitset_override = env::var(spec.bitset_env).ok();
    let metadata_override = env::var(spec.metadata_env).ok();
    let bitset_path = detailed_source_label(
        bitset_override.as_deref(),
        spec.default_bitset_path,
        spec.embedded_bitset,
    );
    let metadata_path = detailed_source_label(
        metadata_override.as_deref(),
        spec.default_metadata_path,
        spec.embedded_metadata,
    );

    if !detailed_source_exists(
        bitset_override.as_deref(),
        spec.default_bitset_path,
        spec.embedded_bitset,
    ) || !detailed_source_exists(
        metadata_override.as_deref(),
        spec.default_metadata_path,
        spec.embedded_metadata,
    ) {
        return DetailedExtTableLoad::Missing {
            spec,
            bitset_path,
            metadata_path,
        };
    }

    let loaded = (|| {
        let data = read_detailed_source(
            bitset_override.as_deref(),
            spec.default_bitset_path,
            spec.embedded_bitset,
            "bitset",
        )?;
        let metadata_bytes = read_detailed_source(
            metadata_override.as_deref(),
            spec.default_metadata_path,
            spec.embedded_metadata,
            "metadata",
        )?;
        load_detailed_ext_table_data(spec, &bitset_path, &metadata_path, data, &metadata_bytes)
    })();

    match loaded {
        Ok(table) => DetailedExtTableLoad::Loaded(table),
        Err(error) => DetailedExtTableLoad::Invalid {
            spec,
            bitset_path,
            metadata_path,
            error,
        },
    }
}

fn detailed_source_label(
    override_path: Option<&str>,
    default_path: &str,
    embedded: Option<&[u8]>,
) -> String {
    override_path.map(str::to_string).unwrap_or_else(|| {
        if embedded.is_some() {
            format!("embedded:{default_path}")
        } else {
            default_path.to_string()
        }
    })
}

fn detailed_source_exists(
    override_path: Option<&str>,
    default_path: &str,
    embedded: Option<&[u8]>,
) -> bool {
    override_path
        .map(|path| Path::new(path).exists())
        .unwrap_or_else(|| embedded.is_some() || Path::new(default_path).exists())
}

fn read_detailed_source(
    override_path: Option<&str>,
    default_path: &str,
    embedded: Option<&[u8]>,
    label: &str,
) -> Result<Vec<u8>, String> {
    if let Some(path) = override_path {
        return fs::read(path)
            .map_err(|err| format!("failed to read detailed {label} {path}: {err}"));
    }
    if let Some(bytes) = embedded {
        return Ok(bytes.to_vec());
    }
    fs::read(default_path)
        .map_err(|err| format!("failed to read detailed {label} {default_path}: {err}"))
}

fn load_detailed_ext_table_data(
    spec: DetailedFiniteSubalgebraSpec,
    bitset_path: &str,
    metadata_path: &str,
    data: Vec<u8>,
    metadata_bytes: &[u8],
) -> Result<DetailedExtTable, String> {
    let metadata: DetailedExtMetadata = serde_json::from_slice(metadata_bytes)
        .map_err(|err| format!("failed to parse detailed metadata {metadata_path}: {err}"))?;
    if metadata.format != "raw-bitset" {
        return Err(format!(
            "unsupported detailed Ext table format {:?}; expected raw-bitset",
            metadata.format
        ));
    }

    let s_count = metadata
        .bounds
        .s_max
        .checked_sub(metadata.bounds.s_min)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "invalid detailed Ext s bounds".to_string())?;
    let u_count = metadata
        .bounds
        .u_max
        .checked_sub(metadata.bounds.u_min)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "invalid detailed Ext internal-degree bounds".to_string())?;
    if metadata.shape != [s_count, u_count] {
        return Err(format!(
            "metadata shape {:?} does not match inclusive bounds {}x{}",
            metadata.shape, s_count, u_count
        ));
    }

    let bit_count = s_count
        .checked_mul(u_count)
        .ok_or_else(|| "detailed Ext table shape overflow".to_string())?;
    let expected_bytes = bit_count.div_ceil(8);
    if data.len() != expected_bytes {
        return Err(format!(
            "detailed bitset has {} bytes, expected {} for shape {}x{}",
            data.len(),
            expected_bytes,
            s_count,
            u_count
        ));
    }
    if metadata
        .byte_count
        .map(|byte_count| byte_count != expected_bytes)
        .unwrap_or(false)
    {
        return Err(format!(
            "metadata byte_count {:?} does not match expected {}",
            metadata.byte_count, expected_bytes
        ));
    }

    let nonzero_entries = count_nonzero_bits(&data, bit_count);
    Ok(DetailedExtTable {
        algebra_name: spec.name,
        profile: spec.profile,
        bitset_path: bitset_path.to_string(),
        metadata_path: metadata_path.to_string(),
        s_min: metadata.bounds.s_min,
        s_max: metadata.bounds.s_max,
        u_min: metadata.bounds.u_min,
        u_max: metadata.bounds.u_max,
        u_count,
        nonzero_entries,
        data,
    })
}

fn count_nonzero_bits(data: &[u8], bit_count: usize) -> usize {
    (0..bit_count)
        .filter(|&index| ((data[index / 8] >> (index % 8)) & 1) != 0)
        .count()
}

impl DetailedExtTable {
    fn query(&self, p: usize, u: usize) -> ExtQuery {
        if p < self.s_min || p > self.s_max || u < self.u_min || u > self.u_max {
            return ExtQuery::Unknown;
        }
        let row = p - self.s_min;
        let col = u - self.u_min;
        let index = row * self.u_count + col;
        let nonzero = ((self.data[index / 8] >> (index % 8)) & 1) != 0;
        if nonzero {
            ExtQuery::KnownNonzero(1)
        } else {
            ExtQuery::KnownZero
        }
    }
}

fn a2_detailed_support() -> &'static [u8] {
    A2_DETAILED_SUPPORT.get_or_init(|| {
        let data = match env::var("EXT_A2_EXT_BITSET") {
            Ok(path) => fs::read(&path).unwrap_or_else(|err| {
                panic!("failed to read A2 detailed support bitset {path}: {err}")
            }),
            Err(_) => A2_DETAILED_EMBEDDED_BITSET.to_vec(),
        };
        let expected = A2_DETAILED_SHAPE * A2_DETAILED_SHAPE;
        let expected_bytes = expected.div_ceil(8);
        assert_eq!(
            data.len(),
            expected_bytes,
            "A2 detailed support bitset has {} bytes, expected {} for a {A2_DETAILED_SHAPE}x{A2_DETAILED_SHAPE} raw bitset",
            data.len(),
            expected_bytes
        );
        data
    })
}

fn a2_ext_nonzero(s: usize, t: usize) -> bool {
    if s > A2_DETAILED_MAX_DEGREE || t > A2_DETAILED_MAX_DEGREE {
        return true;
    }
    let index = s * A2_DETAILED_SHAPE + t;
    let byte = a2_detailed_support()[index / 8];
    ((byte >> (index % 8)) & 1) != 0
}

fn a2_detailed_condition_applies(s: usize, t: usize) -> bool {
    for d in 1..=A2_DETAILED_D_MAX {
        let Some(shifted_t) = t.checked_sub(d) else {
            continue;
        };
        if a2_ext_nonzero(s, shifted_t) {
            return false;
        }
        if s > 0 && a2_ext_nonzero(s - 1, shifted_t) {
            return false;
        }
    }
    true
}

impl fmt::Display for Subalgebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

fn compatible_bit_order(profile: &[u32]) -> Vec<(usize, u32)> {
    let mut bit_order = Vec::new();
    for j in 1..=profile.len() {
        for bit in 0..profile[j - 1] {
            bit_order.push((j, bit));
        }
    }
    bit_order.sort_by(|(j_a, bit_a), (j_b, bit_b)| j_b.cmp(j_a).then_with(|| bit_b.cmp(bit_a)));
    bit_order
}

fn fprime_bit_order(n: usize, max_degree: usize) -> Vec<(usize, u32)> {
    let max_index = max_milnor_index(max_degree);
    let mut bit_order = Vec::new();
    for j in 1..=max_index {
        let max_value = (max_degree / weight(j)) as u32;
        let bits = bits_needed(max_value);
        let start_bit = if j < n {
            bits
        } else if j == n {
            1
        } else {
            0
        };
        for bit in start_bit..bits {
            bit_order.push((j, bit));
        }
    }
    bit_order.sort_by(|(j_a, bit_a), (j_b, bit_b)| j_b.cmp(j_a).then_with(|| bit_b.cmp(bit_a)));
    bit_order
}

fn generate_signatures(
    index: usize,
    profile: &[u32],
    max_degree: usize,
    current_degree: usize,
    entries: &mut [u32],
    signatures: &mut Vec<Milnor>,
) {
    if index == profile.len() {
        signatures.push(Milnor::new(entries.to_vec()));
        return;
    }
    let entry_weight = weight(index + 1);
    let max_by_degree = ((max_degree - current_degree) / entry_weight) as u32;
    let max_by_profile = if profile[index] >= u32::BITS {
        u32::MAX
    } else {
        (1_u32 << profile[index]).saturating_sub(1)
    };
    for value in 0..=max_by_profile.min(max_by_degree) {
        entries[index] = value;
        generate_signatures(
            index + 1,
            profile,
            max_degree,
            current_degree + value as usize * entry_weight,
            entries,
            signatures,
        );
    }
    entries[index] = 0;
}

fn generate_fprime_signatures(
    index: usize,
    n: usize,
    max_index: usize,
    max_degree: usize,
    current_degree: usize,
    entries: &mut [u32],
    signatures: &mut Vec<Milnor>,
) {
    if index > max_index {
        signatures.push(Milnor::new(entries.to_vec()));
        return;
    }

    let entry_weight = weight(index);
    let max_value = ((max_degree - current_degree) / entry_weight) as u32;
    if index < n {
        entries[index - 1] = 0;
        generate_fprime_signatures(
            index + 1,
            n,
            max_index,
            max_degree,
            current_degree,
            entries,
            signatures,
        );
    } else if index == n {
        for value in (0..=max_value).step_by(2) {
            entries[index - 1] = value;
            generate_fprime_signatures(
                index + 1,
                n,
                max_index,
                max_degree,
                current_degree + value as usize * entry_weight,
                entries,
                signatures,
            );
        }
    } else {
        for value in 0..=max_value {
            entries[index - 1] = value;
            generate_fprime_signatures(
                index + 1,
                n,
                max_index,
                max_degree,
                current_degree + value as usize * entry_weight,
                entries,
                signatures,
            );
        }
    }
    entries[index - 1] = 0;
}

fn signature_key(sig: &Milnor, bit_order: &[(usize, u32)]) -> Vec<u8> {
    bit_order
        .iter()
        .map(|(j, bit)| {
            let value = sig.entries().get(j - 1).copied().unwrap_or(0);
            ((value >> bit) & 1) as u8
        })
        .collect()
}

fn sort_signatures(family: Family, bit_order: &[(usize, u32)], signatures: &mut [Milnor]) {
    signatures.sort_by(|a, b| compare_signature_order(family, bit_order, a, b));
}

fn compare_signature_order(
    family: Family,
    bit_order: &[(usize, u32)],
    a: &Milnor,
    b: &Milnor,
) -> Ordering {
    match family {
        Family::F | Family::FPrime => a
            .degree()
            .cmp(&b.degree())
            .then_with(|| a.entries().cmp(b.entries()))
            .then_with(|| a.cmp(b)),
        Family::A | Family::B => signature_key(a, bit_order)
            .cmp(&signature_key(b, bit_order))
            .then_with(|| a.cmp(b)),
    }
}

fn validate_finite_profile(profile: &[u32]) -> Result<(), String> {
    if profile.is_empty() {
        return Err("finite profile must have at least one coordinate".to_string());
    }
    if profile.iter().all(|&entry| entry == 0) {
        return Err("finite profile must have a positive coordinate".to_string());
    }
    for &entry in profile {
        if entry >= usize::BITS {
            return Err(format!(
                "finite profile coordinate {entry} is too large for this build"
            ));
        }
    }
    Ok(())
}

fn validate_registered_b_profile(profile: &[u32]) -> Result<(), String> {
    match profile {
        [3, 2, 1, 1] | [3, 2, 2, 1] | [3, 3, 2, 1] => Ok(()),
        _ => Err(format!(
            "unsupported B-profile {}; supported lower-line profiles are B3211, B3221, and B3321",
            profile_label(profile)
        )),
    }
}

fn profile_label(profile: &[u32]) -> String {
    let suffix = profile
        .iter()
        .map(|entry| entry.to_string())
        .collect::<Vec<_>>()
        .join("");
    format!("B{suffix}")
}

fn profile_fingerprint(profile: &[u32]) -> usize {
    let mut hash = 0xcbf29ce484222325usize;
    for &entry in profile {
        hash ^= entry as usize;
        hash = hash.wrapping_mul(0x100000001b3usize);
    }
    hash ^= profile.len();
    hash
}

fn parse_b_profile(input: &str) -> Result<Option<(String, Vec<u32>)>, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let Some(rest) = trimmed
        .strip_prefix('B')
        .or_else(|| trimmed.strip_prefix('b'))
    else {
        return Ok(None);
    };
    let profile = if let Some(inner) = rest
        .strip_prefix('(')
        .and_then(|text| text.strip_suffix(')'))
    {
        let mut entries = Vec::new();
        for part in inner.split(',') {
            let text = part.trim();
            if text.is_empty() {
                return Err(format!(
                    "invalid profile `{input}`; empty B-profile coordinate"
                ));
            }
            let value = text
                .parse::<u32>()
                .map_err(|_| format!("invalid profile `{input}`; coordinates must be integers"))?;
            entries.push(value);
        }
        entries
    } else {
        if rest.is_empty() || !rest.chars().all(|ch| ch.is_ascii_digit()) {
            return Err(format!(
                "invalid profile `{input}`; use B3211 or B(3,2,1,1)"
            ));
        }
        rest.chars()
            .map(|ch| ch.to_digit(10).expect("ASCII digit"))
            .collect::<Vec<_>>()
    };
    validate_finite_profile(&profile)?;
    Ok(Some((profile_label(&profile), profile)))
}

fn parse_family_index(input: &str, family: char) -> Option<&str> {
    let family_string = family.to_string();
    if let Some(inner) = input
        .strip_prefix(&(family_string.clone() + "("))
        .and_then(|s| s.strip_suffix(')'))
    {
        return Some(inner);
    }
    input.strip_prefix(family)
}

fn parse_fprime_index(input: &str) -> Option<&str> {
    for prefix in ["Fprime", "fprime", "Fp", "fp", "F'", "f'"] {
        if let Some(inner) = input
            .strip_prefix(&(prefix.to_string() + "("))
            .and_then(|s| s.strip_suffix(')'))
        {
            return Some(inner);
        }
        if let Some(rest) = input.strip_prefix(prefix) {
            return Some(rest);
        }
    }
    None
}

fn max_milnor_index(degree: usize) -> usize {
    let mut index = 0;
    while weight(index + 1) <= degree {
        index += 1;
    }
    index
}

fn bits_needed(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        u32::BITS - value.leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::milnor::{
        basis_of_degree, multiply_packed_btrivial_keys_with_row_cache,
        multiply_packed_keys_with_row_cache_matching,
    };
    use std::collections::BTreeSet;

    #[test]
    fn subalgebra_selection_default_alias_tracks_detailed_default() {
        assert_eq!(
            SubalgebraSelectionMode::parse("default").unwrap(),
            SubalgebraSelectionMode::Detailed
        );
        assert_eq!(
            SubalgebraSelectionMode::parse("current").unwrap(),
            SubalgebraSelectionMode::Detailed
        );
        assert_eq!(
            SubalgebraSelectionMode::parse("original").unwrap(),
            SubalgebraSelectionMode::Original
        );
    }

    #[test]
    fn a1_has_eight_signatures() {
        let a1 = Subalgebra::a(1).unwrap();
        assert_eq!(a1.signatures().len(), 8);
        assert_eq!(a1.signatures()[0], Milnor::one());
        assert_eq!(a1.signature(&Milnor::parse("4,2").unwrap()), Milnor::one());
    }

    #[test]
    fn tau_values_for_a_n() {
        assert_eq!(tau_a(0), 1);
        assert_eq!(tau_a(1), 6);
        assert_eq!(tau_a(2), 23);
        assert_eq!(weight(3), 7);
    }

    #[test]
    fn finite_profile_invariants_match_lower_line_data() {
        let cases = [
            (
                Subalgebra::a(3).unwrap(),
                "A3",
                vec![4, 3, 2, 1],
                1024,
                72,
                15,
            ),
            (
                Subalgebra::b3321().unwrap(),
                "B3321",
                vec![3, 3, 2, 1],
                512,
                64,
                15,
            ),
            (
                Subalgebra::b3221().unwrap(),
                "B3221",
                vec![3, 2, 2, 1],
                256,
                52,
                15,
            ),
            (
                Subalgebra::b3211().unwrap(),
                "B3211",
                vec![3, 2, 1, 1],
                128,
                38,
                15,
            ),
            (Subalgebra::a(2).unwrap(), "A2", vec![3, 2, 1], 64, 23, 7),
            (Subalgebra::a(1).unwrap(), "A1", vec![2, 1], 8, 6, 3),
        ];

        for (subalgebra, name, profile, dim, tau, d) in cases {
            assert_eq!(subalgebra.name(), name);
            assert_eq!(subalgebra.profile(), profile.as_slice());
            assert_eq!(subalgebra.profile_dim(), dim, "{name} dim");
            assert_eq!(subalgebra.signatures().len(), dim, "{name} signatures");
            assert_eq!(subalgebra.profile_tau(), tau, "{name} tau");
            assert_eq!(subalgebra.profile_d(), d, "{name} d");
        }
    }

    #[test]
    fn detailed_registry_tau_values_match_conditions() {
        let cases = [
            ("A2", 23),
            ("B3211", 38),
            ("B3221", 52),
            ("B3321", 64),
            ("A3", 72),
        ];
        for (name, tau) in cases {
            let spec = detailed_spec_by_name(name).unwrap();
            assert_eq!(spec.tau, tau, "{name} tau");
        }
    }

    #[test]
    fn detailed_signature_degrees_fill_zero_to_tau() {
        let subalgebras = [
            Subalgebra::a(2).unwrap(),
            Subalgebra::b3211().unwrap(),
            Subalgebra::b3221().unwrap(),
            Subalgebra::b3321().unwrap(),
            Subalgebra::a(3).unwrap(),
        ];

        for subalgebra in subalgebras {
            let degrees = subalgebra
                .signatures()
                .iter()
                .map(Milnor::degree)
                .collect::<BTreeSet<_>>();
            let tau = subalgebra.profile_tau();
            let expected = (0..=tau).collect::<BTreeSet<_>>();
            assert_eq!(
                degrees,
                expected,
                "{} signature degrees should fill 0..={tau}",
                subalgebra.name()
            );
        }
    }

    #[test]
    fn detailed_applicability_windows_have_expected_boundaries() {
        assert_eq!(detailed_applicability_window(0, 52), (0, None));
        assert_eq!(detailed_applicability_window(1, 52), (0, Some(0)));
        assert_eq!(detailed_applicability_window(52, 52), (0, Some(51)));
        assert_eq!(detailed_applicability_window(53, 52), (1, Some(52)));
        assert_eq!(detailed_applicability_window(150, 52), (98, Some(149)));
    }

    #[test]
    fn original_b_lower_rule_uses_s_not_s_plus_one() {
        let subalgebra = Subalgebra::b3221().unwrap();
        for s in [0, 1, 6, 7, 8] {
            let boundary = subalgebra.lower_line_bound(s);
            assert!(
                !subalgebra.lower_line_applies(s, boundary),
                "B3221 should be false on T=d*s+tau boundary at s={s}"
            );
            assert!(
                subalgebra.lower_line_applies(s, boundary + 1),
                "B3221 should be true just above T=d*s+tau at s={s}"
            );
        }
    }

    #[test]
    fn original_a_lower_rule_uses_s_not_s_plus_one() {
        for subalgebra in [Subalgebra::a(0).unwrap(), Subalgebra::a(1).unwrap()] {
            for s in [0, 1, 6, 7, 8] {
                let boundary = subalgebra.lower_line_bound(s);
                assert!(
                    !subalgebra.lower_line_applies(s, boundary),
                    "{} should be false on T=d*s+tau boundary at s={s}",
                    subalgebra.name()
                );
                assert!(
                    subalgebra.lower_line_applies(s, boundary + 1),
                    "{} should be true just above T=d*s+tau at s={s}",
                    subalgebra.name()
                );
            }
        }
    }

    #[test]
    fn detailed_table_query_unknowns_outside_declared_bounds() {
        let table = detailed_table_for_test("B3221", 0, 512, 0, 512, &[]);
        assert_eq!(table.query(0, 0), ExtQuery::KnownZero);
        assert_eq!(table.query(512, 512), ExtQuery::KnownZero);
        assert_eq!(table.query(513, 0), ExtQuery::Unknown);
        assert_eq!(table.query(0, 513), ExtQuery::Unknown);
    }

    #[test]
    fn detailed_s_zero_skips_s_minus_one_row() {
        let spec = detailed_spec_by_name("B3221").unwrap();
        let table = detailed_table_for_test("B3221", 0, 0, 0, 0, &[]);
        let result = detailed_table_applicability(spec, &table, 0, 1);
        assert!(result.usable);
        assert_eq!(result.certification_source, "detailed_ext_table");
    }

    #[test]
    fn detailed_nonzero_entry_in_row_s_blocks() {
        let spec = detailed_spec_by_name("B3221").unwrap();
        let table = detailed_table_for_test("B3221", 2, 2, 0, 2, &[(2, 0)]);
        let result = detailed_table_applicability(spec, &table, 2, 1);
        assert!(!result.usable);
        assert_eq!(result.blocking_reason, Some("nonzero_Ext_row_s"));
    }

    #[test]
    fn detailed_nonzero_entry_in_row_s_minus_one_blocks() {
        let spec = detailed_spec_by_name("B3221").unwrap();
        let table = detailed_table_for_test("B3221", 1, 2, 0, 2, &[(1, 0)]);
        let result = detailed_table_applicability(spec, &table, 2, 1);
        assert!(!result.usable);
        assert_eq!(result.blocking_reason, Some("nonzero_Ext_row_s_minus_1"));
    }

    #[test]
    fn detailed_unknown_outside_table_blocks_certification() {
        let spec = detailed_spec_by_name("B3221").unwrap();
        let table = detailed_table_for_test("B3221", 0, 1, 0, 1, &[]);
        let result = detailed_table_applicability(spec, &table, 2, 1);
        assert!(!result.usable);
        assert_eq!(result.blocking_reason, Some("unknown_out_of_range"));
    }

    #[test]
    fn generalized_a2_detailed_agrees_with_existing_a2_logic_inside_table() {
        let spec = detailed_spec_by_name("A2").unwrap();
        let table = detailed_ext_table_for("A2")
            .and_then(Result::ok)
            .expect("tracked Ext_A2 detailed table should load");
        for s in 0..=8 {
            for t in 0..=80 {
                let generic = detailed_table_applicability(spec, table, s, t).usable;
                let existing = a2_detailed_condition_applies(s, t);
                assert_eq!(generic, existing, "A2 mismatch at s={s}, t={t}");
            }
        }
    }

    fn detailed_table_for_test(
        algebra_name: &'static str,
        s_min: usize,
        s_max: usize,
        u_min: usize,
        u_max: usize,
        nonzero: &[(usize, usize)],
    ) -> DetailedExtTable {
        let u_count = u_max - u_min + 1;
        let bit_count = (s_max - s_min + 1) * u_count;
        let mut data = vec![0_u8; bit_count.div_ceil(8)];
        for &(s, u) in nonzero {
            let index = (s - s_min) * u_count + (u - u_min);
            data[index / 8] |= 1 << (index % 8);
        }
        let spec = detailed_spec_by_name(algebra_name).unwrap();
        DetailedExtTable {
            algebra_name: spec.name,
            profile: spec.profile,
            bitset_path: "test.bin".to_string(),
            metadata_path: "test.json".to_string(),
            s_min,
            s_max,
            u_min,
            u_max,
            u_count,
            nonzero_entries: nonzero.len(),
            data,
        }
    }

    #[test]
    fn finite_profile_lower_line_is_strict() {
        let cases = [
            (Subalgebra::b3321().unwrap(), 64),
            (Subalgebra::b3221().unwrap(), 52),
            (Subalgebra::b3211().unwrap(), 38),
        ];
        for (subalgebra, tau) in cases {
            for s in [0, 1, 6, 7, 8] {
                let boundary = 15 * s + tau;
                assert!(
                    !subalgebra.lower_line_applies(s, boundary),
                    "{} should be false on the boundary t={boundary}, s={s}",
                    subalgebra.name()
                );
                assert!(
                    subalgebra.lower_line_applies(s, boundary + 1),
                    "{} should be true just above the boundary t={}, s={s}",
                    subalgebra.name(),
                    boundary + 1
                );
            }
        }
    }

    #[test]
    fn finite_profile_parse_and_signature_use_profile_residues() {
        let b = Subalgebra::parse("B(3,2,1,1)", 80).unwrap();
        assert_eq!(b.name(), "B3211");
        assert_eq!(b.profile(), &[3, 2, 1, 1]);
        assert_eq!(
            b.signature(&Milnor::parse("9,6,3,2").unwrap()),
            Milnor::parse("1,2,1,0").unwrap()
        );

        let packed = Milnor::parse("8,4,2,2").unwrap().packed().unwrap();
        assert!(b.profile_signature_is_zero_packed_unchecked(packed));
        let (signature, quotient) = b.split_profile_signature_packed(packed).unwrap();
        assert_eq!(signature, 0);
        assert_eq!(quotient, packed);
    }

    #[test]
    fn unregistered_b_profiles_are_rejected() {
        for input in ["B321", "B4321", "B(3,2,1)", "B(4,3,2,1)", "B(2,2)"] {
            let err = Subalgebra::parse(input, 80).unwrap_err();
            assert!(
                err.contains("supported lower-line profiles"),
                "{input} should be rejected as an unregistered B-profile, got {err}"
            );
        }
    }

    #[test]
    fn btrivial_products_match_full_products_filtered_to_zero_signature() {
        let subalgebras = [
            Subalgebra::b3211().unwrap(),
            Subalgebra::b3221().unwrap(),
            Subalgebra::b3321().unwrap(),
        ];
        let basis = (0..=18).flat_map(basis_of_degree).collect::<Vec<_>>();

        for subalgebra in subalgebras {
            let mut full_cache = HashMap::default();
            let mut btrivial_cache = HashMap::default();
            for left in &basis {
                for right in &basis {
                    if left.degree() + right.degree() > 18 {
                        continue;
                    }
                    let left_key = left.packed().unwrap();
                    let right_key = right.packed().unwrap();
                    let full_zero = multiply_packed_keys_with_row_cache_matching(
                        left_key,
                        right_key,
                        &mut full_cache,
                        |packed| subalgebra.profile_signature_is_zero_packed_unchecked(packed),
                    );
                    let btrivial_zero = multiply_packed_btrivial_keys_with_row_cache(
                        left_key,
                        right_key,
                        &mut btrivial_cache,
                        subalgebra.profile(),
                        |packed| subalgebra.profile_signature_is_zero_packed_unchecked(packed),
                    );
                    assert_eq!(
                        btrivial_zero,
                        full_zero,
                        "{} E0 product mismatch for {left} * {right}",
                        subalgebra.name()
                    );
                }
            }
        }
    }

    #[test]
    fn f1_tracks_high_tail_as_signature() {
        let f1 = Subalgebra::f(1, 20).unwrap();
        assert_eq!(f1.signature(&Milnor::parse("19").unwrap()), Milnor::one());
        assert_eq!(
            f1.signature(&Milnor::parse("1,1").unwrap()),
            Milnor::parse("0,1").unwrap()
        );
        assert!(f1.lower_line_applies(10, 20));
        assert!(!f1.lower_line_applies(1, 20));
    }

    #[test]
    fn upper_methods_use_strict_upper_lines() {
        let cases = [
            (Subalgebra::fprime(3, 160).unwrap(), 14),
            (Subalgebra::f(2, 160).unwrap(), 7),
            (Subalgebra::fprime(2, 160).unwrap(), 6),
            (Subalgebra::f(1, 160).unwrap(), 3),
            (Subalgebra::fprime(1, 160).unwrap(), 2),
        ];
        for (subalgebra, slope) in cases {
            for s in [1, 7, 19, 80] {
                let boundary = slope * s;
                assert!(
                    !subalgebra.lower_line_applies(s, boundary),
                    "{} should be false on the upper boundary T={boundary}, S={s}",
                    subalgebra.name()
                );
                assert!(
                    subalgebra.lower_line_applies(s, boundary - 1),
                    "{} should be true just below the upper boundary T={}, S={s}",
                    subalgebra.name(),
                    boundary - 1
                );
            }
        }
    }

    #[test]
    fn upper_quotient_counts_match_closed_forms() {
        let f1 = Subalgebra::f(1, 160).unwrap();
        assert_eq!(f1.quotient_count(0), 1);
        assert_eq!(f1.quotient_count(37), 1);
        assert_eq!(f1.quotient_count(160), 1);

        let fp1 = Subalgebra::fprime(1, 160).unwrap();
        assert_eq!(fp1.quotient_count(0), 1);
        assert_eq!(fp1.quotient_count(1), 1);
        assert_eq!(fp1.quotient_count(2), 0);
        assert_eq!(fp1.quotient_count(160), 0);

        let f2 = Subalgebra::f(2, 160).unwrap();
        assert_eq!(f2.quotient_count(0), 1);
        assert_eq!(f2.quotient_count(1), 1);
        assert_eq!(f2.quotient_count(2), 1);
        assert_eq!(f2.quotient_count(3), 2);
        assert_eq!(f2.quotient_count(160), 54);

        let fp2 = Subalgebra::fprime(2, 160).unwrap();
        assert_eq!(fp2.quotient_count(0), 1);
        assert_eq!(fp2.quotient_count(1), 1);
        assert_eq!(fp2.quotient_count(2), 1);
        assert_eq!(fp2.quotient_count(3), 2);
        assert_eq!(fp2.quotient_count(160), 2);

        let fp3 = Subalgebra::fprime(3, 160).unwrap();
        assert_eq!(fp3.quotient_count(0), 1);
        assert_eq!(fp3.quotient_count(7), 4);
        assert_eq!(fp3.quotient_count(160), 106);
    }

    #[test]
    fn profile_signature_indices_match_signature_order() {
        let subalgebras = [
            Subalgebra::a(0).unwrap(),
            Subalgebra::a(1).unwrap(),
            Subalgebra::a(2).unwrap(),
            Subalgebra::a(3).unwrap(),
            Subalgebra::b3211().unwrap(),
            Subalgebra::b3221().unwrap(),
            Subalgebra::b3321().unwrap(),
            Subalgebra::f(1, 30).unwrap(),
            Subalgebra::f(2, 30).unwrap(),
            Subalgebra::fprime(1, 30).unwrap(),
            Subalgebra::fprime(2, 30).unwrap(),
            Subalgebra::fprime(3, 30).unwrap(),
        ];

        for subalgebra in subalgebras {
            assert_eq!(subalgebra.signatures()[0], Milnor::one());
            for (index, signature) in subalgebra.signatures().iter().enumerate() {
                assert_eq!(
                    subalgebra.signature_index(signature),
                    index,
                    "{} signature {signature}",
                    subalgebra.name()
                );
            }
            for degree in 0..=30 {
                for coeff in basis_of_degree(degree) {
                    let index = subalgebra.signature_index(&coeff);
                    let zero_index = subalgebra.signature_index_packed(0);
                    assert_eq!(
                        &subalgebra.signature(&coeff),
                        &subalgebra.signatures()[index]
                    );
                    let packed = coeff.packed().unwrap();
                    assert_eq!(
                        subalgebra.signature_is_zero_packed(packed),
                        index == zero_index,
                        "{} coeff {coeff}",
                        subalgebra.name()
                    );
                    let (signature, quotient) = subalgebra.split_signature_packed(packed).unwrap();
                    if index == zero_index {
                        assert_eq!(
                            quotient,
                            packed,
                            "{} zero-signature coeff {coeff}",
                            subalgebra.name()
                        );
                    }
                    assert_eq!(subalgebra.quotient_part_packed(packed), Some(quotient));
                    assert!(subalgebra.signature_is_zero_packed(quotient));
                    assert_eq!(
                        subalgebra.compose_signature_with_quotient_packed(signature, quotient),
                        Some(packed)
                    );
                    assert_eq!(Milnor::from_packed(signature), subalgebra.signature(&coeff));
                    for i in 0..PACKED_ENTRY_LIMIT {
                        assert_eq!(
                            packed_entry(signature, i) + packed_entry(quotient, i),
                            packed_entry(packed, i)
                        );
                    }
                    if subalgebra.profile_cache_key().is_some() {
                        assert_eq!(
                            subalgebra.profile_signature_is_zero_packed_unchecked(packed),
                            index == zero_index,
                            "{} coeff {coeff}",
                            subalgebra.name()
                        );
                        let (profile_signature, profile_quotient) =
                            subalgebra.split_profile_signature_packed(packed).unwrap();
                        assert_eq!(profile_signature, signature);
                        assert_eq!(profile_quotient, quotient);
                        assert_eq!(subalgebra.profile_quotient_packed(packed), Some(quotient));
                    }
                }
            }
        }
    }

    #[test]
    fn upper_signature_decomposition_round_trips_through_degree_80() {
        let subalgebras = [
            Subalgebra::f(1, 80).unwrap(),
            Subalgebra::fprime(1, 80).unwrap(),
            Subalgebra::f(2, 80).unwrap(),
            Subalgebra::fprime(2, 80).unwrap(),
            Subalgebra::fprime(3, 80).unwrap(),
        ];
        for subalgebra in subalgebras {
            for degree in 0..=80 {
                for coeff in basis_of_degree(degree) {
                    let packed = coeff.packed().unwrap();
                    let (sig, quotient) = subalgebra.split_signature_packed(packed).unwrap();
                    assert_eq!(
                        subalgebra.compose_signature_with_quotient_packed(sig, quotient),
                        Some(packed),
                        "{} failed to recompose {coeff}",
                        subalgebra.name()
                    );
                    assert!(
                        subalgebra.signature_is_zero_packed(quotient),
                        "{} quotient of {coeff} is not signature-zero",
                        subalgebra.name()
                    );
                    assert!(
                        subalgebra.same_signature_packed(packed, sig),
                        "{} signature changed after split for {coeff}",
                        subalgebra.name()
                    );
                }
            }
        }
    }

    #[test]
    fn upper_projected_products_land_in_quotient_basis() {
        let subalgebras = [
            Subalgebra::f(1, 50).unwrap(),
            Subalgebra::fprime(1, 50).unwrap(),
            Subalgebra::f(2, 50).unwrap(),
            Subalgebra::fprime(2, 50).unwrap(),
            Subalgebra::fprime(3, 50).unwrap(),
        ];
        let basis_by_degree = (0..=50).map(basis_of_degree).collect::<Vec<_>>();

        for subalgebra in subalgebras {
            let mut quotient_basis_by_degree = (0..=50)
                .map(|degree| {
                    subalgebra
                        .quotient_basis(degree)
                        .into_iter()
                        .map(|packed| (packed, ()))
                        .collect::<HashMap<_, _>>()
                })
                .collect::<Vec<_>>();
            let mut row_cache = HashMap::default();
            for left_degree in 0..=50 {
                for right_degree in 0..=50 - left_degree {
                    for left in &basis_by_degree[left_degree] {
                        let left_key = left.packed().unwrap();
                        let (sig, _) = subalgebra.split_signature_packed(left_key).unwrap();
                        for right in &basis_by_degree[right_degree] {
                            let right_key = right.packed().unwrap();
                            let products = multiply_packed_keys_with_row_cache_matching(
                                left_key,
                                right_key,
                                &mut row_cache,
                                |_| true,
                            );
                            for product in products {
                                if !subalgebra.same_signature_packed(product, sig) {
                                    continue;
                                }
                                let quotient = subalgebra.quotient_part_packed(product).unwrap();
                                let quotient_degree = Milnor::from_packed(quotient).degree();
                                assert!(
                                    quotient_basis_by_degree[quotient_degree]
                                        .contains_key(&quotient),
                                    "{} projected product quotient {} is not in quotient basis degree {}",
                                    subalgebra.name(),
                                    Milnor::from_packed(quotient),
                                    quotient_degree
                                );
                            }
                        }
                    }
                }
            }
            quotient_basis_by_degree.clear();
        }
    }

    #[test]
    fn fprime_tracks_even_part_and_high_tail_as_signature() {
        let fp1 = Subalgebra::fprime(1, 20).unwrap();
        assert_eq!(fp1.signature(&Milnor::parse("1").unwrap()), Milnor::one());
        assert_eq!(
            fp1.signature(&Milnor::parse("2").unwrap()),
            Milnor::parse("2").unwrap()
        );
        assert!(fp1.lower_line_applies(11, 20));
        assert!(!fp1.lower_line_applies(10, 20));

        let fp2 = Subalgebra::fprime(2, 20).unwrap();
        assert_eq!(fp2.signature(&Milnor::parse("5,1").unwrap()), Milnor::one());
        assert_eq!(
            fp2.signature(&Milnor::parse("5,2").unwrap()),
            Milnor::parse("0,2").unwrap()
        );
        assert_eq!(
            fp2.signature(&Milnor::parse("5,1,1").unwrap()),
            Milnor::parse("0,0,1").unwrap()
        );
    }
}
