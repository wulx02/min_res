// Portions of this file were translated, structurally adapted, or implemented
// with reference to SSeqCpp and SpectralSequences/sseq, then modified for this
// project. Each affected routine is labeled below; see
// PROVENANCE.md, THIRD_PARTY_NOTICES.md, and LICENSE-APACHE.

use std::cmp::Ordering;
use std::fmt;

use crate::fast_hash::{FastHashMap as HashMap, FastHashSet as HashSet};

pub type CoeffKey = u64;

pub const PACKED_ENTRY_LIMIT: usize = 9;
pub const PACKED_BASIS_UNCHECKED_MAX_DEGREE: usize = 512;
const PACKED_ENTRY_WIDTHS: [usize; PACKED_ENTRY_LIMIT] = [10, 8, 7, 6, 5, 4, 3, 2, 1];
const M0: CoeffKey = (1_u64 << 10) - 1;
const M1: CoeffKey = (1_u64 << 8) - 1;
const M2: CoeffKey = (1_u64 << 7) - 1;
const M3: CoeffKey = (1_u64 << 6) - 1;
const M4: CoeffKey = (1_u64 << 5) - 1;
const M5: CoeffKey = (1_u64 << 4) - 1;
const M6: CoeffKey = (1_u64 << 3) - 1;
const M7: CoeffKey = (1_u64 << 2) - 1;
const M8: CoeffKey = (1_u64 << 1) - 1;
const PRODUCT_COLUMN_LIMIT: usize = PACKED_ENTRY_LIMIT + 1;
const PRODUCT_DIAG_LIMIT: usize = PACKED_ENTRY_LIMIT + PRODUCT_COLUMN_LIMIT;

pub struct RowDecompOptions {
    width: usize,
    data: Vec<u32>,
}

impl RowDecompOptions {
    fn new(width: usize, data: Vec<u32>) -> Self {
        debug_assert!(width > 0);
        debug_assert_eq!(data.len() % width, 0);
        Self { width, data }
    }

    fn rows(&self) -> impl Iterator<Item = &[u32]> {
        self.data.chunks_exact(self.width)
    }

    pub fn row_count(&self) -> usize {
        self.data.len() / self.width
    }

    pub fn storage_capacity(&self) -> usize {
        self.data.capacity()
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Milnor {
    entries: Vec<u32>,
}

impl Milnor {
    pub fn new(mut entries: Vec<u32>) -> Self {
        while entries.last() == Some(&0) {
            entries.pop();
        }
        Self { entries }
    }

    pub fn one() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn parse(input: &str) -> Result<Self, String> {
        let trimmed = input.trim();
        if trimmed.is_empty()
            || trimmed == "0"
            || trimmed == "()"
            || trimmed == "unit"
            || trimmed == "Sq()"
        {
            return Ok(Self::one());
        }
        let entries = trimmed
            .trim_start_matches("Sq(")
            .trim_end_matches(')')
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<u32>()
                    .map_err(|_| format!("invalid Milnor entry in `{input}`"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(entries))
    }

    pub fn entries(&self) -> &[u32] {
        &self.entries
    }

    // Provenance: implementation reference to SSeqCpp `MMilnor::deg` and
    // sseq's `xi_degrees`; this is the standard Milnor-basis degree formula.
    pub fn degree(&self) -> usize {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, &r)| r as usize * weight(i + 1))
            .sum()
    }

    // Provenance: implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
    // and sseq `MilnorHashMap::code`. This project uses a different bit layout.
    pub fn packed(&self) -> Option<CoeffKey> {
        pack_entries(&self.entries)
    }

    // Provenance: implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
    // and sseq `MilnorHashMap::code`. See PROVENANCE.md.
    pub fn from_packed(packed: CoeffKey) -> Self {
        milnor_from_packed(packed)
    }
}

impl Ord for Milnor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.degree()
            .cmp(&other.degree())
            .then_with(|| self.entries.cmp(&other.entries))
    }
}

impl PartialOrd for Milnor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Milnor {
    // Provenance: implementation reference to SSeqCpp `MMilnor::Str`; the
    // cited sseq revision uses `P(...)`, not this `Sq(...)` format.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.entries.is_empty() {
            return write!(f, "1");
        }
        let body = self
            .entries
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        write!(f, "Sq({body})")
    }
}

// Provenance: shared Milnor-basis mathematics also used by SSeqCpp and sseq;
// this formula is not a source-code translation. See PROVENANCE.md.
pub fn weight(index: usize) -> usize {
    (1usize << index) - 1
}

pub fn tau_a(n: usize) -> usize {
    (1..=n + 1)
        .map(|j| ((1usize << (n + 2 - j)) - 1) * weight(j))
        .sum()
}

// Provenance: implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi` and
// sseq `MilnorHashMap::code`. This project uses a different bit layout.
pub fn pack_entries(entries: &[u32]) -> Option<CoeffKey> {
    if entries.len() > PACKED_ENTRY_LIMIT {
        return None;
    }
    let mut padded = [0_u32; PACKED_ENTRY_LIMIT];
    for (i, &entry) in entries.iter().enumerate() {
        padded[i] = entry;
    }
    pack_padded_entries(&padded)
}

// Provenance: implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi` and
// sseq `MilnorHashMap::code`; the local field widths are different.
pub(crate) fn pack_padded_entries(entries: &[u32; PACKED_ENTRY_LIMIT]) -> Option<CoeffKey> {
    for (i, &entry) in entries.iter().enumerate() {
        if entry as CoeffKey > packed_entry_mask(i) {
            return None;
        }
    }
    Some(pack_padded_entries_unchecked(entries))
}

#[inline]
// Provenance: same packing references as `pack_padded_entries`; this is the
// unchecked local-layout encoder. See PROVENANCE.md.
pub(crate) fn pack_padded_entries_unchecked(entries: &[u32; PACKED_ENTRY_LIMIT]) -> CoeffKey {
    (entries[0] as CoeffKey)
        | ((entries[1] as CoeffKey) << 10)
        | ((entries[2] as CoeffKey) << 18)
        | ((entries[3] as CoeffKey) << 25)
        | ((entries[4] as CoeffKey) << 31)
        | ((entries[5] as CoeffKey) << 36)
        | ((entries[6] as CoeffKey) << 40)
        | ((entries[7] as CoeffKey) << 43)
        | ((entries[8] as CoeffKey) << 45)
}

#[inline]
// Provenance: implementation reference to SSeqCpp `MMilnor` conversion and
// sseq `MilnorHashMap::code`; direct field access and bit positions are local.
pub fn packed_entry(packed: CoeffKey, index: usize) -> u32 {
    match index {
        0 => (packed & M0) as u32,
        1 => ((packed >> 10) & M1) as u32,
        2 => ((packed >> 18) & M2) as u32,
        3 => ((packed >> 25) & M3) as u32,
        4 => ((packed >> 31) & M4) as u32,
        5 => ((packed >> 36) & M5) as u32,
        6 => ((packed >> 40) & M6) as u32,
        7 => ((packed >> 43) & M7) as u32,
        8 => ((packed >> 45) & M8) as u32,
        _ => 0,
    }
}

#[inline]
// Provenance: local-layout helper for the packed representations referenced
// from SSeqCpp `MMilnor` and sseq `MilnorHashMap::code`. See PROVENANCE.md.
fn packed_entry_mask(index: usize) -> CoeffKey {
    (1_u64 << PACKED_ENTRY_WIDTHS[index]) - 1
}

#[cfg(test)]
// Provenance: test wrapper around the Milnor-basis enumeration referenced
// against sseq `MilnorAlgebra::{compute_ppart, generate_basis_2}`.
pub fn basis_through_degree(max_degree: usize) -> Vec<Vec<Milnor>> {
    (0..=max_degree).map(basis_of_degree).collect()
}

#[cfg(test)]
// Provenance: packed test wrapper around the Milnor-basis enumeration
// referenced against sseq `MilnorAlgebra::{compute_ppart, generate_basis_2}`.
pub fn basis_keys_through_degree(max_degree: usize) -> Vec<Vec<CoeffKey>> {
    let counts = packed_basis_counts_through_degree(max_degree);
    (0..=max_degree)
        .map(|degree| basis_keys_of_degree_with_capacity(degree, counts[degree]))
        .collect()
}

// Provenance: implementation reference and shared mathematics with sseq
// `MilnorAlgebra::{compute_ppart, generate_basis_2}`; this recursion is local.
pub fn basis_of_degree(degree: usize) -> Vec<Milnor> {
    if degree == 0 {
        return vec![Milnor::one()];
    }
    let max_index = max_milnor_index(degree);
    let mut current = vec![0; max_index];
    let mut out = Vec::new();
    basis_rec(1, max_index, degree, &mut current, &mut out);
    out.sort();
    out
}

// Provenance: implementation reference and shared mathematics with sseq
// `MilnorAlgebra::{compute_ppart, generate_basis_2}`; this body is local.
pub fn basis_keys_of_degree(degree: usize) -> Vec<CoeffKey> {
    basis_keys_of_degree_with_capacity(degree, packed_basis_count_of_degree(degree))
}

// Provenance: implementation reference and shared mathematics with sseq
// `MilnorAlgebra::{compute_ppart, generate_basis_2}`. See PROVENANCE.md.
fn basis_keys_of_degree_with_capacity(degree: usize, capacity: usize) -> Vec<CoeffKey> {
    if degree == 0 {
        return vec![0];
    }
    let max_index = max_milnor_index(degree);
    assert!(
        max_index <= PACKED_ENTRY_LIMIT,
        "Milnor basis degree {degree} needs {max_index} entries, but the packed coefficient format stores {PACKED_ENTRY_LIMIT}"
    );
    let mut current = [0_u32; PACKED_ENTRY_LIMIT];
    let mut out = Vec::with_capacity(capacity);
    let checked_pack = degree > PACKED_BASIS_UNCHECKED_MAX_DEGREE;
    basis_keys_rec(1, max_index, degree, &mut current, &mut out, checked_pack);
    debug_assert_eq!(out.len(), capacity);
    out
}

// Provenance: implementation reference to sseq's degree-indexed Milnor-basis
// construction; this dynamic-programming count is a local implementation.
fn packed_basis_counts_through_degree(max_degree: usize) -> Vec<usize> {
    let max_index = max_milnor_index(max_degree);
    assert!(
        max_index <= PACKED_ENTRY_LIMIT,
        "Milnor basis degree {max_degree} needs {max_index} entries, but the packed coefficient format stores {PACKED_ENTRY_LIMIT}"
    );
    let mut counts = vec![0_usize; max_degree + 1];
    counts[0] = 1;
    for index in 1..=max_index {
        let w = weight(index);
        for degree in w..=max_degree {
            counts[degree] = counts[degree]
                .checked_add(counts[degree - w])
                .expect("packed Milnor basis count overflowed usize");
        }
    }
    counts
}

// Provenance: local count wrapper around the Milnor-basis enumeration
// referenced against sseq `MilnorAlgebra::{compute_ppart, generate_basis_2}`.
fn packed_basis_count_of_degree(degree: usize) -> usize {
    packed_basis_counts_through_degree(degree)[degree]
}

// Provenance: shared weighted-partition enumeration with sseq
// `MilnorAlgebra::{compute_ppart, generate_basis_2}`; not a body translation.
fn basis_rec(
    index: usize,
    max_index: usize,
    remaining: usize,
    current: &mut [u32],
    out: &mut Vec<Milnor>,
) {
    if index > max_index {
        if remaining == 0 {
            out.push(Milnor::new(current.to_vec()));
        }
        return;
    }
    let w = weight(index);
    for value in 0..=remaining / w {
        current[index - 1] = value as u32;
        basis_rec(index + 1, max_index, remaining - value * w, current, out);
    }
    current[index - 1] = 0;
}

// Provenance: shared weighted-partition enumeration with sseq
// `MilnorAlgebra::{compute_ppart, generate_basis_2}`; packed output is local.
fn basis_keys_rec(
    index: usize,
    max_index: usize,
    remaining: usize,
    current: &mut [u32; PACKED_ENTRY_LIMIT],
    out: &mut Vec<CoeffKey>,
    checked_pack: bool,
) {
    if index > max_index {
        if remaining == 0 {
            let packed = if checked_pack {
                pack_padded_entries(current).unwrap_or_else(|| {
                    panic!(
                        "Milnor basis element {} cannot be packed",
                        Milnor::new(current.to_vec())
                    )
                })
            } else {
                pack_padded_entries_unchecked(current)
            };
            out.push(packed);
        }
        return;
    }
    let w = weight(index);
    for value in 0..=remaining / w {
        current[index - 1] = value as u32;
        basis_keys_rec(
            index + 1,
            max_index,
            remaining - value * w,
            current,
            out,
            checked_pack,
        );
    }
    current[index - 1] = 0;
}

// Provenance: shared Milnor-basis degree mathematics also used by sseq's basis
// generation; this helper is not an upstream body translation.
fn max_milnor_index(degree: usize) -> usize {
    let mut index = 0;
    while weight(index + 1) <= degree {
        index += 1;
    }
    index
}

pub fn multiply(left: &Milnor, right: &Milnor) -> Vec<Milnor> {
    let mut row_cache = HashMap::default();
    multiply_with_row_cache(left, right, &mut row_cache)
}

pub fn multiply_with_row_cache(
    left: &Milnor,
    right: &Milnor,
    row_cache: &mut HashMap<(u32, usize), RowDecompOptions>,
) -> Vec<Milnor> {
    multiply_packed_with_row_cache(left, right, row_cache)
        .into_iter()
        .map(Milnor::from_packed)
        .collect()
}

pub fn multiply_packed_with_row_cache(
    left: &Milnor,
    right: &Milnor,
    row_cache: &mut HashMap<(u32, usize), RowDecompOptions>,
) -> Vec<CoeffKey> {
    multiply_packed_with_row_cache_matching(left, right, row_cache, |_| true)
}

#[cfg(test)]
// Provenance: local normalization wrapper around the SSeqCpp-derived
// `MulMilnorV3` kernel. See PROVENANCE.md.
pub fn multiply_packed_fast(left: CoeffKey, right: CoeffKey) -> Vec<CoeffKey> {
    let mut out = multiply_packed_fast_raw(left, right);
    sort_packed_mod2(&mut out);
    out
}

// Provenance: local bounded/filtering wrapper around the SSeqCpp-derived
// `MulMilnorV3` kernel. See PROVENANCE.md.
pub fn multiply_packed_fast_bounded_matching<F>(
    left: CoeffKey,
    right: CoeffKey,
    product_degree_bound: usize,
    keep: F,
) -> Vec<CoeffKey>
where
    F: Fn(CoeffKey) -> bool,
{
    let mut out = Vec::new();
    multiply_packed_fast_raw_for_each_bounded_degree(left, right, product_degree_bound, |term| {
        if keep(term) {
            out.push(term);
        }
    });
    sort_packed_mod2(&mut out);
    out
}

#[cfg(test)]
// Provenance: local collection wrapper around the SSeqCpp-derived
// `MulMilnorV3` kernel. See PROVENANCE.md.
pub fn multiply_packed_fast_raw(left: CoeffKey, right: CoeffKey) -> Vec<CoeffKey> {
    if left == 0 {
        return vec![right];
    }
    if right == 0 {
        return vec![left];
    }
    let mut out = Vec::new();
    multiply_packed_fast_raw_for_each(left, right, |term| out.push(term));
    out
}

#[cfg(test)]
// Provenance: local dispatch wrapper around the SSeqCpp-derived
// `MulMilnorV3` kernel. See PROVENANCE.md.
pub fn multiply_packed_fast_raw_for_each(
    left: CoeffKey,
    right: CoeffKey,
    mut emit: impl FnMut(CoeffKey),
) {
    if left == 0 {
        emit(right);
        return;
    }
    if right == 0 {
        emit(left);
        return;
    }
    if packed_entry(left, 8) == 0 && packed_entry(right, 8) == 0 {
        mul_packed_xi_v3_for_each_8(unpack_xi_8(left), unpack_xi_8(right), |xi| {
            emit(pack_xi_8(&xi));
        });
    } else {
        mul_packed_xi_v3_for_each_9(unpack_xi_9(left), unpack_xi_9(right), |xi| {
            emit(pack_xi_9(&xi));
        });
    }
}

// Provenance: local degree-bounded dispatcher for the SSeqCpp-derived
// `MulMilnorV3` kernel. See PROVENANCE.md.
pub fn multiply_packed_fast_raw_for_each_bounded_degree(
    left: CoeffKey,
    right: CoeffKey,
    product_degree_bound: usize,
    mut emit: impl FnMut(CoeffKey),
) {
    if left == 0 {
        emit(right);
        return;
    }
    if right == 0 {
        emit(left);
        return;
    }
    let width = bounded_product_width(product_degree_bound)
        .min(packed_support_width(left).saturating_add(packed_support_width(right)));
    multiply_packed_fast_raw_for_each_width(left, right, width, emit);
}

// Provenance: local width dispatcher for the SSeqCpp-derived `MulMilnorV3`
// kernel. The width-specific functions are generated by the adapted macro.
pub fn multiply_packed_fast_raw_for_each_width(
    left: CoeffKey,
    right: CoeffKey,
    width: usize,
    mut emit: impl FnMut(CoeffKey),
) {
    if left == 0 {
        emit(right);
        return;
    }
    if right == 0 {
        emit(left);
        return;
    }
    match width {
        0 | 1 => mul_packed_xi_v3_for_each_1(unpack_xi_1(left), unpack_xi_1(right), |xi| {
            emit(pack_xi_1(&xi));
        }),
        2 => mul_packed_xi_v3_for_each_2(unpack_xi_2(left), unpack_xi_2(right), |xi| {
            emit(pack_xi_2(&xi));
        }),
        3 => mul_packed_xi_v3_for_each_3(unpack_xi_3(left), unpack_xi_3(right), |xi| {
            emit(pack_xi_3(&xi));
        }),
        4 => mul_packed_xi_v3_for_each_4(unpack_xi_4(left), unpack_xi_4(right), |xi| {
            emit(pack_xi_4(&xi));
        }),
        5 => mul_packed_xi_v3_for_each_5(unpack_xi_5(left), unpack_xi_5(right), |xi| {
            emit(pack_xi_5(&xi));
        }),
        6 => mul_packed_xi_v3_for_each_6(unpack_xi_6(left), unpack_xi_6(right), |xi| {
            emit(pack_xi_6(&xi));
        }),
        7 => mul_packed_xi_v3_for_each_7(unpack_xi_7(left), unpack_xi_7(right), |xi| {
            emit(pack_xi_7(&xi));
        }),
        8 => mul_packed_xi_v3_for_each_8(unpack_xi_8(left), unpack_xi_8(right), |xi| {
            emit(pack_xi_8(&xi));
        }),
        _ => mul_packed_xi_v3_for_each_9(unpack_xi_9(left), unpack_xi_9(right), |xi| {
            emit(pack_xi_9(&xi));
        }),
    }
}

#[inline]
pub(crate) fn packed_support_width(packed: CoeffKey) -> usize {
    for index in (0..PACKED_ENTRY_LIMIT).rev() {
        if packed_entry(packed, index) != 0 {
            return index + 1;
        }
    }
    0
}

#[inline]
pub(crate) fn bounded_product_width(product_degree_bound: usize) -> usize {
    for width in 1..PACKED_ENTRY_LIMIT {
        if product_degree_bound < weight(width + 1) {
            return width;
        }
    }
    PACKED_ENTRY_LIMIT
}

// Provenance: structural adaptation of SSeqCpp `SortMod2(MMilnor1d&)`: sort,
// then cancel equal pairs. This version scans full runs in the local packed
// order, writes odd-parity terms in place, and truncates the vector.
fn sort_packed_mod2(terms: &mut Vec<CoeffKey>) {
    if terms.len() < 2 {
        return;
    }
    terms.sort_unstable_by(|&a, &b| packed_cmp(a, b));
    let mut write = 0;
    let mut read = 0;
    while read < terms.len() {
        let term = terms[read];
        let mut parity = false;
        while read < terms.len() && terms[read] == term {
            parity = !parity;
            read += 1;
        }
        if parity {
            terms[write] = term;
            write += 1;
        }
    }
    terms.truncate(write);
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; this uses the local layout. See PROVENANCE.md.
fn unpack_xi_1(packed: CoeffKey) -> [u32; 1] {
    [(packed & M0) as u32]
}

// Provenance: same packing reference as `unpack_xi_1`; local bit layout.
fn pack_xi_1(xi: &[u32; 1]) -> CoeffKey {
    xi[0] as CoeffKey
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; this uses the local layout. See PROVENANCE.md.
fn unpack_xi_2(packed: CoeffKey) -> [u32; 2] {
    [(packed & M0) as u32, ((packed >> 10) & M1) as u32]
}

// Provenance: same packing reference as `unpack_xi_2`; local bit layout.
fn pack_xi_2(xi: &[u32; 2]) -> CoeffKey {
    (xi[0] as CoeffKey) | ((xi[1] as CoeffKey) << 10)
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; this uses the local layout. See PROVENANCE.md.
fn unpack_xi_3(packed: CoeffKey) -> [u32; 3] {
    [
        (packed & M0) as u32,
        ((packed >> 10) & M1) as u32,
        ((packed >> 18) & M2) as u32,
    ]
}

// Provenance: same packing reference as `unpack_xi_3`; local bit layout.
fn pack_xi_3(xi: &[u32; 3]) -> CoeffKey {
    (xi[0] as CoeffKey) | ((xi[1] as CoeffKey) << 10) | ((xi[2] as CoeffKey) << 18)
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; this uses the local layout. See PROVENANCE.md.
fn unpack_xi_4(packed: CoeffKey) -> [u32; 4] {
    [
        (packed & M0) as u32,
        ((packed >> 10) & M1) as u32,
        ((packed >> 18) & M2) as u32,
        ((packed >> 25) & M3) as u32,
    ]
}

// Provenance: same packing reference as `unpack_xi_4`; local bit layout.
fn pack_xi_4(xi: &[u32; 4]) -> CoeffKey {
    (xi[0] as CoeffKey)
        | ((xi[1] as CoeffKey) << 10)
        | ((xi[2] as CoeffKey) << 18)
        | ((xi[3] as CoeffKey) << 25)
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; this uses the local layout. See PROVENANCE.md.
fn unpack_xi_5(packed: CoeffKey) -> [u32; 5] {
    [
        (packed & M0) as u32,
        ((packed >> 10) & M1) as u32,
        ((packed >> 18) & M2) as u32,
        ((packed >> 25) & M3) as u32,
        ((packed >> 31) & M4) as u32,
    ]
}

// Provenance: same packing reference as `unpack_xi_5`; local bit layout.
fn pack_xi_5(xi: &[u32; 5]) -> CoeffKey {
    (xi[0] as CoeffKey)
        | ((xi[1] as CoeffKey) << 10)
        | ((xi[2] as CoeffKey) << 18)
        | ((xi[3] as CoeffKey) << 25)
        | ((xi[4] as CoeffKey) << 31)
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; this uses the local layout. See PROVENANCE.md.
fn unpack_xi_6(packed: CoeffKey) -> [u32; 6] {
    [
        (packed & M0) as u32,
        ((packed >> 10) & M1) as u32,
        ((packed >> 18) & M2) as u32,
        ((packed >> 25) & M3) as u32,
        ((packed >> 31) & M4) as u32,
        ((packed >> 36) & M5) as u32,
    ]
}

// Provenance: same packing reference as `unpack_xi_6`; local bit layout.
fn pack_xi_6(xi: &[u32; 6]) -> CoeffKey {
    (xi[0] as CoeffKey)
        | ((xi[1] as CoeffKey) << 10)
        | ((xi[2] as CoeffKey) << 18)
        | ((xi[3] as CoeffKey) << 25)
        | ((xi[4] as CoeffKey) << 31)
        | ((xi[5] as CoeffKey) << 36)
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; this uses the local layout. See PROVENANCE.md.
fn unpack_xi_7(packed: CoeffKey) -> [u32; 7] {
    [
        (packed & M0) as u32,
        ((packed >> 10) & M1) as u32,
        ((packed >> 18) & M2) as u32,
        ((packed >> 25) & M3) as u32,
        ((packed >> 31) & M4) as u32,
        ((packed >> 36) & M5) as u32,
        ((packed >> 40) & M6) as u32,
    ]
}

// Provenance: same packing reference as `unpack_xi_7`; local bit layout.
fn pack_xi_7(xi: &[u32; 7]) -> CoeffKey {
    (xi[0] as CoeffKey)
        | ((xi[1] as CoeffKey) << 10)
        | ((xi[2] as CoeffKey) << 18)
        | ((xi[3] as CoeffKey) << 25)
        | ((xi[4] as CoeffKey) << 31)
        | ((xi[5] as CoeffKey) << 36)
        | ((xi[6] as CoeffKey) << 40)
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; this uses the local layout. See PROVENANCE.md.
fn unpack_xi_8(packed: CoeffKey) -> [u32; 8] {
    let mut xi = [0_u32; 8];
    for (i, entry) in xi.iter_mut().enumerate() {
        *entry = packed_entry(packed, i);
    }
    xi
}

// Provenance: same packing reference as `unpack_xi_8`; local bit layout.
fn pack_xi_8(xi: &[u32; 8]) -> CoeffKey {
    (xi[0] as CoeffKey)
        | ((xi[1] as CoeffKey) << 10)
        | ((xi[2] as CoeffKey) << 18)
        | ((xi[3] as CoeffKey) << 25)
        | ((xi[4] as CoeffKey) << 31)
        | ((xi[5] as CoeffKey) << 36)
        | ((xi[6] as CoeffKey) << 40)
        | ((xi[7] as CoeffKey) << 43)
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; this uses the local layout. See PROVENANCE.md.
fn unpack_xi_9(packed: CoeffKey) -> [u32; 9] {
    let mut xi = [0_u32; 9];
    for (i, entry) in xi.iter_mut().enumerate() {
        *entry = packed_entry(packed, i);
    }
    xi
}

// Provenance: same packing reference as `unpack_xi_9`; local bit layout.
fn pack_xi_9(xi: &[u32; 9]) -> CoeffKey {
    debug_assert!(pack_padded_entries(xi).is_some());
    pack_padded_entries_unchecked(xi)
}

#[inline]
// Provenance: directly ported and modified from Weinan Lin's SSeqCpp
// `max_mask` (Apache-2.0). See PROVENANCE.md and THIRD_PARTY_NOTICES.md.
fn max_mask(upper_bound: u32, mask: u32) -> u32 {
    let mut m = upper_bound & mask;
    let mut n = 0;
    while {
        m >>= 1;
        m != 0
    } {
        n |= m;
    }
    (upper_bound | n) & !mask
}

// Provenance: directly ported and modified from Weinan Lin's SSeqCpp
// `MulMilnorV3` (Apache-2.0). The generated Rust functions retain its matrix
// state and traversal; packing, widths, callbacks, and bounds are local.
// See PROVENANCE.md and THIRD_PARTY_NOTICES.md.
macro_rules! define_mul_packed_xi_v3_for_each {
    ($name:ident, $n:expr) => {
        #[inline]
        fn $name(r: [u32; $n], s: [u32; $n], mut emit: impl FnMut([u32; $n])) {
            const N: usize = $n;
            const NCOL: usize = N + 1;
            let mut x = [0u32; NCOL * NCOL];
            let mut xr = [0u32; NCOL * NCOL];
            let mut xs = [0u32; NCOL * NCOL];
            let mut xt = [0u32; NCOL * NCOL];

            let mut r_floor = [0usize; N + 1];
            for row in 1..=N {
                if r[row - 1] != 0 {
                    r_floor[row] = row;
                    xr[row * NCOL + (N - row)] = r[row - 1];
                } else {
                    r_floor[row] = r_floor[row - 1];
                }
            }
            if r_floor[N] == 0 {
                emit(s);
                return;
            }

            let mut r_min = 0;
            while r_min < N {
                let row = r_min;
                r_min += 1;
                if r[row] != 0 {
                    break;
                }
            }
            let r_max = r_floor[N];

            for col in 1..=N - r_min {
                xs[r_floor[N - col] * NCOL + col] = s[col - 1];
            }
            for row in 1..=N {
                xt[r_floor[row] * NCOL + row - r_floor[row]] = 0;
            }
            for col in (N + 1 - r_min)..=N {
                x[col] = s[col - 1];
                let mut row = r_max;
                while row > 0 {
                    if col >= row {
                        xt[row * NCOL + col - row] = s[col - 1];
                        break;
                    }
                    row = r_floor[row - 1];
                }
            }

            let mut i = r_max;
            let mut j = N - i;
            let mut decrease = false;
            loop {
                let mut move_right = false;
                if j != 0 {
                    let index = i * NCOL + j;
                    let index_up = r_floor[i - 1] * NCOL + j;
                    let index_up_right = r_floor[i - 1] * N + j + i;
                    let index_left = index - 1;
                    if i == r_min {
                        if decrease {
                            x[index] = x[index].wrapping_sub(1) & !(xt[index] | x[index_up_right]);
                            decrease = false;
                        } else {
                            x[index] = max_mask(
                                (xr[index] >> j).min(xs[index]),
                                xt[index] | x[index_up_right],
                            );
                        }
                        let x_index = x[index];
                        x[index_up] = xs[index].wrapping_sub(x_index);
                        if j >= r_min && (x[index_up] & xt[index - r_min]) != 0 {
                            if x_index != 0 {
                                decrease = true;
                            } else {
                                move_right = true;
                            }
                        } else {
                            xr[index_left] = xr[index].wrapping_sub(x_index << j);
                            xt[index_up_right] = xt[index] | x_index | x[index_up_right];
                            j -= 1;
                        }
                    } else {
                        if decrease {
                            x[index] = x[index].wrapping_sub(1) & !xt[index];
                            decrease = false;
                        } else {
                            x[index] = max_mask((xr[index] >> j).min(xs[index]), xt[index]);
                        }
                        let x_index = x[index];
                        xr[index_left] = xr[index].wrapping_sub(x_index << j);
                        xs[index_up] = xs[index].wrapping_sub(x_index);
                        xt[index_up_right] = xt[index] | x_index;
                        j -= 1;
                    }
                } else if i == r_min {
                    if (xr[i * NCOL] & x[i]) == 0 {
                        xt[i] = xr[i * NCOL] | x[i];
                        xt[1..i].copy_from_slice(&x[1..i]);
                        let mut xi = [0u32; N];
                        xi[..N].copy_from_slice(&xt[1..=N]);
                        emit(xi);
                    }
                    move_right = true;
                } else {
                    let xt_index = xt[i * NCOL];
                    let xr_index = xr[i * NCOL];
                    if (xt_index & xr_index) != 0 {
                        move_right = true;
                    } else {
                        xt[r_floor[i - 1] * N + i] = xt_index | xr_index;
                        i = r_floor[i - 1];
                        j = N - i;
                    }
                }

                if move_right {
                    loop {
                        if i + j < N {
                            j += 1;
                        } else {
                            loop {
                                i += 1;
                                if i > r_max || r_floor[i] == i {
                                    break;
                                }
                            }
                            if i > r_max {
                                break;
                            }
                            j = 1;
                        }
                        if i > r_max || i + j > N {
                            break;
                        }
                        if x[i * NCOL + j] != 0 {
                            break;
                        }
                    }
                    if i > r_max || i + j > N {
                        break;
                    }
                    decrease = true;
                }
            }
        }
    };
}

define_mul_packed_xi_v3_for_each!(mul_packed_xi_v3_for_each_8, 8);
define_mul_packed_xi_v3_for_each!(mul_packed_xi_v3_for_each_9, 9);
define_mul_packed_xi_v3_for_each!(mul_packed_xi_v3_for_each_1, 1);
define_mul_packed_xi_v3_for_each!(mul_packed_xi_v3_for_each_2, 2);
define_mul_packed_xi_v3_for_each!(mul_packed_xi_v3_for_each_3, 3);
define_mul_packed_xi_v3_for_each!(mul_packed_xi_v3_for_each_4, 4);
define_mul_packed_xi_v3_for_each!(mul_packed_xi_v3_for_each_5, 5);
define_mul_packed_xi_v3_for_each!(mul_packed_xi_v3_for_each_6, 6);
define_mul_packed_xi_v3_for_each!(mul_packed_xi_v3_for_each_7, 7);

pub fn multiply_packed_with_row_cache_matching<F>(
    left: &Milnor,
    right: &Milnor,
    row_cache: &mut HashMap<(u32, usize), RowDecompOptions>,
    keep: F,
) -> Vec<CoeffKey>
where
    F: Fn(CoeffKey) -> bool,
{
    multiply_packed_with_row_cache_internal(left, right, row_cache, keep, None)
}

pub fn multiply_packed_keys_with_row_cache(
    left_key: CoeffKey,
    right_key: CoeffKey,
    row_cache: &mut HashMap<(u32, usize), RowDecompOptions>,
) -> Vec<CoeffKey> {
    multiply_packed_keys_with_row_cache_matching(left_key, right_key, row_cache, |_| true)
}

pub fn multiply_packed_keys_with_row_cache_matching<F>(
    left_key: CoeffKey,
    right_key: CoeffKey,
    row_cache: &mut HashMap<(u32, usize), RowDecompOptions>,
    keep: F,
) -> Vec<CoeffKey>
where
    F: Fn(CoeffKey) -> bool,
{
    multiply_packed_keys_with_row_cache_internal(left_key, right_key, row_cache, keep, None)
}

#[cfg(test)]
pub fn multiply_packed_btrivial_with_row_cache<F>(
    left: &Milnor,
    right: &Milnor,
    row_cache: &mut HashMap<(u32, usize), RowDecompOptions>,
    profile: &[u32],
    keep: F,
) -> Vec<CoeffKey>
where
    F: Fn(CoeffKey) -> bool,
{
    multiply_packed_with_row_cache_internal(left, right, row_cache, keep, Some(profile))
}

#[cfg(test)]
pub fn multiply_packed_btrivial_keys_with_row_cache<F>(
    left_key: CoeffKey,
    right_key: CoeffKey,
    row_cache: &mut HashMap<(u32, usize), RowDecompOptions>,
    profile: &[u32],
    keep: F,
) -> Vec<CoeffKey>
where
    F: Fn(CoeffKey) -> bool,
{
    multiply_packed_keys_with_row_cache_internal(
        left_key,
        right_key,
        row_cache,
        keep,
        Some(profile),
    )
}

fn multiply_packed_with_row_cache_internal<F>(
    left: &Milnor,
    right: &Milnor,
    row_cache: &mut HashMap<(u32, usize), RowDecompOptions>,
    keep: F,
    btrivial_profile: Option<&[u32]>,
) -> Vec<CoeffKey>
where
    F: Fn(CoeffKey) -> bool,
{
    multiply_packed_entries_with_row_cache_internal(
        left.entries(),
        right.entries(),
        row_cache,
        keep,
        btrivial_profile,
    )
}

fn multiply_packed_keys_with_row_cache_internal<F>(
    left_key: CoeffKey,
    right_key: CoeffKey,
    row_cache: &mut HashMap<(u32, usize), RowDecompOptions>,
    keep: F,
    btrivial_profile: Option<&[u32]>,
) -> Vec<CoeffKey>
where
    F: Fn(CoeffKey) -> bool,
{
    if left_key == 0 {
        return keep(right_key)
            .then_some(vec![right_key])
            .unwrap_or_default();
    }
    if right_key == 0 {
        return keep(left_key).then_some(vec![left_key]).unwrap_or_default();
    }
    let (left_entries, left_len) = unpack_packed_entries_trimmed(left_key);
    let (right_entries, right_len) = unpack_packed_entries_trimmed(right_key);
    multiply_packed_entries_with_row_cache_internal(
        &left_entries[..left_len],
        &right_entries[..right_len],
        row_cache,
        keep,
        btrivial_profile,
    )
}

// Provenance: structural adaptation of sseq `PPartMultiplier` and shared
// Milnor-product mathematics. This version caches weighted row decompositions,
// uses recursive traversal and local packed output, and filters emitted terms.
fn multiply_packed_entries_with_row_cache_internal<F>(
    left_entries: &[u32],
    right_entries: &[u32],
    row_cache: &mut HashMap<(u32, usize), RowDecompOptions>,
    keep: F,
    btrivial_profile: Option<&[u32]>,
) -> Vec<CoeffKey>
where
    F: Fn(CoeffKey) -> bool,
{
    if left_entries.is_empty() {
        return pack_entries(right_entries)
            .filter(|&packed| keep(packed))
            .map(|packed| vec![packed])
            .unwrap_or_default();
    }
    if right_entries.is_empty() {
        return pack_entries(left_entries)
            .filter(|&packed| keep(packed))
            .map(|packed| vec![packed])
            .unwrap_or_default();
    }

    let max_i = left_entries.len();
    let max_j = max_column_index_entries(left_entries, right_entries);
    for &r in left_entries {
        row_cache
            .entry((r, max_j))
            .or_insert_with(|| row_decompositions(r, max_j));
    }
    debug_assert!(max_i <= PACKED_ENTRY_LIMIT);
    debug_assert!(max_j < PRODUCT_COLUMN_LIMIT);
    debug_assert!(max_i + max_j < PRODUCT_DIAG_LIMIT);
    let mut row_options = [None; PACKED_ENTRY_LIMIT];
    for (index, &r) in left_entries.iter().enumerate() {
        row_options[index] = Some(row_cache.get(&(r, max_j)).unwrap());
    }
    let mut col_sums_storage = [0_u32; PRODUCT_COLUMN_LIMIT];
    let mut diag_bits_storage = [0_u32; PRODUCT_DIAG_LIMIT];
    let mut diag_sums_storage = [0_u32; PRODUCT_DIAG_LIMIT];
    let col_sums = &mut col_sums_storage[..=max_j];
    let diag_len = max_i + max_j + 1;
    let diag_bits = &mut diag_bits_storage[..diag_len];
    let diag_sums = &mut diag_sums_storage[..diag_len];
    let mut terms = HashSet::default();

    multiply_rec(
        1,
        right_entries,
        &row_options[..max_i],
        col_sums,
        diag_bits,
        diag_sums,
        &mut terms,
        &keep,
        btrivial_profile,
    );

    let mut out = terms.into_iter().collect::<Vec<_>>();
    out.sort_unstable_by(|&a, &b| packed_cmp(a, b));
    out
}

// Keeping the recursion state explicit avoids allocating a context object in
// this product hot path.
// Provenance: structural adaptation of sseq
// `PPartMultiplier::{next_val, update, next}` and shared Milnor-product
// mathematics. The explicit recursive state and hash-set parity are local.
#[allow(clippy::too_many_arguments)]
fn multiply_rec<F>(
    row: usize,
    right_entries: &[u32],
    row_options: &[Option<&RowDecompOptions>],
    col_sums: &mut [u32],
    diag_bits: &mut [u32],
    diag_sums: &mut [u32],
    terms: &mut HashSet<CoeffKey>,
    keep: &F,
    btrivial_profile: Option<&[u32]>,
) where
    F: Fn(CoeffKey) -> bool,
{
    if row > row_options.len() {
        match leaf_term_packed(right_entries, col_sums, diag_bits, diag_sums)
            .filter(|&term| keep(term))
        {
            Some(term) if !terms.insert(term) => {
                terms.remove(&term);
            }
            _ => {}
        }
        return;
    }

    let options = row_options[row - 1].expect("row options were populated before recursion");
    for option in options.rows() {
        if btrivial_profile.is_some_and(|profile| !is_btrivial_option(row, option, profile)) {
            continue;
        }
        if option
            .iter()
            .enumerate()
            .skip(1)
            .any(|(j, &value)| col_sums[j] + value > right_entries.get(j - 1).copied().unwrap_or(0))
        {
            continue;
        }
        if option
            .iter()
            .enumerate()
            .any(|(j, &value)| value != 0 && (diag_bits[row + j] & value) != 0)
        {
            continue;
        }

        for (j, &value) in option.iter().enumerate() {
            if value != 0 {
                diag_bits[row + j] |= value;
                diag_sums[row + j] += value;
            }
        }
        for (j, &value) in option.iter().enumerate().skip(1) {
            col_sums[j] += value;
        }
        multiply_rec(
            row + 1,
            right_entries,
            row_options,
            col_sums,
            diag_bits,
            diag_sums,
            terms,
            keep,
            btrivial_profile,
        );
        for (j, &value) in option.iter().enumerate().skip(1) {
            col_sums[j] -= value;
        }
        for (j, &value) in option.iter().enumerate() {
            if value != 0 {
                diag_bits[row + j] ^= value;
                diag_sums[row + j] -= value;
            }
        }
    }
}

fn is_btrivial_option(row: usize, option: &[u32], profile: &[u32]) -> bool {
    let p = profile.get(row - 1).copied().unwrap_or(0);
    for (j, &value) in option.iter().enumerate() {
        if value == 0 || j as u32 > p {
            continue;
        }
        let exponent = p - j as u32;
        if exponent >= u32::BITS {
            return false;
        }
        let modulus = 1_u32 << exponent;
        if value % modulus != 0 {
            return false;
        }
    }
    true
}

fn max_column_index_entries(left_entries: &[u32], right_entries: &[u32]) -> usize {
    let from_right = right_entries.len();
    let from_left = left_entries
        .iter()
        .copied()
        .max()
        .map(floor_log2)
        .unwrap_or(0);
    from_right.max(from_left)
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; trimming and bit positions are local.
fn unpack_packed_entries_trimmed(packed: CoeffKey) -> ([u32; PACKED_ENTRY_LIMIT], usize) {
    let mut entries = [0_u32; PACKED_ENTRY_LIMIT];
    let mut len = 0;
    for (index, entry) in entries.iter_mut().enumerate() {
        let value = packed_entry(packed, index);
        *entry = value;
        if value != 0 {
            len = index + 1;
        }
    }
    (entries, len)
}

fn floor_log2(value: u32) -> usize {
    if value == 0 {
        0
    } else {
        u32::BITS as usize - 1 - value.leading_zeros() as usize
    }
}

// Provenance: structural adaptation of the admissible-row enumeration in sseq
// `PPartMultiplier`; precomputation and the compact cached rows are local.
fn row_decompositions(value: u32, max_j: usize) -> RowDecompOptions {
    debug_assert!(max_j < PRODUCT_COLUMN_LIMIT);
    let width = max_j + 1;
    let mut current = [0_u32; PRODUCT_COLUMN_LIMIT];
    let mut out = Vec::new();
    row_decomp_rec(value, max_j, width, &mut current[..width], &mut out);
    RowDecompOptions::new(width, out)
}

// Provenance: structural adaptation of the weighted row enumeration in sseq
// `PPartMultiplier`; this descending recursive enumerator is local.
fn row_decomp_rec(remaining: u32, j: usize, width: usize, current: &mut [u32], out: &mut Vec<u32>) {
    let place = 1_u32 << j;
    for value in 0..=remaining / place {
        current[j] = value;
        let next_remaining = remaining - value * place;
        if j == 0 {
            if next_remaining == 0 {
                out.extend_from_slice(&current[..width]);
            }
        } else {
            row_decomp_rec(next_remaining, j - 1, width, current, out);
        }
    }
    current[j] = 0;
}

// Provenance: structural adaptation of sseq `PPartMultiplier`'s diagonal
// overlap test and output-exponent construction. Packed fields, bounds, and
// error handling are local.
fn leaf_term_packed(
    right_entries: &[u32],
    col_sums: &[u32],
    diag_bits: &[u32],
    diag_sums: &[u32],
) -> Option<CoeffKey> {
    let max_j = col_sums.len() - 1;
    let mut entries = [0_u32; PACKED_ENTRY_LIMIT];
    for k in 1..diag_sums.len() {
        let bits = diag_bits[k];
        let mut sum = diag_sums[k];
        if k <= max_j {
            let target = right_entries.get(k - 1).copied().unwrap_or(0);
            let used = col_sums[k];
            if used > target {
                return None;
            }
            let row_zero = target - used;
            if row_zero != 0 {
                if bits & row_zero != 0 {
                    return None;
                }
                sum += row_zero;
            }
        }
        if sum == 0 {
            continue;
        }
        let entry_index = k - 1;
        if entry_index >= PACKED_ENTRY_LIMIT || sum as CoeffKey > packed_entry_mask(entry_index) {
            return None;
        }
        entries[entry_index] = sum;
    }
    Some(pack_padded_entries_unchecked(&entries))
}

// Provenance: packing implementation reference to SSeqCpp `MMilnor::Xi`/`ToXi`
// and sseq `MilnorHashMap::code`; this decoder uses the local bit layout.
fn milnor_from_packed(packed: CoeffKey) -> Milnor {
    let mut len = PACKED_ENTRY_LIMIT;
    while len > 0 && packed_entry(packed, len - 1) == 0 {
        len -= 1;
    }

    let mut entries = Vec::with_capacity(len);
    for index in 0..len {
        entries.push(packed_entry(packed, index));
    }
    Milnor { entries }
}

pub fn packed_cmp(a: CoeffKey, b: CoeffKey) -> Ordering {
    packed_degree(a)
        .cmp(&packed_degree(b))
        .then_with(|| packed_entries_cmp(a, b))
}

fn packed_degree(packed: CoeffKey) -> usize {
    (packed & M0) as usize
        + (((packed >> 10) & M1) as usize) * 3
        + (((packed >> 18) & M2) as usize) * 7
        + (((packed >> 25) & M3) as usize) * 15
        + (((packed >> 31) & M4) as usize) * 31
        + (((packed >> 36) & M5) as usize) * 63
        + (((packed >> 40) & M6) as usize) * 127
        + (((packed >> 43) & M7) as usize) * 255
        + (((packed >> 45) & M8) as usize) * 511
}

fn packed_entries_cmp(a: CoeffKey, b: CoeffKey) -> Ordering {
    macro_rules! cmp_entry {
        ($shift:expr, $mask:expr) => {{
            let ordering = ((a >> $shift) & $mask).cmp(&((b >> $shift) & $mask));
            if !matches!(ordering, Ordering::Equal) {
                return ordering;
            }
        }};
    }
    cmp_entry!(0, M0);
    cmp_entry!(10, M1);
    cmp_entry!(18, M2);
    cmp_entry!(25, M3);
    cmp_entry!(31, M4);
    cmp_entry!(36, M5);
    cmp_entry!(40, M6);
    cmp_entry!(43, M7);
    cmp_entry!(45, M8);
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn m(s: &str) -> Milnor {
        Milnor::parse(s).unwrap()
    }

    #[test]
    fn basic_products() {
        assert!(multiply(&m("1"), &m("1")).is_empty());
        assert_eq!(multiply(&m("1"), &m("2")), vec![m("3")]);
        assert_eq!(multiply(&m("2"), &m("1")), vec![m("0,1"), m("3")]);
    }

    #[test]
    fn fast_packed_products_match_row_decomposition_in_small_degrees() {
        let basis = basis_through_degree(18)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let mut row_cache = HashMap::default();
        for left in &basis {
            for right in &basis {
                if left.degree() + right.degree() > 18 {
                    continue;
                }
                let left_key = left.packed().unwrap();
                let right_key = right.packed().unwrap();
                let expected = multiply_packed_with_row_cache(left, right, &mut row_cache);
                let mut actual = multiply_packed_fast(left_key, right_key);
                actual.sort_unstable_by(|&a, &b| packed_cmp(a, b));
                assert_eq!(actual, expected, "{left} * {right}");
            }
        }
    }

    #[test]
    fn bounded_degree_fast_products_match_generic_through_t150() {
        let basis = basis_keys_through_degree(150);

        fn assert_bounded_matches_generic(
            left_key: CoeffKey,
            right_key: CoeffKey,
            product_degree: usize,
        ) {
            let mut expected = multiply_packed_fast(left_key, right_key);
            let mut actual = Vec::new();
            multiply_packed_fast_raw_for_each_bounded_degree(
                left_key,
                right_key,
                product_degree,
                |term| actual.push(term),
            );
            sort_packed_mod2(&mut actual);
            expected.sort_unstable_by(|&a, &b| packed_cmp(a, b));
            assert_eq!(
                actual, expected,
                "bounded product mismatch in product degree {product_degree}"
            );
        }

        fn assert_sum_width_matches_generic(
            left_key: CoeffKey,
            right_key: CoeffKey,
            product_degree: usize,
        ) {
            let mut expected = multiply_packed_fast(left_key, right_key);
            let width = bounded_product_width(product_degree).min(
                packed_support_width(left_key).saturating_add(packed_support_width(right_key)),
            );
            let mut actual = Vec::new();
            multiply_packed_fast_raw_for_each_width(left_key, right_key, width, |term| {
                actual.push(term)
            });
            sort_packed_mod2(&mut actual);
            expected.sort_unstable_by(|&a, &b| packed_cmp(a, b));
            assert_eq!(
                actual, expected,
                "sum-width product mismatch in product degree {product_degree} using width {width}"
            );
        }

        fn sample_keys(keys: &[CoeffKey]) -> Vec<CoeffKey> {
            if keys.is_empty() {
                return Vec::new();
            }
            let mut indices = vec![
                0,
                keys.len() / 4,
                keys.len() / 2,
                keys.len() * 3 / 4,
                keys.len() - 1,
            ];
            indices.sort_unstable();
            indices.dedup();
            indices.into_iter().map(|index| keys[index]).collect()
        }

        for left_degree in 0..=30 {
            for right_degree in 0..=30 - left_degree {
                for &left_key in &basis[left_degree] {
                    for &right_key in &basis[right_degree] {
                        assert_bounded_matches_generic(
                            left_key,
                            right_key,
                            left_degree + right_degree,
                        );
                        assert_sum_width_matches_generic(
                            left_key,
                            right_key,
                            left_degree + right_degree,
                        );
                    }
                }
            }
        }

        for product_degree in [63, 64, 126, 127, 128, 150] {
            for left_degree in 0..=product_degree {
                let right_degree = product_degree - left_degree;
                for left_key in sample_keys(&basis[left_degree]) {
                    for right_key in sample_keys(&basis[right_degree]) {
                        assert_bounded_matches_generic(left_key, right_key, product_degree);
                        assert_sum_width_matches_generic(left_key, right_key, product_degree);
                    }
                }
            }
        }
    }

    #[test]
    fn fast_bounded_profile_zero_products_match_btrivial_row_path() {
        fn profile_signature_zero(packed: CoeffKey, profile: &[u32]) -> bool {
            profile.iter().enumerate().all(|(index, &entry)| {
                if entry == 0 {
                    return true;
                }
                if entry >= u32::BITS {
                    packed_entry(packed, index) == 0
                } else {
                    let mask = (1_u32 << entry) - 1;
                    packed_entry(packed, index) & mask == 0
                }
            })
        }

        let basis = basis_through_degree(20)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let profiles = [
            vec![3, 2, 1],
            vec![4, 3, 2, 1],
            vec![3, 2, 1, 1],
            vec![3, 2, 2, 1],
            vec![3, 3, 2, 1],
        ];

        for profile in profiles {
            let mut row_cache = HashMap::default();
            for left in &basis {
                for right in &basis {
                    let product_degree = left.degree() + right.degree();
                    if product_degree > 20 {
                        continue;
                    }
                    let left_key = left.packed().unwrap();
                    let right_key = right.packed().unwrap();
                    let expected = multiply_packed_btrivial_keys_with_row_cache(
                        left_key,
                        right_key,
                        &mut row_cache,
                        &profile,
                        |packed| profile_signature_zero(packed, &profile),
                    );
                    let actual = multiply_packed_fast_bounded_matching(
                        left_key,
                        right_key,
                        product_degree,
                        |packed| profile_signature_zero(packed, &profile),
                    );
                    assert_eq!(actual, expected, "profile {profile:?}: {left} * {right}");
                }
            }
        }
    }

    #[test]
    fn packed_key_row_cache_products_match_milnor_entry_path() {
        let basis = basis_through_degree(14)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let profile = vec![3, 2, 1, 0, 0, 0, 0, 0, 0];
        let mut expected_row_cache = HashMap::default();
        let mut actual_row_cache = HashMap::default();
        let mut expected_btrivial_row_cache = HashMap::default();
        let mut actual_btrivial_row_cache = HashMap::default();
        for left in &basis {
            for right in &basis {
                if left.degree() + right.degree() > 14 {
                    continue;
                }
                let left_key = left.packed().unwrap();
                let right_key = right.packed().unwrap();
                let expected = multiply_packed_with_row_cache(left, right, &mut expected_row_cache);
                let actual =
                    multiply_packed_keys_with_row_cache(left_key, right_key, &mut actual_row_cache);
                assert_eq!(actual, expected, "packed key path for {left} * {right}");

                let expected_btrivial = multiply_packed_btrivial_with_row_cache(
                    left,
                    right,
                    &mut expected_btrivial_row_cache,
                    &profile,
                    |packed| packed_entry(packed, 0) & 1 == 0,
                );
                let actual_btrivial = multiply_packed_btrivial_keys_with_row_cache(
                    left_key,
                    right_key,
                    &mut actual_btrivial_row_cache,
                    &profile,
                    |packed| packed_entry(packed, 0) & 1 == 0,
                );
                assert_eq!(
                    actual_btrivial, expected_btrivial,
                    "packed key B-trivial path for {left} * {right}"
                );
            }
        }
    }

    #[test]
    fn packed_basis_keys_match_milnor_basis_in_small_degrees() {
        for degree in 0..=18 {
            let from_keys = basis_keys_of_degree(degree)
                .into_iter()
                .map(Milnor::from_packed)
                .collect::<Vec<_>>();
            assert_eq!(from_keys, basis_of_degree(degree), "degree {degree}");
        }
    }

    #[test]
    fn packed_coefficients_cover_degree_512_boundary() {
        for coeff in [m("512"), m("0,0,0,0,0,0,0,0,1"), m("511,0,0,0,0,0,0,0,0")] {
            let packed = coeff
                .packed()
                .unwrap_or_else(|| panic!("{coeff} should pack for t <= 512"));
            assert_eq!(Milnor::from_packed(packed), coeff);
        }
    }

    #[test]
    fn packed_basis_keys_cover_degree_512_boundary() {
        let basis_511 = basis_keys_of_degree(511);
        let xi_9 = m("0,0,0,0,0,0,0,0,1").packed().unwrap();
        assert!(
            basis_511
                .binary_search_by(|&key| packed_cmp(key, xi_9))
                .is_ok(),
            "Sq(0,0,0,0,0,0,0,0,1) should appear in the packed degree-511 basis"
        );

        let basis_512 = basis_keys_of_degree(512);
        let sq_512 = m("512").packed().unwrap();
        assert!(
            basis_512
                .binary_search_by(|&key| packed_cmp(key, sq_512))
                .is_ok(),
            "Sq(512) should appear in the packed degree-512 basis"
        );
    }

    #[test]
    fn packed_basis_through_degree_smoke() {
        // Exercise the cumulative allocator at a size suitable for the default
        // test suite. The separate boundary test above covers degrees 511/512
        // without retaining every lower-degree basis at once.
        let basis = basis_keys_through_degree(64);
        assert_eq!(basis.len(), 65);

        let xi_6 = m("0,0,0,0,0,1").packed().unwrap();
        assert!(
            basis[63]
                .binary_search_by(|&key| packed_cmp(key, xi_6))
                .is_ok(),
            "Sq(0,0,0,0,0,1) should appear in degree 63"
        );

        let sq_64 = m("64").packed().unwrap();
        assert!(
            basis[64]
                .binary_search_by(|&key| packed_cmp(key, sq_64))
                .is_ok(),
            "Sq(64) should appear in degree 64"
        );
    }

    #[test]
    fn fast_packed_products_match_row_decomposition_at_512_boundary() {
        let pairs = [
            (m("512"), m("1")),
            (m("1"), m("512")),
            (m("0,0,0,0,0,0,0,0,1"), m("1")),
        ];
        let mut row_cache = HashMap::default();
        for (left, right) in pairs {
            let left_key = left.packed().unwrap();
            let right_key = right.packed().unwrap();
            let expected = multiply_packed_with_row_cache(&left, &right, &mut row_cache);
            let mut actual = multiply_packed_fast(left_key, right_key);
            actual.sort_unstable_by(|&a, &b| packed_cmp(a, b));
            assert_eq!(actual, expected, "{left} * {right}");
        }
    }

    #[test]
    fn multiplication_is_associative_in_small_degrees() {
        let basis = basis_through_degree(5)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        for a in &basis {
            for b in &basis {
                for c in &basis {
                    let left = multiply_sums(&multiply(a, b), std::slice::from_ref(c));
                    let right = multiply_sums(std::slice::from_ref(a), &multiply(b, c));
                    assert_eq!(left, right, "({a} * {b}) * {c} != {a} * ({b} * {c})");
                }
            }
        }
    }

    fn multiply_sums(left: &[Milnor], right: &[Milnor]) -> Vec<Milnor> {
        let mut terms = BTreeSet::new();
        for l in left {
            for r in right {
                for term in multiply(l, r) {
                    if !terms.insert(term.clone()) {
                        terms.remove(&term);
                    }
                }
            }
        }
        terms.into_iter().collect()
    }
}
