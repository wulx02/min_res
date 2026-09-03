// Portions of this file were structurally adapted or implemented with reference
// to SSeqCpp and SpectralSequences/sseq, then modified for this project. Each
// affected routine is labeled below; see PROVENANCE.md,
// THIRD_PARTY_NOTICES.md, and LICENSE-APACHE.

use crate::memory_probe::log_process_memory;

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct BitVec {
    len: usize,
    words: Vec<u64>,
}

impl BitVec {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            words: vec![0; len.div_ceil(64)],
        }
    }

    pub fn unit(len: usize, index: usize) -> Self {
        let mut out = Self::new(len);
        out.set(index, true);
        out
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn storage_bytes_for_len(len: usize) -> usize {
        len.div_ceil(64) * std::mem::size_of::<u64>()
    }

    pub fn storage_bytes(&self) -> usize {
        self.words.capacity() * std::mem::size_of::<u64>()
    }

    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    #[cfg(test)]
    pub fn get(&self, index: usize) -> bool {
        debug_assert!(index < self.len);
        (self.words[index / 64] & (1_u64 << (index % 64))) != 0
    }

    pub fn set(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.len);
        let mask = 1_u64 << (index % 64);
        if value {
            self.words[index / 64] |= mask;
        } else {
            self.words[index / 64] &= !mask;
        }
    }

    pub fn toggle(&mut self, index: usize) {
        debug_assert!(index < self.len);
        self.words[index / 64] ^= 1_u64 << (index % 64);
    }

    pub fn xor_assign(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a ^= *b;
        }
    }

    pub fn leading_one(&self) -> Option<usize> {
        self.leading_one_from(0)
    }

    pub fn leading_one_from(&self, start: usize) -> Option<usize> {
        if start >= self.len {
            return None;
        }
        let mut word_index = start / 64;
        let mut word = self.words.get(word_index).copied().unwrap_or(0);
        word &= u64::MAX << (start % 64);
        loop {
            if word != 0 {
                let bit = word.trailing_zeros() as usize;
                let index = word_index * 64 + bit;
                if index < self.len {
                    return Some(index);
                }
                return None;
            }
            word_index += 1;
            word = self.words.get(word_index).copied()?;
        }
    }

    pub fn leading_one_from_start(&self, start: &mut usize) -> Option<usize> {
        let pivot = self.leading_one_from(*start)?;
        *start = pivot + 1;
        Some(pivot)
    }

    pub fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        Ones {
            bits: self,
            word_index: 0,
            word: self.words.first().copied().unwrap_or(0),
        }
    }
}

struct Ones<'a> {
    bits: &'a BitVec,
    word_index: usize,
    word: u64,
}

impl Iterator for Ones<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.word != 0 {
                let bit = self.word.trailing_zeros() as usize;
                self.word &= self.word - 1;
                let index = self.word_index * 64 + bit;
                if index < self.bits.len {
                    return Some(index);
                }
            } else {
                self.word_index += 1;
                self.word = self.bits.words.get(self.word_index).copied()?;
            }
        }
    }
}

#[cfg(test)]
pub fn kernel(columns: &[BitVec], target_dim: usize) -> Vec<BitVec> {
    kernel_with_label(columns, target_dim, None)
}

// Provenance: structural adaptation of dependency tracking from SSeqCpp
// `SetLinearMap` and sseq `Matrix::compute_kernel`. See PROVENANCE.md.
fn kernel_with_label(columns: &[BitVec], target_dim: usize, label: Option<&str>) -> Vec<BitVec> {
    let domain_dim = columns.len();
    if domain_dim == 0 {
        return Vec::new();
    }
    if let Some(label) = label {
        log_process_memory(
            "f2_kernel_start",
            format!(
                "label={label} columns={} target_dim={target_dim}",
                columns.len()
            ),
        );
    }
    let mut image = ImageBasis::new(target_dim, domain_dim);
    let mut basis = Vec::new();
    for (column_index, column) in columns.iter().enumerate() {
        debug_assert_eq!(column.len(), target_dim);
        let combo = BitVec::unit(domain_dim, column_index);
        if let Some(relation) = image.insert_or_relation(column.clone(), combo) {
            basis.push(relation);
        }
    }
    if let Some(label) = label {
        log_process_memory(
            "f2_kernel_end",
            format!(
                "label={label} columns={} target_dim={target_dim} relations={} image_basis_mb={:.1} relation_mb={:.1}",
                columns.len(),
                basis.len(),
                bytes_to_mb(image.estimated_bytes()),
                bytes_to_mb(bitvec_slice_bytes(&basis)),
            ),
        );
    }
    basis
}

#[cfg(test)]
// Provenance: structural adaptation of quotient-space construction from
// SSeqCpp `QuotientSpace` and sseq `Subquotient::from_parts`.
pub fn quotient_representatives(
    subspace: &[BitVec],
    mod_out_by: &[BitVec],
    ambient_dim: usize,
) -> Vec<BitVec> {
    let mut basis = XorBasis::new(ambient_dim);
    for vector in mod_out_by {
        basis.insert(vector.clone());
    }

    let mut representatives = Vec::new();
    for vector in subspace {
        if basis.insert(vector.clone()).is_some() {
            representatives.push(vector.clone());
        }
    }
    representatives
}

// Provenance: implementation reference to SSeqCpp `GetImage` and sseq
// `Matrix::apply`; this dense-column F2 body is local.
pub fn apply_columns(columns: &[BitVec], target_dim: usize, vector: &BitVec) -> BitVec {
    debug_assert_eq!(columns.len(), vector.len());
    let mut out = BitVec::new(target_dim);
    for i in vector.ones() {
        out.xor_assign(&columns[i]);
    }
    out
}

// Provenance: structural adaptation of kernel/image homology construction from
// sseq's resolution functions and `Subquotient::from_parts`.
pub fn homology_representatives(
    differential_columns: &[BitVec],
    differential_target_dim: usize,
    boundary_columns: &[BitVec],
    ambient_dim: usize,
) -> Vec<BitVec> {
    homology_representatives_with_label(
        differential_columns,
        differential_target_dim,
        boundary_columns,
        ambient_dim,
        None,
    )
}

// Provenance: structural adaptation of kernel/image homology construction from
// sseq's resolution functions and `Subquotient::from_parts`.
pub fn homology_representatives_with_label(
    differential_columns: &[BitVec],
    differential_target_dim: usize,
    boundary_columns: &[BitVec],
    ambient_dim: usize,
    label: Option<&str>,
) -> Vec<BitVec> {
    let mut batches = homology_representative_batches_with_label(
        differential_columns,
        differential_target_dim,
        boundary_columns,
        ambient_dim,
        label,
    );
    let mut representatives = Vec::with_capacity(batches.len());
    while let Some(mut batch) = batches.next_batch(usize::MAX) {
        representatives.append(&mut batch);
    }
    if let Some(label) = label {
        let representatives_mb = bytes_to_mb(bitvec_slice_bytes(&representatives));
        log_process_memory(
            "f2_homology_end",
            format!(
                "label={label} representatives={} representatives_mb={representatives_mb:.1}",
                representatives.len(),
            ),
        );
    }
    representatives
}

pub struct HomologyRepresentativeBatches {
    quotient_basis: Vec<BitVec>,
    cycles: Vec<BitVec>,
    ambient_dim: usize,
    next_cycle: usize,
}

impl HomologyRepresentativeBatches {
    pub fn len(&self) -> usize {
        self.cycles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cycles.is_empty()
    }

    pub fn next_batch(&mut self, batch_len: usize) -> Option<Vec<BitVec>> {
        if self.next_cycle >= self.cycles.len() {
            return None;
        }
        let batch_len = batch_len.max(1);
        let end = (self.next_cycle + batch_len).min(self.cycles.len());
        let representatives = self.cycles[self.next_cycle..end]
            .iter()
            .map(|cycle| {
                let mut representative = BitVec::new(self.ambient_dim);
                for i in cycle.ones() {
                    representative.xor_assign(&self.quotient_basis[i]);
                }
                representative
            })
            .collect::<Vec<_>>();
        self.next_cycle = end;
        Some(representatives)
    }
}

// Provenance: structural adaptation of sseq `Subquotient::from_parts`,
// `Subspace::reduce`, and `Matrix::compute_kernel`; batching is local.
pub fn homology_representative_batches_with_label(
    differential_columns: &[BitVec],
    differential_target_dim: usize,
    boundary_columns: &[BitVec],
    ambient_dim: usize,
    label: Option<&str>,
) -> HomologyRepresentativeBatches {
    if ambient_dim == 0 {
        return HomologyRepresentativeBatches {
            quotient_basis: Vec::new(),
            cycles: Vec::new(),
            ambient_dim,
            next_cycle: 0,
        };
    }
    if let Some(label) = label {
        log_process_memory(
            "f2_homology_start",
            format!(
                "label={label} diff_cols={} diff_target_dim={} boundary_cols={} ambient_dim={} diff_mb={:.1} boundary_mb={:.1}",
                differential_columns.len(),
                differential_target_dim,
                boundary_columns.len(),
                ambient_dim,
                bytes_to_mb(bitvec_slice_bytes(differential_columns)),
                bytes_to_mb(bitvec_slice_bytes(boundary_columns)),
            ),
        );
    }
    let mut quotient_space = XorBasis::new(ambient_dim);
    for boundary in boundary_columns {
        quotient_space.insert(boundary.clone());
    }
    if let Some(label) = label {
        log_process_memory(
            "f2_homology_after_boundaries",
            format!(
                "label={label} ambient_dim={ambient_dim} boundary_cols={} quotient_space_mb={:.1}",
                boundary_columns.len(),
                bytes_to_mb(quotient_space.estimated_bytes()),
            ),
        );
    }

    let mut quotient_basis = Vec::new();
    for i in 0..ambient_dim {
        if let Some(representative) = quotient_space.insert(BitVec::unit(ambient_dim, i)) {
            quotient_basis.push(representative);
        }
    }
    if quotient_basis.is_empty() {
        return HomologyRepresentativeBatches {
            quotient_basis,
            cycles: Vec::new(),
            ambient_dim,
            next_cycle: 0,
        };
    }
    if let Some(label) = label {
        let quotient_space_mb = bytes_to_mb(quotient_space.estimated_bytes());
        log_process_memory(
            "f2_homology_after_quotient_basis",
            format!(
                "label={label} ambient_dim={ambient_dim} quotient_basis={} quotient_basis_mb={:.1} quotient_space_mb={:.1}",
                quotient_basis.len(),
                bytes_to_mb(bitvec_slice_bytes(&quotient_basis)),
                quotient_space_mb,
            ),
        );
    }
    drop(quotient_space);
    if let Some(label) = label {
        log_process_memory(
            "f2_homology_after_drop_quotient_space",
            format!(
                "label={label} quotient_basis={} quotient_basis_mb={:.1}",
                quotient_basis.len(),
                bytes_to_mb(bitvec_slice_bytes(&quotient_basis)),
            ),
        );
    }

    let quotient_differential = quotient_basis
        .iter()
        .map(|representative| {
            apply_columns(
                differential_columns,
                differential_target_dim,
                representative,
            )
        })
        .collect::<Vec<_>>();
    if let Some(label) = label {
        log_process_memory(
            "f2_homology_after_quotient_differential",
            format!(
                "label={label} quotient_basis={} quotient_differential_mb={:.1}",
                quotient_basis.len(),
                bytes_to_mb(bitvec_slice_bytes(&quotient_differential)),
            ),
        );
    }

    let cycles = kernel_with_label(&quotient_differential, differential_target_dim, label);
    if let Some(label) = label {
        let quotient_differential_mb = bytes_to_mb(bitvec_slice_bytes(&quotient_differential));
        log_process_memory(
            "f2_homology_after_kernel",
            format!(
                "label={label} cycles={} quotient_basis_mb={:.1} quotient_differential_mb={:.1}",
                cycles.len(),
                bytes_to_mb(bitvec_slice_bytes(&quotient_basis)),
                quotient_differential_mb,
            ),
        );
    }
    drop(quotient_differential);
    if let Some(label) = label {
        log_process_memory(
            "f2_homology_after_drop_quotient_differential",
            format!(
                "label={label} cycles={} quotient_basis_mb={:.1}",
                cycles.len(),
                bytes_to_mb(bitvec_slice_bytes(&quotient_basis)),
            ),
        );
    }

    if let Some(label) = label {
        log_process_memory(
            "f2_homology_batch_source_ready",
            format!(
                "label={label} cycles={} quotient_basis_mb={:.1}",
                cycles.len(),
                bytes_to_mb(bitvec_slice_bytes(&quotient_basis)),
            ),
        );
    }

    HomologyRepresentativeBatches {
        quotient_basis,
        cycles,
        ambient_dim,
        next_cycle: 0,
    }
}

struct XorBasis {
    pivots: Vec<Option<BitVec>>,
}

pub struct LinearSolver {
    basis: ImageBasis,
    target_dim: usize,
}

struct ImageBasis {
    pivots: Vec<Option<(BitVec, BitVec)>>,
    domain_dim: usize,
}

impl LinearSolver {
    // Provenance: structural adaptation of SSeqCpp `GetInvMap` and sseq
    // `Matrix::compute_quasi_inverse`; storage and APIs are local.
    pub fn new(columns: &[BitVec], target_dim: usize) -> Self {
        let domain_dim = columns.len();
        let mut basis = ImageBasis::new(target_dim, domain_dim);

        for (column_index, column) in columns.iter().enumerate() {
            debug_assert_eq!(column.len(), target_dim);
            let combo = BitVec::unit(domain_dim, column_index);
            basis.insert(column.clone(), combo);
        }

        Self { basis, target_dim }
    }

    // Provenance: structural adaptation of SSeqCpp `GetImage`/`GetInvImage`
    // and sseq `QuasiInverse::apply`; the dense BitVec solver is local.
    pub fn solve(&self, target: &BitVec) -> Option<BitVec> {
        debug_assert_eq!(target.len(), self.target_dim);
        self.basis.solve(target.clone())
    }

    pub fn domain_dim(&self) -> usize {
        self.basis.domain_dim()
    }

    pub fn target_dim(&self) -> usize {
        self.target_dim
    }

    pub fn pivot_count(&self) -> usize {
        self.basis.pivot_count()
    }

    pub fn estimated_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.basis.estimated_bytes()
    }
}

impl ImageBasis {
    fn new(target_dim: usize, domain_dim: usize) -> Self {
        Self {
            pivots: vec![None; target_dim],
            domain_dim,
        }
    }

    // Provenance: structural adaptation of pivot reduction in SSeqCpp
    // `GetInvMap`/`SetLinearMap*` and sseq `Matrix::row_reduce`.
    fn reduce(&self, mut image: BitVec, mut combo: BitVec) -> (BitVec, BitVec) {
        let mut start = 0;
        while let Some(pivot) = image.leading_one_from_start(&mut start) {
            let Some((pivot_image, pivot_combo)) = &self.pivots[pivot] else {
                break;
            };
            image.xor_assign(pivot_image);
            combo.xor_assign(pivot_combo);
        }
        (image, combo)
    }

    // Provenance: structural adaptation of image/preimage insertion in
    // SSeqCpp `SetLinearMap*` and sseq's quasi-inverse construction.
    fn insert(&mut self, image: BitVec, combo: BitVec) {
        let (image, combo) = self.reduce(image, combo);
        if let Some(pivot) = image.leading_one() {
            self.pivots[pivot] = Some((image, combo));
        }
    }

    // Provenance: structural adaptation of kernel-relation detection in
    // SSeqCpp `SetLinearMap*` and sseq `Matrix::compute_kernel`.
    fn insert_or_relation(&mut self, image: BitVec, combo: BitVec) -> Option<BitVec> {
        let (image, combo) = self.reduce(image, combo);
        if let Some(pivot) = image.leading_one() {
            self.pivots[pivot] = Some((image, combo));
            None
        } else {
            Some(combo)
        }
    }

    // Provenance: structural adaptation of SSeqCpp `GetInvImage` and sseq
    // `QuasiInverse::apply`; the representation is local.
    fn solve(&self, target: BitVec) -> Option<BitVec> {
        let combo = BitVec::new(self.domain_dim);
        let (image, combo) = self.reduce(target, combo);
        image.is_zero().then_some(combo)
    }

    fn domain_dim(&self) -> usize {
        self.domain_dim
    }

    fn pivot_count(&self) -> usize {
        self.pivots.iter().filter(|pivot| pivot.is_some()).count()
    }

    fn estimated_bytes(&self) -> usize {
        let pivot_slots = self.pivots.capacity() * std::mem::size_of::<Option<(BitVec, BitVec)>>();
        let pivot_vectors = self
            .pivots
            .iter()
            .filter_map(Option::as_ref)
            .map(|(image, combo)| image.storage_bytes() + combo.storage_bytes())
            .sum::<usize>();
        std::mem::size_of::<Self>() + pivot_slots + pivot_vectors
    }
}

impl XorBasis {
    fn new(ambient_dim: usize) -> Self {
        Self {
            pivots: vec![None; ambient_dim],
        }
    }

    // Provenance: structural adaptation of SSeqCpp `Residue`/`ResidueInplace`
    // and sseq `Subspace::reduce`; dense BitVec storage is local.
    fn reduce(&self, mut vector: BitVec) -> BitVec {
        let mut start = 0;
        while let Some(pivot) = vector.leading_one_from_start(&mut start) {
            let Some(row) = &self.pivots[pivot] else {
                break;
            };
            vector.xor_assign(row);
        }
        vector
    }

    // Provenance: structural adaptation of SSeqCpp `AddToSpace`/`GetSpace`
    // and implementation reference to sseq `Subspace::add_vector`; unlike
    // sseq, this incrementally installs one reduced pivot. See PROVENANCE.md.
    fn insert(&mut self, vector: BitVec) -> Option<BitVec> {
        let reduced = self.reduce(vector);
        let pivot = reduced.leading_one()?;
        self.pivots[pivot] = Some(reduced.clone());
        Some(reduced)
    }

    fn estimated_bytes(&self) -> usize {
        let pivot_slots = self.pivots.capacity() * std::mem::size_of::<Option<BitVec>>();
        let pivot_vectors = self
            .pivots
            .iter()
            .filter_map(Option::as_ref)
            .map(BitVec::storage_bytes)
            .sum::<usize>();
        std::mem::size_of::<Self>() + pivot_slots + pivot_vectors
    }
}

fn bitvec_slice_bytes(vectors: &[BitVec]) -> usize {
    std::mem::size_of_val(vectors) + vectors.iter().map(BitVec::storage_bytes).sum::<usize>()
}

fn bytes_to_mb(bytes: usize) -> f64 {
    bytes as f64 / 1_048_576.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_of_single_row() {
        let mut c0 = BitVec::new(1);
        c0.set(0, true);
        let result = kernel(&[c0], 1);
        assert!(result.is_empty());
    }

    #[test]
    fn quotient_mods_out_image() {
        let a = BitVec::unit(3, 0);
        let b = BitVec::unit(3, 1);
        let mut ab = a.clone();
        ab.xor_assign(&b);
        let reps = quotient_representatives(&[a.clone(), b, ab], &[a], 3);
        assert_eq!(reps.len(), 1);
    }

    #[test]
    fn solves_linear_system() {
        let c0 = BitVec::unit(2, 0);
        let c1 = BitVec::unit(2, 1);
        let mut target = c0.clone();
        target.xor_assign(&c1);
        let solver = LinearSolver::new(&[c0, c1], 2);
        let solution = solver.solve(&target).unwrap();
        assert!(solution.get(0));
        assert!(solution.get(1));
    }
}
