use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

pub type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
pub type FastHashSet<T> = HashSet<T, BuildHasherDefault<FastHasher>>;

pub fn fast_hash_map_with_capacity<K, V>(capacity: usize) -> FastHashMap<K, V> {
    HashMap::with_capacity_and_hasher(capacity, BuildHasherDefault::<FastHasher>::default())
}

#[derive(Default)]
pub struct FastHasher {
    state: u64,
}

impl FastHasher {
    // Provenance: direct source adaptation of Sebastiano Vigna's public-domain
    // SplitMix64 output mixer. The state combination after `x ^= x >> 31` is
    // specific to this hasher.
    // https://prng.di.unimi.it/splitmix64.c
    fn mix(&mut self, value: u64) {
        let mut x = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        self.state ^= x;
        self.state = self
            .state
            .rotate_left(27)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.mix(u64::from_le_bytes(*chunk));
        }
        if !rest.is_empty() {
            let mut tail = [0_u8; 8];
            tail[..rest.len()].copy_from_slice(rest);
            self.mix(u64::from_le_bytes(tail));
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.mix(i as u64);
    }

    fn write_u16(&mut self, i: u16) {
        self.mix(i as u64);
    }

    fn write_u32(&mut self, i: u32) {
        self.mix(i as u64);
    }

    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }

    fn write_u128(&mut self, i: u128) {
        self.mix(i as u64);
        self.mix((i >> 64) as u64);
    }

    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }
}
