//! Deterministic end-to-end and allocation qualification for multiplication changes.
use bitz::{
    piop::spartan::{
        MulLayout, MulWitness,
        protocol::{self, PreparedRelation},
    },
    transcript::Blake3Transcript,
};
use std::{hint::black_box, time::Instant};
#[cfg(feature = "bench-peak-memory")]
#[path = "../benches/common/peak_memory.rs"]
mod memory;
#[cfg(feature = "bench-peak-memory")]
#[global_allocator]
static ALLOCATOR: memory::PeakAlloc = memory::PeakAlloc;
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let kind = args.get(1).map(String::as_str).unwrap_or("u32");
    let log: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(15);
    let reps: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);
    let n = std::env::var("BITZ_QUALIFICATION_N")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(1usize << log);
    let witness_only = std::env::var_os("BITZ_QUALIFICATION_WITNESS_ONLY").is_some();
    let width: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1);
    let prefix_only = std::env::var_os("BITZ_QUALIFICATION_PREFIX_ONLY").is_some();
    let pack_only = std::env::var_os("BITZ_QUALIFICATION_PACK_ONLY").is_some();
    macro_rules! run {($layout:expr,$build:expr)=>{{
  let prepared=(!witness_only).then(|| PreparedRelation::new($layout).unwrap());
  let mut reusable_rows = None;
  for rep in 0..=reps {
   #[cfg(feature="bench-peak-memory")] memory::reset_peak();
   let start=Instant::now(); let witness=$build; let witness_ms=start.elapsed().as_secs_f64()*1000.;
   if witness_only {
    #[cfg(feature="bench-peak-memory")] let peak=memory::peak_bytes();
    #[cfg(not(feature="bench-peak-memory"))] let peak=0usize;
    println!("{}",serde_json::json!({"kind":kind,"width":width,"log":log,"n":n,"rep":rep,"witness_ms":witness_ms,"peak_live_bytes":peak}));
    black_box(witness); continue;
   }
   let prepared=prepared.as_ref().unwrap();
   if prefix_only {
    let binding=[42;32];
    let start=Instant::now();
    let proved=protocol::prove_piop(&mut Blake3Transcript::new(),prepared.prefix(),&witness,&binding).unwrap();
    let piop_ms=start.elapsed().as_secs_f64()*1000.;
    let start=Instant::now();
    let verified=protocol::verify_piop(&mut Blake3Transcript::new(),prepared.prefix(),&binding,&proved.messages).unwrap();
    let verify_ms=start.elapsed().as_secs_f64()*1000.;
    assert_eq!(proved.terminal_claim,verified.terminal_claim);
    let digest=blake3::hash(format!("{:?}",proved.messages).as_bytes()).to_hex().to_string();
    println!("{}",serde_json::json!({"kind":kind,"width":width,"log":log,"n":n,"rep":rep,"witness_ms":witness_ms,"piop_ms":piop_ms,"verify_ms":verify_ms,"digest":digest}));
    black_box(proved);continue;
   }
   if pack_only && std::env::var_os("BITZ_QUALIFICATION_REUSE_ROWS").is_some() {
    let rows=reusable_rows.get_or_insert_with(|| witness.bitz_bit_rows());
    let start=Instant::now();witness.write_bitz_bit_rows(rows).unwrap();let pack_ms=start.elapsed().as_secs_f64()*1000.;
    println!("{}",serde_json::json!({"kind":kind,"width":width,"log":log,"n":n,"rep":rep,"witness_ms":witness_ms,"pack_ms":pack_ms,"reused_rows":true}));
    black_box(rows);continue;
   }
   let start=Instant::now(); let rows=witness.bitz_bit_rows(); let pack_ms=start.elapsed().as_secs_f64()*1000.;
   if pack_only {
    println!("{}",serde_json::json!({"kind":kind,"width":width,"log":log,"n":n,"rep":rep,"witness_ms":witness_ms,"pack_ms":pack_ms}));
    black_box(rows);continue;
   }
   let start=Instant::now(); let hint=protocol::commit(&prepared,rows).unwrap(); let commit_ms=start.elapsed().as_secs_f64()*1000.;
   let start=Instant::now(); let proof=protocol::prove(&mut Blake3Transcript::new(),&prepared,&witness,&hint).unwrap(); let prove_ms=start.elapsed().as_secs_f64()*1000.;
   let start=Instant::now(); protocol::verify(&mut Blake3Transcript::new(),&prepared,&hint.commitment,&proof).unwrap(); let verify_ms=start.elapsed().as_secs_f64()*1000.;
   #[cfg(feature="bench-peak-memory")] let peak=memory::peak_bytes();
   #[cfg(not(feature="bench-peak-memory"))] let peak=0usize;
   let mut h=blake3::Hasher::new();h.update(&hint.commitment.root);h.update(&proof.bitz().to_bytes());h.update(format!("{:?}",proof.prefix()).as_bytes());
   println!("{}",serde_json::json!({"kind":kind,"width":width,"log":log,"n":n,"rep":rep,"witness_ms":witness_ms,"witness_to_proof_ms":witness_ms+pack_ms+commit_ms+prove_ms,"pack_ms":pack_ms,"commit_ms":commit_ms,"prove_ms":prove_ms,"verify_ms":verify_ms,"online_ms":pack_ms+commit_ms+prove_ms,"peak_live_bytes":peak,"proof_bytes":proof.size_bytes(prepared.security()),"digest":h.finalize().to_hex().to_string()}));
   black_box(proof);
  }
 }}}
    match kind {
        "u32" => run!(
            MulLayout::<u32>::new_with_word_bits(n, width).unwrap(),
            MulWitness::<u32>::from_fn_with_word_bits(n, width, |i| (
                mix(i as u64) as u32,
                mix(i as u64 ^ 42) as u32
            ))
            .unwrap()
        ),
        "u64" => run!(
            MulLayout::<u64>::new_with_word_bits(n, width).unwrap(),
            MulWitness::<u64>::from_fn_with_word_bits(n, width, |i| (
                mix(i as u64),
                mix(i as u64 ^ 42)
            ))
            .unwrap()
        ),
        "u128" => run!(
            MulLayout::<u128>::new_with_word_bits(n, width).unwrap(),
            MulWitness::<u128>::from_fn_with_word_bits(n, width, |i| (
                (u128::from(mix(i as u64)) << 64) | u128::from(mix(i as u64 ^ 13)),
                (u128::from(mix(i as u64 ^ 42)) << 64) | u128::from(mix(i as u64 ^ 17))
            ))
            .unwrap()
        ),
        _ => panic!("expected u32, u64 or u128"),
    }
}
