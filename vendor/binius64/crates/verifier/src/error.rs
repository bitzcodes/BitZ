// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use binius_core::ConstraintSystemError;
use binius_iop::channel::Error as IOPChannelError;
use binius_ip::{channel::Error as ChannelError, sumcheck};

use crate::{
	fri,
	protocols::{intmul, shift},
	ring_switch,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("require 1..=128 FRI query security bits and positive log inverse rate")]
	InvalidSecurityParameters,
	#[error("transcript error: {0}")]
	Transcript(#[from] binius_transcript::Error),
	#[error("channel error: {0}")]
	Channel(#[from] ChannelError),
	#[error("IOP channel error: {0}")]
	IOPChannel(#[from] IOPChannelError),
	#[error("FRI error: {0}")]
	FRI(#[from] fri::Error),
	#[error("ring switch error: {0}")]
	RingSwitch(#[from] ring_switch::Error),
	#[error("IntMul error: {0}")]
	IntMul(#[from] intmul::Error),
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck::Error),
	#[error("incorrect public inputs length: expected {expected}, got {actual}")]
	IncorrectPublicInputLength { expected: usize, actual: usize },
	#[error("constraint system error: {0}")]
	ConstraintSystem(#[from] ConstraintSystemError),
	#[error("shift reduction error: {0}")]
	ShiftReduction(#[from] shift::Error),
}
