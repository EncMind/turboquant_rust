//! Error types for fallible TurboQuant public APIs.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Result alias used across the crate.
pub type Result<T> = std::result::Result<T, TurboQuantError>;

/// Typed error for invalid inputs and payload/schema mismatches.
#[derive(Debug, Clone, PartialEq)]
pub enum TurboQuantError {
    InvalidDimension {
        param: &'static str,
        got: usize,
        min: usize,
    },
    InvalidBitWidth {
        param: &'static str,
        got: u32,
        min: u32,
        max: u32,
    },
    InvalidBitWidthU8 {
        param: &'static str,
        got: u8,
        min: u8,
        max: u8,
    },
    InvalidTargetBits {
        got: f64,
        reason: &'static str,
    },
    EmptyInput {
        param: &'static str,
    },
    TooManyLevels {
        param: &'static str,
        got: usize,
        max: usize,
    },
    NonFiniteValue {
        param: &'static str,
        index: Option<usize>,
        value: f64,
    },
    UnsortedInput {
        param: &'static str,
    },
    LengthMismatch {
        param: &'static str,
        expected: usize,
        got: usize,
    },
    BufferTooShort {
        param: &'static str,
        expected_at_least: usize,
        got: usize,
    },
    ValueOutOfRange {
        param: &'static str,
        index: usize,
        got: u8,
        max: u8,
    },
    InvalidSignValue {
        param: &'static str,
        index: usize,
        got: i8,
    },
    MissingPayload {
        field: &'static str,
    },
    InvalidOutlierIndex {
        index: usize,
        d: usize,
    },
    DuplicateOutlierIndex {
        index: usize,
    },
    InvalidPackedMetadata {
        param: &'static str,
        expected: usize,
        got: usize,
    },
    Internal {
        context: &'static str,
        message: String,
    },
}

impl Display for TurboQuantError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimension { param, got, min } => {
                write!(f, "{param} must be >= {min}, got {got}")
            }
            Self::InvalidBitWidth {
                param,
                got,
                min,
                max,
            } => write!(f, "{param} must be in {min}..={max}, got {got}"),
            Self::InvalidBitWidthU8 {
                param,
                got,
                min,
                max,
            } => write!(f, "{param} must be in {min}..={max}, got {got}"),
            Self::InvalidTargetBits { got, reason } => {
                write!(f, "target_bits={got} is invalid: {reason}")
            }
            Self::EmptyInput { param } => write!(f, "{param} must be non-empty"),
            Self::TooManyLevels { param, got, max } => {
                write!(f, "{param} has {got} levels, max supported is {max}")
            }
            Self::NonFiniteValue {
                param,
                index,
                value,
            } => {
                if let Some(i) = index {
                    write!(f, "{param}[{i}] must be finite, got {value}")
                } else {
                    write!(f, "{param} must be finite, got {value}")
                }
            }
            Self::UnsortedInput { param } => {
                write!(f, "{param} must be sorted in non-decreasing order")
            }
            Self::LengthMismatch {
                param,
                expected,
                got,
            } => write!(f, "{param} length mismatch: expected {expected}, got {got}"),
            Self::BufferTooShort {
                param,
                expected_at_least,
                got,
            } => write!(
                f,
                "{param} buffer too short: expected at least {expected_at_least} bytes, got {got}"
            ),
            Self::ValueOutOfRange {
                param,
                index,
                got,
                max,
            } => write!(f, "{param}[{index}]={got} exceeds max {max}"),
            Self::InvalidSignValue { param, index, got } => {
                write!(f, "{param}[{index}] must be either -1 or 1, got {got}")
            }
            Self::MissingPayload { field } => write!(f, "missing payload: {field}"),
            Self::InvalidOutlierIndex { index, d } => {
                write!(
                    f,
                    "outlier index {index} is out of bounds for dimension {d}"
                )
            }
            Self::DuplicateOutlierIndex { index } => write!(f, "duplicate outlier index {index}"),
            Self::InvalidPackedMetadata {
                param,
                expected,
                got,
            } => write!(
                f,
                "invalid packed metadata for {param}: expected {expected}, got {got}"
            ),
            Self::Internal { context, message } => write!(f, "{context}: {message}"),
        }
    }
}

impl Error for TurboQuantError {}
