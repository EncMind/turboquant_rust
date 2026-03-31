# Changelog

All notable changes to the Rust implementation are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cross-platform `rust-py` CI job in GitHub Actions:
  - `ubuntu-latest` build without Accelerate.
  - `macos-latest` build with `--features accelerate`.
- Portable default feature set for `turboquant-py` (no macOS-only dependency by default).
- Value-driven outlier channel selection in `OutlierTurboQuant` with per-batch selected channel metadata.
- Shared typed error model (`TurboQuantError`) and crate-wide `Result` alias.
- Publishing metadata for Cargo crates and Python package (repository/homepage/docs URLs, keywords/categories).

### Changed
- Public validation paths across core quantization APIs now return typed `Result<_, TurboQuantError>` instead of panicking on invalid input.
- Python bindings now surface Rust validation failures as `ValueError` with typed error messages.
- Tests updated to assert error returns for invalid inputs instead of panic-string matching.

### Documentation
- README examples and snippets updated for explicit wire/in-memory metric naming and fallible API usage.

