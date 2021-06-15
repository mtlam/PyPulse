# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2021-06-15

### Added

- Fitting errors on the 1D scintillation parameters are now added. This changes the underlying `functionfit` output but this is not part of the public API.

### Changed

- Converted multiple internal `map` calls to list comprehensions.
- `DynamicSpectrum` now copies colormaps with `copy.copy` to remove `MatplotlibDeprecationWarning`
- Fixed scrunching and waterfall plot limits due to updates in numpy in the handling of NaNs. In waterfall plot, duplicate calls to set axis limits have been removed.
- Fixed pointers to `np.str` to `np.str_` due to updates in numpy.

## [0.1.0] - 2021-05-03

### Added

- Added CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

### Changed

- Start following [SemVer](https://semver.org) properly.



[unreleased]: https://github.com/mtlam/pypulse/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/mtlam/pypulse/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mtlam/pypulse/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/mtlam/pypulse/releases/tag/v0.0.1
