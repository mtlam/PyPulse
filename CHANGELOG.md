# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Allow parsing of par file values in "NAME VALUE ERROR" format rather than "NAME VALUE FIT" format for some fold ephemerides inside the PSRFITS files.
- Default behavior of `Archive`'s `fitPulses()` now returns all values from `SinglePulse`'s `fitPulse()` command. Users can still supply specific indices with the `num` argument to `fitPulses()`
- Filled out more of `Calibrator`'s `pacv()` plot with correct labeling and sizing.
- Removed numpy warnings from `NaN`s in data array combination
- Removed unnecessary reference to `scipy.fftpack`, which is in legacy.

### Added

- In `Archive`'s `imshow()`, the y-axis can now be flipped with the `flip=True` argument. This allows inverted frequency axes to be turned right-side up. A `title` argument can now be supplied as well.
- `Archive`'s `plot()` now takes arguments of `subint`, `pol`, and `chan` to specifically select individual profile rather than requiring the data be 1D. By default, these arguments are `0`.
- `SinglePulse.getOffpulseRMS()` convenience function added, which calls `SinglePulse.getOffpulseNoise()`
- Individual phase plot implemented in Calibrator class as `phaseplot()`. This can also be called with `plot(mode="phase")`.

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
