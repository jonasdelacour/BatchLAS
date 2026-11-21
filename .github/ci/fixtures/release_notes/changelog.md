# Changelog {#changelog}

Fixture for `.github/ci/release_notes.py --self-test`: every Keep a Changelog heading shape the
extractor has to read or refuse. Not a real changelog.

## [Unreleased] {#changelog-unreleased}

### Added

- Not released yet.

## [1.0.0] - 2026-11-30 {#changelog-1-0-0}

First release.

### Added {#changelog-1-0-0-added}

- `<batchlas/version.hh>` (#300), see [versioning](cpp-api.md#versioning-and-compatibility).
- A reference-style link to [the guide][guide] and the [1.0.0] comparison.
- A fenced heading does not end the section:

```markdown
## [0.0.1] - 2000-01-01 {#not-a-heading}
```

### Changed {#changelog-1-0-0-changed}

- `SameMajorVersion` ([#301](https://github.com/o/r/pull/301)).

### Fixed

- Nothing is lost after the last subsection.

## 0.9.0 – 2026-01-01

- Unbracketed.

## [0.8.0] - 2025-06-01 [YANKED]

- Pulled.

## [0.7.0] - YYYY-MM-DD

- Not dated yet.

## [0.6.0] - 2026-02-30

- No such day.

## [0.5.0] - 2025-01-01

## [0.4.0] - 2024-01-01

- See @ref guide_build for the steps.

[Unreleased]: https://github.com/o/r/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/o/r/compare/v0.9.0...v1.0.0
[guide]: guide/build.md
[0.9.0]: https://github.com/o/r/releases/tag/v0.9.0
