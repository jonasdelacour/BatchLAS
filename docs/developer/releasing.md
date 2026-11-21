# Releasing {#dev_releasing}

> **Status:** current · release workflow `.github/workflows/release.yml`, extractor `.github/ci/release_notes.py`

How to cut a BatchLAS release: bump the version, date the changelog section, push a `vX.Y.Z` tag.
The tag publishes a source-only GitHub Release whose notes are that version's section of the
[changelog](../changelog.md). What a major, minor or patch number promises is in
[versioning and compatibility](../cpp-api.md#versioning-and-compatibility).

## Releasing: where the version number lives {#releasing-version-bump-points}

Two files carry the number, and they must agree:

| File | What to change |
| --- | --- |
| `CMakeLists.txt` | `project(BatchLAS VERSION X.Y.Z ...)` |
| `include/batchlas/version.hh` | `BATCHLAS_VERSION_MAJOR`, `_MINOR`, `_PATCH` and `BATCHLAS_VERSION_STRING` |

`version.hh` is checked in, not generated, because header-only use never runs CMake. Configure
stops when the two disagree (`cmake/BatchLASVersion.cmake`). Everything else follows from
`project()`:

- the SONAME: `SOVERSION` is the major (`src/CMakeLists.txt`), so `libbatchlas*.so.1`;
- the CMake package version file, `SameMajorVersion` (`cmake/BatchLASPackaging.cmake`);
- `Version:` in `batchlas.pc` (`cmake/batchlas.pc.in`);
- `batchlas.__version__`, compiled from `BATCHLAS_VERSION_STRING` (`python/batchlas/module.cc`).

A major bump is an ABI or API break of the covered surface, so it also changes the SONAME. The
consumer test (`examples/consumer_test.sh`, Phase 3b) asks `find_package` for the current major
and minor, and checks it refuses the version below and the next minor and major. Update its probes
with a major bump.

## Releasing: the changelog section {#releasing-the-changelog-section}

`docs/changelog.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Changes land
under `## [Unreleased]` as they merge. To release:

1. Rename `[Unreleased]` to the version with the release date, in ISO form, and open a new empty
   `[Unreleased]` above it:
   `## [1.1.0] - 2027-02-14 {#changelog-1-1-0}`. The `{#...}` id is for Doxygen; keep ids unique.
2. Group entries under `### Added`, `### Changed`, `### Deprecated`, `### Removed`, `### Fixed`
   and `### Security`. Cite a pull request as `#N`, or a commit by its short id.
3. Use Markdown links only, relative to `docs/` (`design/known-defects.md#...`); no `@ref`, which
   GitHub cannot render. The extractor rewrites relative links to `blob/<tag>/docs/...`.
4. Update the link definitions at the bottom: `[Unreleased]` compares the new tag with `HEAD`, and
   the new version gets its own line.

Until the date is known the heading may hold a placeholder such as `2026-11-XX`; the extractor
refuses it, so a tag cannot be published from an undated section.

Check the section before tagging, exactly as the workflow will:

```bash
python3 .github/ci/release_notes.py --self-test       # the extractor still reads the format
python3 .github/ci/release_notes.py v1.1.0 --check    # tag, CMake, version.hh, a dated section
python3 .github/ci/release_notes.py v1.1.0            # the notes, on stdout
```

## Releasing: the tag and what release.yml does {#releasing-the-tag-and-the-workflow}

Merge the bump and the dated changelog into `main`, let CI and the GPU gate pass (@ref dev_ci,
@ref testing), then tag that merge commit and push the tag:

```bash
git fetch origin
git tag -a v1.1.0 -m "BatchLAS 1.1.0" origin/main
git push origin v1.1.0
```

A pushed `v*` tag runs `.github/workflows/release.yml`:

| Job | Token | Does |
| --- | --- | --- |
| `notes` | read-only | runs `release_notes.py --self-test`, then `--check <tag>`, then extracts the notes with `--link-base` set to the tag's tree and uploads them as an artifact |
| `publish` | `contents: write` | no checkout, no repository code: `gh release create --verify-tag` with the notes, titled `BatchLAS X.Y.Z`; on a re-run, `gh release edit` |

`--check` refuses a tag that is not `vX.Y.Z` (so `v1.1.0-rc1` is refused), a tag that disagrees
with `project()` or `version.hh`, and a changelog section that is missing, undated, `[YANKED]`,
empty or uses `@ref`. A refused tag fails `notes` and publishes nothing; fix the tree on `main`,
then delete the tag and push it again at the fixed commit.

The release carries no built binaries: its assets are GitHub's own "Source code" zip and tar.gz.

## Releasing: Git LFS objects in the source archives {#releasing-git-lfs-in-archives}

`benchmarks/results/` is in Git LFS. GitHub's source archives contain the LFS pointer files, not
their contents, unless the repository setting **Settings → General → Archives → Include Git LFS
objects in archives** is on. The library, the tests and the documentation build without those
files, so a release works either way; turn the setting on when the archive should carry the raw
measurements behind the docs. The setting is per repository and applies to every archive GitHub
serves, so check it once before the first release.
