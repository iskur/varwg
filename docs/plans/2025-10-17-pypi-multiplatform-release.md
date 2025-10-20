# PyPI Multi-Platform Release with Cython Extensions Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Configure and deploy multi-platform wheel builds (Linux, Windows, macOS) for PyPI with Cython extensions using GitHub Actions and cibuildwheel.

**Architecture:** Replace current single-platform build with cibuildwheel-based GitHub Actions workflow that builds wheels for multiple Python versions and platforms. Pre-compiled wheels eliminate need for users to have C compilers, making installation smooth across all platforms.

**Tech Stack:** cibuildwheel, GitHub Actions, PyPI API, setuptools, Cython

---

## Background

**Current State:**
- Project has 3 Cython extensions: `vg.ctimes`, `vg.time_series_analysis.cresample`, `vg.meteo.meteox2y_cy`
- `setup.py` exists with proper Cython configuration and Windows/Linux platform detection
- Already uploaded to test.pypi.org (single platform only)
- Current `.github/workflows/release.yml` builds on Ubuntu only → produces only `linux_x86_64` wheels

**Problem:** Users on Windows/macOS must have C compiler + Cython to install. PyPI best practice is pre-built wheels.

**Solution:** Use `cibuildwheel` to build wheels for:
- Linux: x86_64, aarch64
- Windows: AMD64
- macOS: x86_64, arm64 (Apple Silicon)

---

## Task 1: Add cibuildwheel Configuration

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add cibuildwheel configuration section**

Add to `pyproject.toml` after the `[tool.pytest.ini_options]` section:

```toml
[tool.cibuildwheel]
# Build for common platforms
build = "cp313-*"
skip = "pp* *-musllinux_*"

# Run basic import test for each built wheel
test-command = "python -c \"import vg; import vg.ctimes; import vg.time_series_analysis.cresample; import vg.meteo.meteox2y_cy; print('All extensions loaded successfully')\""
test-requires = ["numpy", "pandas", "scipy", "xarray", "bottleneck", "numexpr"]

# Platform-specific settings
[tool.cibuildwheel.linux]
archs = ["x86_64", "aarch64"]
before-build = "pip install cython numpy setuptools"

[tool.cibuildwheel.windows]
archs = ["AMD64"]
before-build = "pip install cython numpy setuptools"

[tool.cibuildwheel.macos]
archs = ["x86_64", "arm64"]
before-build = "pip install cython numpy setuptools"
```

**Step 2: Verify configuration syntax**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`
Expected: No syntax errors

**Step 3: Commit configuration**

```bash
git add pyproject.toml
git commit -m "build: Add cibuildwheel configuration for multi-platform wheels"
```

---

## Task 2: Create Multi-Platform Build Workflow

**Files:**
- Create: `.github/workflows/build-wheels.yml`

**Step 1: Write the build-wheels workflow**

Create `.github/workflows/build-wheels.yml`:

```yaml
name: Build Multi-Platform Wheels

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
  pull_request:
    paths:
      - 'setup.py'
      - 'pyproject.toml'
      - 'src/vg/**/*.pyx'
      - '.github/workflows/build-wheels.yml'

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-13, macos-14]
        # macos-13 is x86_64, macos-14 is arm64

    steps:
    - uses: actions/checkout@v4

    - name: Build wheels
      uses: pypa/cibuildwheel@v2.21.3
      env:
        CIBW_BUILD_VERBOSITY: 1

    - name: Upload wheels as artifacts
      uses: actions/upload-artifact@v4
      with:
        name: wheels-${{ matrix.os }}
        path: ./wheelhouse/*.whl
        if-no-files-found: error

  build_sdist:
    name: Build source distribution
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Install uv
      uses: astral-sh/setup-uv@v6

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.13'

    - name: Install build dependencies
      run: uv pip install --system cython numpy setuptools build

    - name: Build sdist
      run: uv build --sdist

    - name: Upload sdist as artifact
      uses: actions/upload-artifact@v4
      with:
        name: sdist
        path: dist/*.tar.gz
        if-no-files-found: error

  publish:
    name: Publish to PyPI
    needs: [build_wheels, build_sdist]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')

    steps:
    - name: Download all artifacts
      uses: actions/download-artifact@v4
      with:
        path: dist
        merge-multiple: true

    - name: List distributions to upload
      run: ls -lh dist/

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
        skip-existing: true
```

**Step 2: Validate YAML syntax**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/build-wheels.yml'))"`
Expected: No syntax errors (may need to install pyyaml: `pip install pyyaml`)

**Step 3: Commit the workflow**

```bash
git add .github/workflows/build-wheels.yml
git commit -m "ci: Add multi-platform wheel build workflow with cibuildwheel"
```

---

## Task 3: Update Existing Release Workflow

**Files:**
- Modify: `.github/workflows/release.yml`

**Step 1: Archive the old single-platform workflow**

The new `build-wheels.yml` replaces the functionality in `release.yml`. We should either:
- Option A: Delete `release.yml` entirely
- Option B: Rename to `release.yml.disabled` for reference

Let's rename for safety:

```bash
git mv .github/workflows/release.yml .github/workflows/release.yml.old
```

**Step 2: Commit the change**

```bash
git commit -m "ci: Archive old single-platform release workflow"
```

**Note:** After successful multi-platform release, delete `release.yml.old`.

---

## Task 4: Verify MANIFEST.in Includes Cython Sources

**Files:**
- Check: `MANIFEST.in`

**Step 1: Verify .pyx files are included**

Current `MANIFEST.in` contains:
```
include src/vg/time_series_analysis/cresample.pyx
include src/vg/ctimes.pyx
include src/vg/meteo/meteox2y_cy.pyx
```

These are correct. The source distribution needs .pyx files so users without wheels can build from source.

**Step 2: Verify by building a test sdist**

Run: `python -m build --sdist`
Expected: Creates `dist/vg-1.4.0.tar.gz`

**Step 3: Inspect the sdist contents**

Run: `tar -tzf dist/vg-1.4.0.tar.gz | grep "\.pyx$"`
Expected output:
```
vg-1.4.0/src/vg/ctimes.pyx
vg-1.4.0/src/vg/meteo/meteox2y_cy.pyx
vg-1.4.0/src/vg/time_series_analysis/cresample.pyx
```

**Step 4: No commit needed if verification passes**

---

## Task 5: Test Workflow Locally with act (Optional)

**Files:**
- None (local testing only)

**Step 1: Install act if not already installed**

Run: `which act || echo "act not installed - skip this task"`

If act is not installed and you want to test locally:
```bash
# On Linux
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

**Step 2: Test the workflow locally (Linux only)**

Run: `act -j build_wheels -P ubuntu-latest=catthehacker/ubuntu:act-latest`
Expected: Workflow runs and produces wheels in `wheelhouse/`

**Note:** Local testing is limited - full matrix testing requires pushing to GitHub.

**Step 3: No commit needed**

---

## Task 6: Test Multi-Platform Build on GitHub (Dry Run)

**Files:**
- None (GitHub Actions testing)

**Step 1: Push to a test branch**

```bash
git checkout -b test/multiplatform-wheels
git push origin test/multiplatform-wheels
```

**Step 2: Manually trigger the workflow**

1. Go to GitHub repository
2. Navigate to Actions → Build Multi-Platform Wheels
3. Click "Run workflow" → Select branch `test/multiplatform-wheels`
4. Wait for completion (15-30 minutes for full matrix)

**Step 3: Verify artifacts**

Check that workflow produces:
- `wheels-ubuntu-latest`: Contains `*-cp313-cp313-linux_x86_64.whl` and `*-cp313-cp313-linux_aarch64.whl`
- `wheels-windows-latest`: Contains `*-cp313-cp313-win_amd64.whl`
- `wheels-macos-13`: Contains `*-cp313-cp313-macosx_*_x86_64.whl`
- `wheels-macos-14`: Contains `*-cp313-cp313-macosx_*_arm64.whl`
- `sdist`: Contains `vg-1.4.0.tar.gz`

**Step 4: Download and test a wheel locally**

Download one of the wheels from artifacts, then:

```bash
pip install vg-1.4.0-cp313-cp313-linux_x86_64.whl
python -c "import vg; import vg.ctimes; print('Success!')"
pip uninstall -y vg
```

Expected: Import succeeds without compilation

**Step 5: No commit needed**

---

## Task 7: Update README with Installation Instructions

**Files:**
- Modify: `README.md`

**Step 1: Find installation section**

Run: `grep -n "## Install" README.md`
Expected: Shows line number of installation section

**Step 2: Update installation docs**

In `README.md`, update the installation section to mention pre-built wheels:

```markdown
## Installation

### From PyPI (Recommended)

```bash
pip install vg
```

Pre-built wheels are available for:
- **Linux**: x86_64, aarch64
- **Windows**: AMD64
- **macOS**: x86_64 (Intel), arm64 (Apple Silicon)

No compiler needed! If a wheel isn't available for your platform, pip will automatically build from source (requires C compiler and Cython).

### From Source

```bash
git clone https://github.com/iskur/vg.git
cd vg
pip install -e .
```

Building from source requires:
- C compiler (gcc/clang/MSVC)
- Cython >= 3.1.1
- NumPy >= 1.26.0
```
```

**Step 3: Commit the documentation update**

```bash
git add README.md
git commit -m "docs: Update installation instructions for multi-platform wheels"
```

---

## Task 8: Configure PyPI API Token (if not already set)

**Files:**
- None (GitHub Settings configuration)

**Step 1: Verify PYPI_API_TOKEN secret exists**

1. Go to GitHub repository settings
2. Navigate to Settings → Secrets and variables → Actions
3. Check if `PYPI_API_TOKEN` exists

**Step 2: If token doesn't exist, create it**

1. Go to https://pypi.org/manage/account/token/
2. Create new API token with scope: "Entire account" or specific to "vg" project
3. Copy the token (starts with `pypi-`)
4. Add to GitHub secrets as `PYPI_API_TOKEN`

**Step 3: Test with test.pypi.org first (optional)**

To test the upload process without affecting production PyPI:

1. Create token at https://test.pypi.org/manage/account/token/
2. Add as `TEST_PYPI_API_TOKEN` secret
3. Temporarily modify workflow to use:
   ```yaml
   repository-url: https://test.pypi.org/legacy/
   password: ${{ secrets.TEST_PYPI_API_TOKEN }}
   ```

**Step 4: No commit needed**

---

## Task 9: Merge to Main and Prepare Release

**Files:**
- None (Git operations)

**Step 1: Merge test branch to main**

```bash
git checkout main
git merge test/multiplatform-wheels
```

**Step 2: Verify version in pyproject.toml**

Run: `grep "^version" pyproject.toml`
Expected: `version = "1.4.0"`

If version needs updating:
```bash
# Edit pyproject.toml to bump version to 1.4.0
git add pyproject.toml
git commit -m "chore: Bump version to 1.4.0"
```

**Step 3: Push to main**

```bash
git push origin main
```

**Step 4: Verify CI passes**

Check GitHub Actions for `main` branch - ensure CI workflow passes.

---

## Task 10: Create Release Tag and Publish

**Files:**
- None (Git tags and GitHub release)

**Step 1: Create and push version tag**

```bash
git tag -a v1.4.0 -m "Release v1.4.0: Multi-platform wheel support"
git push origin v1.4.0
```

**Step 2: Monitor GitHub Actions**

1. Go to GitHub Actions
2. Watch "Build Multi-Platform Wheels" workflow triggered by tag
3. Wait for all jobs to complete (15-30 minutes)
4. Verify "publish" job succeeds

**Step 3: Verify publication on PyPI**

1. Go to https://pypi.org/project/vg/
2. Click on "Download files"
3. Verify presence of wheels for all platforms:
   - `vg-1.4.0-cp313-cp313-linux_x86_64.whl`
   - `vg-1.4.0-cp313-cp313-linux_aarch64.whl`
   - `vg-1.4.0-cp313-cp313-win_amd64.whl`
   - `vg-1.4.0-cp313-cp313-macosx_*_x86_64.whl`
   - `vg-1.4.0-cp313-cp313-macosx_*_arm64.whl`
   - `vg-1.4.0.tar.gz`

**Step 4: Test installation from PyPI**

In a clean environment:
```bash
python -m pip install --upgrade vg
python -c "import vg; import vg.ctimes; import vg.time_series_analysis.cresample; import vg.meteo.meteox2y_cy; print('All extensions imported successfully')"
```

Expected: Installation uses pre-built wheel, imports succeed instantly

---

## Task 11: Create GitHub Release

**Files:**
- None (GitHub UI)

**Step 1: Draft release on GitHub**

1. Go to repository → Releases → "Draft a new release"
2. Choose tag: `v1.4.0`
3. Release title: `v1.4.0 - Multi-Platform Wheel Support`
4. Description:

```markdown
## What's New

- 🎉 **Pre-built wheels for Windows, macOS, and Linux**
  - No compiler required for installation
  - Supports Python 3.13
  - Linux: x86_64, aarch64
  - Windows: AMD64
  - macOS: Intel (x86_64) and Apple Silicon (arm64)

## Installation

```bash
pip install vg
```

## Full Changelog

See [CHANGELOG.md](https://github.com/iskur/vg/blob/main/CHANGELOG.md)
```

**Step 2: Publish release**

Click "Publish release"

**Step 3: No commit needed**

---

## Task 12: Update CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

**Step 1: Add entry for v1.4.0**

Add at the top of `CHANGELOG.md`:

```markdown
## [1.4.0] - 2025-10-17

### Added
- Multi-platform wheel distribution via cibuildwheel
- Pre-built wheels for Python 3.13 on:
  - Linux (x86_64, aarch64)
  - Windows (AMD64)
  - macOS (x86_64, arm64/Apple Silicon)

### Changed
- Migrated release workflow from uv build to cibuildwheel
- Installation no longer requires C compiler on supported platforms

### Infrastructure
- Added `.github/workflows/build-wheels.yml` for multi-platform builds
- Updated `pyproject.toml` with cibuildwheel configuration

```

**Step 2: Commit changelog**

```bash
git add CHANGELOG.md
git commit -m "docs: Update CHANGELOG for v1.4.0 multi-platform release"
git push origin main
```

---

## Task 13: Monitor and Respond to Issues

**Files:**
- None (Monitoring)

**Step 1: Monitor PyPI download statistics**

Over the next few days, check:
1. PyPI project page for download counts per platform
2. GitHub issues for installation problems

**Step 2: Common issues to watch for**

**Issue:** Windows users report missing MSVC runtime
**Solution:** Usually auto-installed, but document requirement if needed

**Issue:** macOS arm64 wheel not compatible with older macOS
**Solution:** Adjust `MACOSX_DEPLOYMENT_TARGET` in cibuildwheel config if needed

**Issue:** Linux aarch64 wheel too large or slow
**Solution:** Normal for ARM builds, document in README if users report concerns

**Step 3: Be prepared to release patch version**

If critical issues arise:
1. Fix in `main` branch
2. Bump to `v1.4.1`
3. Push tag to trigger rebuild

---

## Success Criteria

- [ ] `build-wheels.yml` workflow runs successfully on all platforms
- [ ] PyPI shows wheels for Linux (x86_64, aarch64), Windows (AMD64), macOS (x86_64, arm64)
- [ ] Fresh `pip install vg` works without compilation on all supported platforms
- [ ] All Cython extensions import successfully after wheel installation
- [ ] README documents multi-platform wheel support
- [ ] GitHub Release v1.4.0 published with release notes

---

## Rollback Plan

If issues arise during release:

1. **Before tagging:** Just fix issues in branch and re-test workflow
2. **After tagging, before PyPI publish:** Cancel GitHub Action, delete tag, fix, re-tag
3. **After PyPI publish:** Cannot delete releases, must release patch version (v1.4.1) with fixes

**Emergency PyPI removal:** Contact PyPI admins (only for security issues)

---

## References

- cibuildwheel docs: https://cibuildwheel.pypa.io/
- PyPA wheel naming: https://packaging.python.org/specifications/binary-distribution-format/
- GitHub Actions pypi-publish: https://github.com/pypa/gh-action-pypi-publish
- Platform tags: https://packaging.python.org/specifications/platform-compatibility-tags/

---

## Notes

- **Python 3.13 only:** Current config builds only for cp313. To support 3.10-3.13, change `build = "cp313-*"` to `build = "cp310-* cp311-* cp312-* cp313-*"` (increases build time ~4x)
- **ARM builds are slow:** `aarch64` builds use QEMU emulation, expect 30-60 min per wheel
- **OpenMP on Windows:** Current `setup.py` disables OpenMP on Windows (lines 16-23). If needed, requires Windows-specific MSVC configuration
- **Test suite:** Workflow runs basic import test. For full test suite, add to `test-command` (increases build time significantly)
