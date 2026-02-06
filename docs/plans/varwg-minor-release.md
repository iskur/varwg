# VarWG Minor Version Release Plan

**Goal**: Release a new minor version of varwg to PyPI with the new plotting parameters (`obs_title_str`, `sim_title_str` for `plot_meteogram_daily()`).

**Context**: These parameters are required by WeatherCop's updated plotting code in the quick start example. WeatherCop cannot be deployed until this varwg version is released.

---

## Phase 1: Pre-Release Verification

### Step 1.1: Understand Current State
- [ ] Check current version in `pyproject.toml` or `setup.py`
- [ ] Review git log to identify when the plotting parameters were added
- [ ] Verify that `obs_title_str` and `sim_title_str` parameters are already implemented in the codebase
- [ ] Confirm these are backward-compatible additions (new optional parameters with defaults)

### Step 1.2: Verify Tests Pass
- [ ] Run full test suite to ensure no regressions
- [ ] Check that any tests for the new plotting parameters pass
- [ ] Verify the package builds successfully

---

## Phase 2: Version and Documentation Updates

### Step 2.1: Update Version Number
- [ ] Identify current version (e.g., X.Y.Z)
- [ ] Bump minor version to X.(Y+1).0
- [ ] Update version in all relevant files:
  - `pyproject.toml` (version field)
  - `setup.py` (if present)
  - `__init__.py` or version module (if present)
  - Any other version references

### Step 2.2: Update Release Notes/Changelog
- [ ] Open CHANGELOG.md, CHANGES.md, or equivalent
- [ ] Add new entry for the version with date
- [ ] Document the new features:
  - Added `obs_title_str` parameter to `plot_meteogram_daily()`
  - Added `sim_title_str` parameter to `plot_meteogram_daily()`
  - Include any other fixes or features in this release
- [ ] Keep changelog entries clear and user-facing

---

## Phase 3: Build and Local Testing

### Step 3.1: Clean Build
- [ ] Remove any build artifacts: `rm -rf build/ dist/ *.egg-info/`
- [ ] Build the package: `python -m build` or `python setup.py sdist bdist_wheel`
- [ ] Verify build completes without errors

### Step 3.2: Test Installation
- [ ] Create a temporary virtual environment
- [ ] Install the built package locally
- [ ] Run quick smoke tests to verify basic functionality
- [ ] Clean up test environment

---

## Phase 4: Commit and Tag

### Step 4.1: Commit Version Changes
- [ ] Stage version and changelog files
- [ ] Create commit: `git commit -m "chore: bump version to X.(Y+1).0"`
- [ ] Include release notes summary in commit message if helpful

### Step 4.2: Create Git Tag
- [ ] Create annotated tag: `git tag -a vX.(Y+1).0 -m "Release vX.(Y+1).0"`
- [ ] Verify tag created: `git tag -l vX.(Y+1).0`

---

## Phase 5: PyPI Release

### Step 5.1: Publish to PyPI
- [ ] Ensure you have PyPI credentials configured (`.pypirc`)
- [ ] Build final distributions: `python -m build`
- [ ] Upload to PyPI: `python -m twine upload dist/*`
- [ ] Or if using uv: `uv build && uv publish` (if supported)

### Step 5.2: Verify PyPI Deployment
- [ ] Visit https://pypi.org/project/varwg/ to confirm new version is listed
- [ ] Check that version shows as latest
- [ ] Verify package metadata and files are correct

---

## Phase 6: Post-Release

### Step 6.1: Push to GitHub
- [ ] Push commits: `git push origin [branch]`
- [ ] Push tag: `git push origin vX.(Y+1).0`

### Step 6.2: Create GitHub Release (Optional)
- [ ] Go to Releases page on GitHub
- [ ] Create release from tag
- [ ] Add release notes from CHANGELOG
- [ ] Publish release

### Step 6.3: Update WeatherCop
- [ ] Return to WeatherCop project session
- [ ] Update `pyproject.toml` to require new varwg version if needed
- [ ] Verify WeatherCop tests still pass
- [ ] Commit and push WeatherCop changes

---

## Key Files to Check
- `pyproject.toml` - Main package configuration and version
- `setup.py` - Alternative build configuration (if present)
- `CHANGELOG.md` or `CHANGES.md` - Release notes
- `src/varwg/__init__.py` or similar - Version reference in code
- `src/varwg/plotting.py` or similar - Where plotting parameters are implemented

## Notes
- This is a **minor version bump** (new features without breaking existing API)
- The new parameters should have sensible defaults for backward compatibility
- All commits and tags should be clear and descriptive
- After successful PyPI release, WeatherCop can be updated and pushed
