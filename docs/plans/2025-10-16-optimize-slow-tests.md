# Implementation Plan: Optimize Slow Tests in test_vg.py

**Date**: 2025-10-16
**Status**: Draft
**Estimated Effort**: Medium (4-6 hours)

## Context

The tests in `src/vg/core/tests/test_vg.py` are extremely slow because:
1. `setUp()` creates 3 VG instances and fits them for every test (11 tests × 3 fits = 33 expensive operations)
2. Some tests simulate very long time periods (20-100 years)
3. No use of pytest fixtures or test parameterization
4. No caching of fitted models between test runs

## Goals

1. Reduce test execution time by at least 50%
2. Maintain test coverage and correctness
3. Make tests easier to maintain and understand
4. Enable developers to run quick smoke tests vs comprehensive tests

## Implementation Tasks

### Task 1: Profile Current Test Performance
**Priority**: High
**Estimated Time**: 30 minutes
**Dependencies**: None

Run pytest with duration reporting to establish baseline metrics.

**Steps**:
1. Run `pytest src/vg/core/tests/test_vg.py --durations=0 -v` and capture output
2. Document current timing for each test
3. Identify the 3-5 slowest tests
4. Calculate total test suite time

**Success Criteria**:
- Have documented baseline timings for all 11 tests
- Know which tests are slowest and why

**Potential Issues**:
- Tests might take very long to complete (10+ minutes)
- May need to interrupt and use pytest-timeout if tests hang

---

### Task 2: Convert setUp to Pytest Session-Scoped Fixtures
**Priority**: High
**Estimated Time**: 1 hour
**Dependencies**: Task 1

Replace the `setUp()` method with pytest fixtures that are created once per test session instead of once per test.

**Steps**:
1. Create `conftest.py` in `src/vg/core/tests/` if it doesn't exist
2. Define session-scoped fixtures for the three fitted VG instances:
   - `@pytest.fixture(scope="session")` for `vg_regr`
   - `@pytest.fixture(scope="session")` for `vg_dist`
   - `@pytest.fixture(scope="session")` for `vg_sim`
3. Define fixture for `sample_sim` data
4. Update test class to use fixtures instead of `setUp()`
5. Convert from `unittest.TestCase` to plain pytest functions (or use function-scoped fixtures)

**Code Changes**:
- New file: `src/vg/core/tests/conftest.py`
- Modified file: `src/vg/core/tests/test_vg.py`

**Success Criteria**:
- VG instances are created only 3 times total (not 33 times)
- All tests still pass
- No shared state mutations between tests

**Potential Issues**:
- Tests might mutate the VG instances, requiring function-scoped fixtures for some tests
- `self.data_dir` creates unique temp directories - need to handle this carefully
- Some tests call `.fit()` again with different parameters - these need special handling

**Testing**:
```bash
pytest src/vg/core/tests/test_vg.py -v
```

---

### Task 3: Reduce Simulation Time Periods for Fast Tests
**Priority**: High
**Estimated Time**: 45 minutes
**Dependencies**: Task 2

Create two test variants: fast (shorter simulations) and comprehensive (current simulation lengths).

**Steps**:
1. Add pytest markers for slow tests:
   ```python
   @pytest.mark.slow
   def test_theta_incr_comprehensive(...):
       # T = 365.25 * 20 (current)
   ```
2. Create fast variants with shorter simulation periods:
   - `test_theta_incr`: Reduce from 20 years to 5 years (still statistically valid)
   - `test_theta_incr_nonnormal`: Reduce from 100 years to 10 years
3. Update pytest configuration in `pyproject.toml` to:
   - Run fast tests by default
   - Run slow tests only with `pytest -m slow` or `pytest --run-slow`

**Code Changes**:
- Modified: `src/vg/core/tests/test_vg.py` - add markers and adjust T values
- Modified: `pyproject.toml` - add pytest markers configuration

**Success Criteria**:
- `pytest` (no args) runs fast tests only, completing in <2 minutes
- `pytest -m slow` runs comprehensive tests
- Both test variants pass and test the same behavior

**Potential Issues**:
- Shorter simulations might not be statistically robust enough
- Need to validate that 5-year and 10-year simulations still test the intended behavior

**Testing**:
```bash
# Fast tests
pytest src/vg/core/tests/test_vg.py -v

# Slow tests
pytest src/vg/core/tests/test_vg.py -m slow -v

# All tests
pytest src/vg/core/tests/test_vg.py --run-slow -v
```

---

### Task 4: Optimize test_resample and test_phase_randomization
**Priority**: Medium
**Estimated Time**: 45 minutes
**Dependencies**: Task 2

These tests create new VG instances inside the test - optimize this pattern.

**Steps**:
1. For `test_resample` (line 347):
   - Extract VG creation to a fixture or use existing fixtures
   - Consider if both `res_dict_nocy` and `res_dict_cy` iterations are necessary
   - Possibly parameterize instead of loop
2. For `test_phase_randomization` (line 652):
   - This test has a bug: iterates over `("regression", "distance")` but always uses "regression"
   - Fix the bug and consider if both methods need testing
   - Reduce number of simulation calls from 8 to 2-4

**Code Changes**:
- Modified: `src/vg/core/tests/test_vg.py` - refactor these two tests

**Success Criteria**:
- `test_resample` runs at least 30% faster
- `test_phase_randomization` bug is fixed and runs 50% faster
- All assertions still pass

**Potential Issues**:
- The bug in `test_phase_randomization` might be intentional (though unlikely)
- Reducing loop iterations might miss edge cases

**Testing**:
```bash
pytest src/vg/core/tests/test_vg.py::TestVG::test_resample -v
pytest src/vg/core/tests/test_vg.py::TestVG::test_phase_randomization -v
```

---

### Task 5: Add Caching for Fitted Models
**Priority**: Low
**Estimated Time**: 1.5 hours
**Dependencies**: Task 2

Cache fitted VG models to disk so they can be reused across test runs.

**Steps**:
1. Create a `tests/fixtures/` directory for cached models
2. Add fixture that:
   - Checks if cached model exists with matching parameters
   - Loads from cache if available
   - Fits and saves to cache if not available
3. Use content hash of `test_out_sample.met` as part of cache key
4. Add `--refresh-fixtures` flag to force re-fitting

**Code Changes**:
- New directory: `src/vg/core/tests/fixtures/`
- Modified: `src/vg/core/tests/conftest.py` - add caching logic
- Modified: `pyproject.toml` - add custom pytest option

**Success Criteria**:
- First test run creates cache
- Subsequent test runs use cache and start in <10 seconds
- `--refresh-fixtures` flag forces re-fitting

**Potential Issues**:
- Cache invalidation is hard - need robust cache key
- VG instances might not pickle correctly
- Cache might become stale if VG implementation changes

**Testing**:
```bash
# First run (creates cache)
pytest src/vg/core/tests/test_vg.py -v

# Second run (uses cache)
pytest src/vg/core/tests/test_vg.py -v

# Force refresh
pytest src/vg/core/tests/test_vg.py --refresh-fixtures -v
```

---

### Task 6: Verify Performance Improvements
**Priority**: High
**Estimated Time**: 30 minutes
**Dependencies**: Tasks 2-5

Measure improvements and document results.

**Steps**:
1. Run full test suite with `--durations=0` after optimizations
2. Compare to baseline from Task 1
3. Document percentage improvements per test and overall
4. Update this plan document with actual results

**Success Criteria**:
- Overall test time reduced by at least 50%
- All tests still pass
- No loss of test coverage

**Testing**:
```bash
pytest src/vg/core/tests/test_vg.py --durations=0 -v
```

---

## Risk Assessment

**High Risk**:
- Shared fixtures might introduce test interdependencies
- Reducing simulation time might reduce test quality

**Medium Risk**:
- Caching might hide bugs in VG fitting logic
- Converting from unittest to pytest might break things

**Low Risk**:
- Adding markers is low-risk and easily reversible

## Rollback Plan

1. All changes are in git - can revert commits
2. Keep original `test_vg.py` as `test_vg_original.py.bak` during refactoring
3. Ensure all tests pass before committing each task

## Testing Strategy

1. Run full test suite after each task
2. Use `pytest -v --tb=short` for detailed failure info
3. Consider using `pytest-xdist` for parallel test execution after optimizations
4. Run tests in isolation to ensure no shared state issues:
   ```bash
   pytest src/vg/core/tests/test_vg.py::TestVG::test_diff -v
   ```

## Success Metrics

- [ ] Test suite runs in <3 minutes (down from ~10+ minutes estimated)
- [ ] Fast tests run in <1 minute
- [ ] All 11 tests still pass
- [ ] No reduction in test coverage
- [ ] Test code is more maintainable (use of fixtures)

## Future Optimizations

After this plan:
1. Consider using `pytest-xdist` for parallel execution
2. Evaluate if some tests can be split into unit tests vs integration tests
3. Look into mocking expensive operations where appropriate
4. Consider if some tests should be moved to a "regression" or "slow" test suite

## Notes for Engineers

- **Context**: This is a legacy test file using `unittest.TestCase` - we're modernizing it
- **Testing VG models**: VG (Vector Generalized Linear Models) require fitting, which is expensive
- **Fixtures**: Read pytest fixture documentation: https://docs.pytest.org/en/stable/fixture.html
- **Markers**: Read pytest markers documentation: https://docs.pytest.org/en/stable/mark.html
- **Important**: The `verbose=False` setting hides important debug info - consider making this configurable
