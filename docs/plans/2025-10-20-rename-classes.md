# Class Renaming Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Rename VG, VGBase, and VGPlotting classes to match the new package structure and improve clarity.

**Architecture:** This is a systematic refactoring that renames three classes in an inheritance hierarchy (Base → Plotting → VarWG) and updates all references throughout the codebase. The approach is to rename each class in the inheritance order and systematically update all imports and usages.

**Tech Stack:** Python, pytest, git

---

## Class Renaming Strategy

**Renaming scheme:**
- `VGBase` → `Base` (lives in `base.py`)
- `VGPlotting` → `Plotting` (lives in `plotting.py`)
- `VG` → `VarWG` (the main weather generator class)

**Inheritance chain after renaming:**
- `Base` (base.py:320)
- `Plotting(Base)` (plotting.py:48)
- `VarWG(Plotting)` (core.py:286)

---

## Task 1: Rename VGBase class to Base

**Files:**
- Modify: `src/varwg/core/base.py:320` (class definition)
- Modify: `src/varwg/core/plotting.py:48` (class inheritance)
- Modify: `src/varwg/core/tests/test_core.py` (docstring references)

**Step 1: Find all VGBase references in base.py**

Run: `grep -n "VGBase" /home/dirk/Sync/vg/src/varwg/core/base.py`

Expected: Show line numbers where VGBase appears (mainly in docstrings and the class definition).

**Step 2: Rename class definition in base.py:320**

Replace:
```python
class VGBase(object):
```

With:
```python
class Base(object):
```

**Step 3: Update plotting.py inheritance**

In `src/varwg/core/plotting.py:48`, replace:
```python
class VGPlotting(base.VGBase):
```

With:
```python
class Plotting(base.Base):
```

**Step 4: Update test file docstring references**

In `src/varwg/core/tests/test_core.py`, replace any references to `VGBase` in comments/docstrings with `Base`.

**Step 5: Run tests to verify nothing broke**

Run: `python3 -m pytest /home/dirk/Sync/vg/src/varwg/core/tests/test_core.py -v --tb=short`

Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/varwg/core/base.py src/varwg/core/plotting.py src/varwg/core/tests/test_core.py
git commit -m "refactor: rename VGBase class to Base"
```

---

## Task 2: Rename VGPlotting class to Plotting

**Files:**
- Modify: `src/varwg/core/plotting.py:48` (class definition - already done in Task 1)
- Modify: `src/varwg/core/core.py:286` (class inheritance)
- Modify: docstrings and comments referencing VGPlotting

**Step 1: Find all VGPlotting references**

Run: `grep -rn "VGPlotting" /home/dirk/Sync/vg/src/varwg/ --include="*.py"`

Expected: Show all files that reference the class (primarily core.py, docstrings, and comments).

**Step 2: Update core.py class inheritance**

In `src/varwg/core/core.py:286`, replace:
```python
class VG(plotting.VGPlotting):
```

With:
```python
class VarWG(plotting.Plotting):
```

**Step 3: Update docstring references**

Search for docstrings mentioning `VGPlotting` and update them to reference `Plotting` instead.

**Step 4: Run tests to verify**

Run: `python3 -m pytest /home/dirk/Sync/vg/src/varwg/core/tests/test_core.py::TestVG -v --tb=short`

Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/varwg/core/plotting.py src/varwg/core/core.py
git commit -m "refactor: rename VGPlotting class to Plotting"
```

---

## Task 3: Rename VG class to VarWG

**Files:**
- Modify: `src/varwg/core/core.py:286` (class definition - already done in Task 2)
- Modify: `src/varwg/__init__.py` (package exports)
- Modify: All files that import and use `VG`

**Step 1: Update package __init__.py**

In `src/varwg/__init__.py`, replace:
```python
from .core.core import VG, read_met
```

With:
```python
from .core.core import VarWG, read_met
# Keep backward compatibility
VG = VarWG
```

This maintains backward compatibility so existing code using `varwg.VG` still works.

**Step 2: Update all file imports**

Search and replace in all Python files:
- `from varwg.core.core import VG` → `from varwg.core.core import VarWG` (or keep using VG via the alias)
- `from varwg import VG` → `from varwg import VarWG` (or keep using VG via the alias)

Files to update:
- `src/varwg/core/tests/test_core.py`
- `src/varwg/core/tests/gen_test_data.py`
- `src/varwg/time_series_analysis/seasonal_distributions.py`
- `src/varwg/time_series_analysis/seasonal_kde.py`
- `src/varwg/time_series_analysis/distributions.py`
- `src/varwg/time_series_analysis/resample.py`
- `src/varwg/time_series_analysis/rain_stats.py`
- `src/varwg/time_series_analysis/models.py`
- `src/varwg/time_series_analysis/time_series.py`
- `src/varwg/time_series_analysis/tests/test_cresample.py`

**Step 3: Update test class setup**

In `src/varwg/core/tests/test_core.py`, update instantiations:

Replace:
```python
self.met_vg = varwg.VG(var_names, ...)
```

With:
```python
self.met_vg = varwg.VarWG(var_names, ...)
```

Or if using the VG alias:
```python
self.met_vg = varwg.VG(var_names, ...)  # VG is an alias for VarWG
```

**Step 4: Update docstrings and comments**

Replace references to `VG` class with `VarWG` in docstrings and comments (be careful not to break "VG" references that mean "weather generator" generically).

**Step 5: Run full test suite**

Run: `python3 -m pytest /home/dirk/Sync/vg/src/varwg/ -v --tb=short`

Expected: All tests pass (81+ tests).

**Step 6: Commit**

```bash
git add src/varwg/__init__.py src/varwg/core/tests/test_core.py src/varwg/core/tests/gen_test_data.py
git add src/varwg/time_series_analysis/*.py
git commit -m "refactor: rename VG class to VarWG with backward compatibility alias"
```

---

## Task 4: Final verification and cleanup

**Files:**
- Verification only (no modifications)

**Step 1: Verify all imports work**

Run:
```bash
python3 -c "from varwg import VarWG, VG; print('✓ VarWG imported'); print('✓ VG alias works'); print('VG is VarWG:', VG is VarWG)"
```

Expected: Both imports work, VG is an alias for VarWG.

**Step 2: Verify class hierarchy**

Run:
```bash
python3 -c "from varwg.core.base import Base; from varwg.core.plotting import Plotting; from varwg.core.core import VarWG; print(f'VarWG bases: {VarWG.__bases__}'); print(f'Plotting bases: {Plotting.__bases__}'); print(f'Base bases: {Base.__bases__}')"
```

Expected: Shows correct inheritance chain:
- VarWG → Plotting
- Plotting → Base
- Base → object

**Step 3: Run full test suite**

Run: `python3 -m pytest /home/dirk/Sync/vg/src/varwg/ -v --tb=short`

Expected: All tests pass (81+ tests, 0 failures).

**Step 4: Check for remaining old class names**

Run: `grep -rn "class VG\|class VGBase\|class VGPlotting" /home/dirk/Sync/vg/src/varwg/ --include="*.py"`

Expected: No results (or only in comments explaining the refactoring).

**Step 5: Final summary**

Create a summary showing:
- Total class renames completed: 3 (VGBase→Base, VGPlotting→Plotting, VG→VarWG)
- Test results: PASS (all tests passing)
- Backward compatibility: VG alias maintained
- No breaking changes for code using the varwg package

No commit needed for this task - it's verification only.

---

## Success Criteria

✅ VGBase renamed to Base
✅ VGPlotting renamed to Plotting
✅ VG renamed to VarWG
✅ Backward compatibility alias (VG = VarWG) maintained
✅ All imports updated
✅ All tests passing (81+)
✅ Inheritance chain correct
✅ No remaining references to old class names (except in comments/docs)

---

## Estimated Time

- Task 1: 5-8 minutes
- Task 2: 3-5 minutes
- Task 3: 10-15 minutes (many files to update)
- Task 4: 3-5 minutes

**Total: 21-33 minutes**

---

## Rollback Plan

If critical issues occur:
1. Run `git log --oneline` to see recent commits
2. Run `git revert HEAD` to undo the most recent commit
3. Investigate and retry

All commits are atomic, making rollback straightforward.
