# Complete VG→VARWG Renaming Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Complete the vg→varwg project renaming by fixing all remaining import statements, deleting duplicate files, and verifying the full test suite passes.

**Architecture:** The renaming is 85% complete. This plan focuses on fixing 5 critical import/reference errors, deleting duplicate files, and updating configuration. Each task updates one or more related files, verifies imports work, and commits the changes.

**Tech Stack:** Python, pytest, Cython (.pyx files), setuptools, pyproject.toml

---

## Task 1: Fix vg_plotting imports in src/varwg/core/core.py

**Files:**
- Modify: `src/varwg/core/core.py:17` (import statement)
- Modify: `src/varwg/core/core.py:286, 409, 2285` (usage references)

**Step 1: Read the file and verify current state**

Run: `head -20 /home/dirk/Sync/vg/src/varwg/core/core.py`

Expected: Line 17 shows `from varwg.core import base, vg_plotting`

**Step 2: Update the import statement on line 17**

Replace:
```python
from varwg.core import base, vg_plotting
```

With:
```python
from varwg.core import base, plotting
```

**Step 3: Update usage references to vg_plotting (3 occurrences)**

These references should be updated:
- Line 286: `vg_plotting.VGPlotting` → `plotting.VGPlotting`
- Line 409: `vg_plotting.conf` → `plotting.conf`
- Line 2285: `vg_plotting.VGPlotting` → `plotting.VGPlotting`

**Step 4: Verify imports work**

Run: `python3 -c "from varwg.core import core; print('✓ core import works'); from varwg.core.core import VG; print('✓ VG import works')"`

Expected: Both imports succeed without errors.

**Step 5: Verify plotting module is accessible**

Run: `python3 -c "from varwg.core import plotting; print('✓ plotting import works')"`

Expected: Import succeeds.

**Step 6: Commit**

```bash
git add src/varwg/core/core.py
git commit -m "fix: update vg_plotting import to plotting in core.py"
```

---

## Task 2: Fix imports in src/varwg/time_series_analysis/distributions.py

**Files:**
- Modify: `src/varwg/time_series_analysis/distributions.py:2763` (import in __main__ block)

**Step 1: Read the file around line 2763**

Run: `sed -n '2760,2770p' /home/dirk/Sync/vg/src/varwg/time_series_analysis/distributions.py`

Expected: Shows imports like `from varwg import base, vg_plotting`

**Step 2: Update the import statement**

Replace:
```python
from varwg import base, vg_plotting
```

With:
```python
from varwg.core import base, plotting
```

Note: This fixes both the module path (should be `varwg.core` not `varwg`) and the module name (`plotting` not `vg_plotting`).

**Step 3: Verify imports work**

Run: `python3 -c "from varwg.time_series_analysis import distributions; print('✓ distributions import works')"`

Expected: Import succeeds without errors.

**Step 4: Commit**

```bash
git add src/varwg/time_series_analysis/distributions.py
git commit -m "fix: update imports in distributions.py to use plotting module"
```

---

## Task 3: Fix Cython references in src/varwg/time_series_analysis/cresample.pyx

**Files:**
- Modify: `src/varwg/time_series_analysis/cresample.pyx` (vg.vg_base references)

**Step 1: Find all vg.vg_base references**

Run: `grep -n "vg\.vg_base" /home/dirk/Sync/vg/src/varwg/time_series_analysis/cresample.pyx`

Expected: Shows lines referencing `vg.vg_base.conf` or similar

**Step 2: Read the file to understand context**

Run: `grep -B2 -A2 "vg\.vg_base" /home/dirk/Sync/vg/src/varwg/time_series_analysis/cresample.pyx`

Expected: Shows the context of these references.

**Step 3: Update all vg.vg_base references to varwg.core.base**

For each occurrence found, replace:
```
vg.vg_base
```

With:
```
varwg.core.base
```

Also update any:
```
vg.vg_plotting
```

To:
```
varwg.core.plotting
```

**Step 4: Verify the Cython syntax is correct**

Run: `head -50 /home/dirk/Sync/vg/src/varwg/time_series_analysis/cresample.pyx`

Expected: File shows proper Python import syntax and Cython declarations.

**Step 5: Commit**

```bash
git add src/varwg/time_series_analysis/cresample.pyx
git commit -m "fix: update vg.vg_base and vg.vg_plotting references to varwg.core in cresample.pyx"
```

---

## Task 4: Delete duplicate vg_base.py file and lock file

**Files:**
- Delete: `src/varwg/core/vg_base.py` (duplicate, base.py is the current file)
- Delete: `src/varwg/core/.#vg_base.py` (Emacs lock file)

**Step 1: Verify these files exist and are duplicates**

Run: `ls -la /home/dirk/Sync/vg/src/varwg/core/vg_base.py /home/dirk/Sync/vg/src/varwg/core/.#vg_base.py 2>&1`

Expected: Both files exist.

**Step 2: Verify base.py is the current file**

Run: `ls -lh /home/dirk/Sync/vg/src/varwg/core/base.py /home/dirk/Sync/vg/src/varwg/core/vg_base.py`

Expected: base.py should be the actively used file (likely larger or more recent in some attribute).

**Step 3: Delete the duplicate files**

```bash
git rm src/varwg/core/vg_base.py
rm -f /home/dirk/Sync/vg/src/varwg/core/.#vg_base.py
```

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove duplicate vg_base.py and editor lock file"
```

---

## Task 5: Update tox.ini to reference varwg instead of vg

**Files:**
- Modify: `tox.ini:18` (pytest command)

**Step 1: Read the tox.ini file**

Run: `cat /home/dirk/Sync/vg/tox.ini`

Expected: Shows pytest configuration with `pytest --pyargs vg` on line 18.

**Step 2: Update the pytest command**

Replace:
```
pytest --pyargs vg
```

With:
```
pytest --pyargs varwg
```

**Step 3: Verify tox.ini syntax is correct**

Run: `python3 -m py_compile /home/dirk/Sync/vg/tox.ini || echo "Note: tox.ini is not Python, this check is informational"`

Expected: File should be readable as configuration.

**Step 4: Commit**

```bash
git add tox.ini
git commit -m "chore: update tox.ini to reference varwg package name"
```

---

## Task 6: Update comment in test_vg.py

**Files:**
- Modify: `src/varwg/core/tests/test_vg.py` (outdated comment reference)

**Step 1: Find the comment reference**

Run: `grep -n "from varwg import vg_plotting" /home/dirk/Sync/vg/src/varwg/core/tests/test_vg.py`

Expected: Shows a commented-out line with old import.

**Step 2: Read context around the comment**

Run: `grep -B2 -A2 "from varwg import vg_plotting" /home/dirk/Sync/vg/src/varwg/core/tests/test_vg.py`

Expected: Shows the comment in context.

**Step 3: Update the comment**

Replace:
```python
#     # from varwg import vg_plotting
```

With:
```python
#     # from varwg.core import plotting
```

**Step 4: Verify test file still works**

Run: `python3 -m pytest /home/dirk/Sync/vg/src/varwg/core/tests/test_vg.py -v --tb=short -k "test_" --co`

Expected: pytest discovers tests successfully (output shows test collection, no execution).

**Step 5: Commit**

```bash
git add src/varwg/core/tests/test_vg.py
git commit -m "docs: update outdated import comment in test_vg.py"
```

---

## Task 7: Run full test suite and verify everything works

**Files:**
- None modified, verification only

**Step 1: Test basic imports**

Run: `python3 -c "from varwg.core import base; print('✓ base import works'); from varwg.core.core import VG; print('✓ VG import works'); from varwg.core import plotting; print('✓ plotting import works')"`

Expected: All three imports succeed.

**Step 2: Run the full test suite**

Run: `python3 -m pytest /home/dirk/Sync/vg/src/varwg/ -v --tb=short`

Expected: All tests pass (or show minimal known failures).

**Step 3: Check for any remaining vg references in code**

Run: `grep -r "import vg\b" /home/dirk/Sync/vg/src/varwg/ --include="*.py" --include="*.pyx" 2>/dev/null || echo "No 'import vg' statements found"`

Expected: No results or only expected/documented references.

**Step 4: Verify package can be imported from installed state**

Run: `cd /tmp && python3 -c "import sys; sys.path.insert(0, '/home/dirk/Sync/vg/src'); from varwg import __version__; print(f'✓ varwg package works, version info accessible')"`

Expected: Import succeeds and shows version info is accessible.

**Step 5: Final verification with pytest --pyargs**

Run: `python3 -m pytest --pyargs varwg -v --tb=short`

Expected: All tests pass.

**Step 6: Create summary**

Output a brief summary showing:
- Total commits made: [count]
- Test results: [PASS/FAIL and summary]
- Any remaining issues: [list or "None"]

No commit needed for this task - it's verification only.

---

## Success Criteria

✅ All 5 import fix commits completed
✅ Duplicate files deleted
✅ Configuration files updated
✅ Full test suite passes
✅ Basic imports work correctly
✅ No remaining references to vg.vg_base or vg.vg_plotting

---

## Estimated Time

- Task 1: 3-4 minutes
- Task 2: 2 minutes
- Task 3: 3-4 minutes
- Task 4: 2 minutes
- Task 5: 2 minutes
- Task 6: 2 minutes
- Task 7: 3-5 minutes

**Total: 17-24 minutes**

---

## Rollback Plan

If any task fails critically:
1. Run `git log --oneline` to see recent commits
2. Run `git revert HEAD` to undo the most recent commit
3. Investigate the issue and retry

All commits are atomic and independent, making rollback straightforward.
