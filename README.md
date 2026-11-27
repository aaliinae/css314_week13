# OpenMP Python Lab — Instructor Reference

## Files
- `openmp_lab_solution.py` — full reference implementation for Tasks 1–4.
- `requirements.txt` — Python dependencies.
- `openmp_output/` — generated JSON verification bundle after a successful run.

## Local setup

Use a local Python installation and a virtual environment. Google Colab/cloud VMs
are excluded by the laboratory specification.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Before running, edit `STUDENT_NAME` and `STUDENT_ID` near the top of the script.

```bash
python openmp_lab_solution.py
```

If Numba was configured with fewer than eight threads, launch with:
```bash
NUMBA_NUM_THREADS=8 python openmp_lab_solution.py
```

The script checks that Numba selected the `omp` threading layer. If the assertion
fails, the local Python/Numba installation does not have a working OpenMP runtime.

## Important instructor notes

1. The unsafe histogram is intentionally racy. Lost updates are hardware/runtime
   dependent; a run can occasionally show zero lost updates. Do not require a
   particular nonzero count as a pass criterion.
2. The safe histogram uses per-thread private rows for the integer histogram.
   To guarantee bit-identical float64 weighted sums across thread counts, it performs
   the weighted accumulation in a fixed input order after the parallel region.
   A conventional parallel floating-point reduction is not generally bit-identical.
3. Task 3 uses deterministic longest-processing-time (LPT) chunk assignment and
   parallel worker buckets. This is a balanced deterministic scheduler, not a literal
   shared atomic fetch-add work-stealing queue. The PDF permits deterministic
   interleaving as an alternative, but if the grading rubric strictly requires a
   shared atomic cursor, this implementation should be treated as a documented
   alternative rather than an exact implementation of that optional mechanism.
4. Timing and speedup are machine-dependent. The script records measured values;
   it does not hard-code expected performance.
5. Task 4 runs 600 Red-Black SOR sweeps and checks agreement against the one-thread
   solution. The Red and Black `prange` loops are separated by an implicit barrier.

## Output
The JSON file is written to:
`openmp_output/verification_<STUDENT_ID>.json`

The verification digest is computed from the checksum fields specified in the
laboratory handout. It is unique to the selected student ID and computed results.
# css314_week13
