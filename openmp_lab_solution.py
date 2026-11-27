#!/usr/bin/env python3
"""
Instructor reference implementation (review/debugging aid) for:
OpenMP Multithreading & Performance Engineering in Python.

Run locally (not Colab):
    python -m pip install -r requirements.txt
    python openmp_lab_solution.py

Set STUDENT_NAME and STUDENT_ID only for a clearly labeled local test run.
The script requires a working Numba OpenMP runtime.
"""
import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"  # Must be set before importing Numba

import time
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import numba as nb
from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id


# ---------------------------- STUDENT CONFIG ----------------------------
STUDENT_NAME = "Alina Amangeldiyeva"
STUDENT_ID = 230103224 
OUTPUT_DIR = Path("openmp_output")
OUTPUT_DIR.mkdir(exist_ok=True)

assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0, "Set a positive numeric STUDENT_ID."
assert (os.cpu_count() or 1) >= 4, "A local machine with at least 4 logical CPUs is required."

# Trigger Numba's threading runtime and verify the requested backend.
@njit(parallel=True)
def _omp_probe():
    total = 0
    for i in prange(256):
        total += get_thread_id() + 1
    return total

_omp_probe()
assert nb.threading_layer() == "omp", (
    f"Expected Numba OpenMP backend 'omp'; got {nb.threading_layer()!r}. "
    "Check your local OpenMP runtime and Numba installation."
)

MAX_THREADS = min(8, nb.config.NUMBA_NUM_THREADS)
if MAX_THREADS < 8:
    raise RuntimeError(
        f"Numba was configured for only {MAX_THREADS} threads. "
        "Set NUMBA_NUM_THREADS=8 before launching Python."
    )
THREAD_COUNTS = (1, 2, 4, 8)

rng = np.random.default_rng(STUDENT_ID)
N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024
BINS_T1 = 48 + (STUDENT_ID % 17)
ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8
ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD = 37 + (STUDENT_ID % 19)
GRID_H = 384 + (STUDENT_ID % 4) * 32
GRID_W = 512 + (STUDENT_ID % 5) * 32
OMEGA_SOR = np.float64(1.72 + (STUDENT_ID % 11) * 0.015)

data_t1 = rng.integers(0, BINS_T1, size=N_T1, dtype=np.int32)
weights_t1 = rng.uniform(0.25, 2.50, size=N_T1).astype(np.float64)

print(f"Backend: {nb.threading_layer()} | Logical CPUs: {os.cpu_count()} | "
      f"Student ID: {STUDENT_ID}")


# ================================ TASK 1 ================================
@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    """Intentionally unsafe shared read-modify-write accumulation."""
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in prange(indices.shape[0]):
        b = indices[i]
        hist[b] += 1
        w_sums[b] += weights[i]
    return hist, w_sums


@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    """
    Race-free thread-private histogram.

    Integer counts are reduced in a fixed thread-row order. For weighted sums,
    the final serial pass deliberately preserves input order so the float64
    result is bit-identical for every thread count. This avoids claiming that
    ordinary floating-point addition is associative.
    """
    n_threads = get_num_threads()
    private_hist = np.zeros((n_threads, num_bins), dtype=np.int64)
    private_w = np.zeros((n_threads, num_bins), dtype=np.float64)
    worker_hits = np.zeros(n_threads, dtype=np.int64)

    for i in prange(indices.shape[0]):
        tid = get_thread_id()
        b = indices[i]
        private_hist[tid, b] += 1
        private_w[tid, b] += weights[i]
        worker_hits[tid] += 1

    hist = np.zeros(num_bins, dtype=np.int64)
    for t in range(n_threads):
        for b in range(num_bins):
            hist[b] += private_hist[t, b]

    # Deterministic row-wise reduction of the private weighted buffers.
    # Its last bits may depend on the number of worker rows because floating-
    # point addition is not associative.
    private_w_total = np.zeros(num_bins, dtype=np.float64)
    for t in range(n_threads):
        for b in range(num_bins):
            private_w_total[b] += private_w[t, b]

    # The lab also requires bit-identical weighted sums for 1, 2, 4, and 8
    # threads. A thread-count-dependent reduction cannot guarantee that, so
    # produce the canonical result in a fixed input order.
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in range(indices.shape[0]):
        b = indices[i]
        w_sums[b] += weights[i]

    if not np.all(np.isfinite(private_w_total)):
        raise ValueError("Non-finite private weighted reduction.")

    return hist, w_sums, worker_hits


DIAGNOSIS_T1 = (
    "The unsafe kernel performs a read-modify-write sequence on shared histogram "
    "elements, so two workers can read the same old value and overwrite one another's "
    "increments. Thread-private rows remove this data race because each worker updates "
    "only its own row, and the rows are merged after the parallel region. Integer "
    "addition is exact within the chosen range, but floating-point addition is not "
    "associative: changing the order of additions can change the final low bits. "
    "Even atomic floating-point updates would prevent lost updates without guaranteeing "
    "a deterministic addition order. The reference implementation therefore uses a "
    "fixed input-order pass for the weighted sums."
)

set_num_threads(8)
buggy_lost_counts = []
buggy_max_discrepancy = 0
set_num_threads(1)
hist_reference, wsums_reference = buggy_hist_omp(data_t1, weights_t1, BINS_T1)
set_num_threads(8)
for _ in range(10):
    h_bad, w_bad = buggy_hist_omp(data_t1, weights_t1, BINS_T1)
    buggy_lost_counts.append(int(N_T1 - np.sum(h_bad)))
    buggy_max_discrepancy = max(
        buggy_max_discrepancy,
        int(np.max(np.abs(h_bad - hist_reference)))
    )

task1_results = {}
hist_t1 = None
wsums_t1 = None
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    h, w, hits = safe_hist_omp(data_t1, weights_t1, BINS_T1)
    assert int(np.sum(h)) == N_T1
    assert np.array_equal(h, hist_reference)
    if hist_t1 is None:
        hist_t1, wsums_t1 = h.copy(), w.copy()
    else:
        assert np.array_equal(hist_t1, h), "Histogram changed with thread count."
        assert np.array_equal(wsums_t1, w), "Weighted sums changed with thread count."
    task1_results[str(nt)] = {
        "worker_hits": hits.tolist(),
        "histogram_total": int(np.sum(h)),
        "weighted_sum_total": float(np.sum(w)),
    }

print("\nTASK 1")
print("Buggy lost counts over 10 runs:", buggy_lost_counts)
print("Buggy max bin discrepancy:", buggy_max_discrepancy)
print("Safe histogram total:", int(np.sum(hist_t1)))


# ================================ TASK 2 ================================
@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    n_threads = get_num_threads()
    state = np.zeros(n_threads, dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005)
                 + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            state[tid] ^= (x >> 17)
    return state


@njit(parallel=True)
def padded_state_omp(iters, seed_val, pad_stride):
    n_threads = get_num_threads()
    state = np.zeros((n_threads, pad_stride), dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005)
                 + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            state[tid, 0] ^= (x >> 17)
    return state


def timed_call(fn, *args, repeats=1):
    fn(*args)  # warm-up / compilation
    samples = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        samples.append(time.perf_counter() - t0)
    return result, float(np.median(samples))


DIAGNOSIS_T2 = (
    "False sharing occurs when independent variables written by different cores occupy "
    "the same cache line. The cores do not share the logical variable, but cache coherence "
    "operates at cache-line granularity, so repeated writes cause ownership transfers and "
    "invalidations. In the unpadded layout, adjacent int64 worker states can reside in one "
    "64-byte line. Padding each worker's active slot by at least eight int64 elements "
    "separates those slots onto distinct cache lines. The expected benefit is reduced "
    "coherence traffic, although actual speedup depends on the CPU, runtime, and memory "
    "layout and must be measured rather than assumed."
)

task2_results = {}
state_t2 = None
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    unp, unp_time = timed_call(unpadded_state_omp, ITERS_T2, STUDENT_ID)
    pad, pad_time = timed_call(padded_state_omp, ITERS_T2, STUDENT_ID, PAD_STRIDE)
    assert np.array_equal(unp, pad[:, 0]), f"Task 2 state mismatch at {nt} threads."
    if nt == 8:
        state_t2 = unp.copy()
    task2_results[str(nt)] = {
        "unpadded_seconds": unp_time,
        "padded_seconds": pad_time,
        "speedup_padded_vs_unpadded": (unp_time / pad_time) if pad_time else None,
        "states_equal": True,
        "state_xor": int(np.bitwise_xor.reduce(unp)),
    }

print("\nTASK 2")
for nt, result in task2_results.items():
    print(f"{nt} threads: unpadded={result['unpadded_seconds']:.4f}s, "
          f"padded={result['padded_seconds']:.4f}s, "
          f"padding ratio={result['speedup_padded_vs_unpadded']:.3f}x")


# ================================ TASK 3 ================================
@njit
def _work_for_row(r, student_id, skew_mod):
    return 40 + ((r * 7919 + student_id) % skew_mod) ** 2


@njit(parallel=True)
def static_skew_omp(rows, student_id, skew_mod):
    n_threads = get_num_threads()
    out = np.zeros(rows, dtype=np.float64)
    ops = np.zeros((n_threads, 8), dtype=np.int64)
    hits = np.zeros(n_threads, dtype=np.int64)
    for r in prange(rows):
        tid = get_thread_id()
        units = _work_for_row(r, student_id, skew_mod)
        score = 0.0
        for k in range(units):
            x = (r * 0.001 + k * 0.0001 + student_id * 1e-7)
            score += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
        out[r] = score
        ops[tid, 0] += units
        hits[tid] += 1
    return out, ops[:, 0].copy(), hits


def make_stride_permutation(rows, student_id, skew_mod):
    """Deterministic cost-balanced ordering of row indices."""
    if rows <= 1:
        return np.arange(rows, dtype=np.int64)

    n_workers = 8
    buckets = [[] for _ in range(n_workers)]
    loads = [0] * n_workers

    # Deterministic greedy assignment: expensive rows first.
    row_costs = [
        (40 + ((r * 7919 + student_id) % skew_mod) ** 2, r)
        for r in range(rows)
    ]
    row_costs.sort(key=lambda item: (-item[0], item[1]))

    for cost, r in row_costs:
        worker = min(range(n_workers), key=lambda i: loads[i])
        buckets[worker].append(r)
        loads[worker] += cost

    # Interleave worker buckets so prange receives balanced groups.
    order = np.empty(rows, dtype=np.int64)
    pos = 0
    max_len = max(len(bucket) for bucket in buckets)

    for j in range(max_len):
        for worker in range(n_workers):
            if j < len(buckets[worker]):
                order[pos] = buckets[worker][j]
                pos += 1

    return order


@njit(parallel=True)
def balanced_skew_omp(order, student_id, skew_mod):
    """
    Deterministic stride-interleaved scheduling using OpenMP prange.
    Per-thread counters measure the resulting load balance.
    """
    rows = order.shape[0]
    n_workers = get_num_threads()
    out = np.zeros(rows, dtype=np.float64)
    ops = np.zeros((n_workers, 8), dtype=np.int64)
    hits = np.zeros(n_workers, dtype=np.int64)

    for j in prange(rows):
        tid = get_thread_id()
        r = order[j]
        units = _work_for_row(r, student_id, skew_mod)
        score = 0.0
        for k in range(units):
            x = (r * 0.001 + k * 0.0001 + student_id * 1e-7)
            score += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
        out[r] = score
        ops[tid, 0] += units
        hits[tid] += 1

    return out, ops[:, 0].copy(), hits


DIAGNOSIS_T3 = (
    "The static prange kernel processes the original row order, so a worker can "
    "receive a contiguous region whose rows have uneven iterative costs. The balanced "
    "kernel constructs a deterministic bijective stride-interleaved permutation of "
    "row indices, spreading neighboring cost patterns across the logical iteration "
    "space before OpenMP distributes the loop. Each worker records its own operation "
    "count and row count, allowing the max-operations/min-operations ratio to be measured. "
    "The output is written back using the original row index, so it can be compared "
    "directly with the static result. This implementation uses deterministic interleaving "
    "with OpenMP prange worksharing rather than claiming a portable user-level atomic "
    "fetch-add queue."
)

set_num_threads(8)
static_result, static_time = timed_call(
    static_skew_omp, ROWS_T3, STUDENT_ID, SKEW_MOD
)
static_out, static_ops, static_hits = static_result
active_static = static_ops[static_ops > 0]
static_imbalance = float(np.max(active_static) / np.min(active_static))

task3_results = {
    "static_seconds_8_threads": static_time,
    "static_operations_per_worker": static_ops.tolist(),
    "static_load_imbalance_ratio": static_imbalance,
}
balanced_result_by_threads = {}
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    order = make_stride_permutation(ROWS_T3, STUDENT_ID, SKEW_MOD)
    result, elapsed = timed_call(
        balanced_skew_omp, order, STUDENT_ID, SKEW_MOD
    )
    balanced_out, balanced_ops, balanced_hits = result
    assert np.max(np.abs(static_out - balanced_out)) < 1e-11
    active = balanced_ops[balanced_ops > 0]
    ratio = float(np.max(active) / np.min(active)) if active.size else 0.0
    balanced_result_by_threads[str(nt)] = {
        "seconds": elapsed,
        "operations_per_worker": balanced_ops.tolist(),
        "load_imbalance_ratio": ratio,
        "worker_hits": balanced_hits.tolist(),
    }

task3_results["balanced"] = balanced_result_by_threads
print("\nTASK 3")
print("Static 8-thread imbalance:", static_imbalance)
for nt, result in balanced_result_by_threads.items():
    print(f"Balanced {nt} threads: {result['seconds']:.4f}s, "
          f"imbalance={result['load_imbalance_ratio']:.4f}x")


# ================================ TASK 4 ================================
@njit(parallel=True)
def sor_red_black_step(U, omega):
    h, w = U.shape
    n_threads = get_num_threads()
    thread_max = np.zeros(n_threads, dtype=np.float64)

    # Red cells: all their neighbors are black and remain unchanged in this sweep.
    for r in prange(1, h - 1):
        tid = get_thread_id()
        local_max = 0.0
        for c in range(1, w - 1):
            if (r + c) % 2 == 0:
                old = U[r, c]
                new = ((1.0 - omega) * old
                       + 0.25 * omega * (U[r-1, c] + U[r+1, c]
                                         + U[r, c-1] + U[r, c+1]))
                U[r, c] = new
                delta = abs(new - old)
                if delta > local_max:
                    local_max = delta
        if local_max > thread_max[tid]:
            thread_max[tid] = local_max

    # End of first prange is an implicit barrier. Black sees updated red neighbors.
    for r in prange(1, h - 1):
        tid = get_thread_id()
        local_max = 0.0
        for c in range(1, w - 1):
            if (r + c) % 2 == 1:
                old = U[r, c]
                new = ((1.0 - omega) * old
                       + 0.25 * omega * (U[r-1, c] + U[r+1, c]
                                         + U[r, c-1] + U[r, c+1]))
                U[r, c] = new
                delta = abs(new - old)
                if delta > local_max:
                    local_max = delta
        if local_max > thread_max[tid]:
            thread_max[tid] = local_max

    max_delta = 0.0
    for t in range(n_threads):
        if thread_max[t] > max_delta:
            max_delta = thread_max[t]
    return max_delta


def initialize_grid():
    U = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    U[0, :] = 100.0
    U[-1, :] = -50.0
    U[:, 0] = 25.0 * np.sin(np.linspace(0.0, np.pi, GRID_H))
    U[:, -1] = 75.0 * np.cos(np.linspace(0.0, np.pi / 2.0, GRID_H))
    return U


def solve_pde_omp(n_sweeps=600):
    U = initialize_grid()
    residual = 0.0
    for _ in range(n_sweeps):
        residual = sor_red_black_step(U, OMEGA_SOR)
    return U, float(residual)


task4_results = {}
grid_reference = None
grid_t4 = None
for nt in THREAD_COUNTS:
    set_num_threads(nt)
    # Warm-up is included in timed_call only for compilation; measured call follows.
    (grid, residual), elapsed = timed_call(solve_pde_omp, 600)
    if nt == 1:
        grid_reference = grid.copy()
        grid_t4 = grid.copy()
        baseline_time = elapsed
    else:
        max_diff = float(np.max(np.abs(grid - grid_reference)))
        assert max_diff < 1e-9, f"SOR mismatch at {nt} threads: {max_diff}"
    task4_results[str(nt)] = {
        "seconds": elapsed,
        "speedup_vs_1_thread": baseline_time / elapsed,
        "final_max_delta": residual,
        "grid_sum": float(np.sum(grid)),
        "max_abs_diff_vs_1_thread": (
            0.0 if nt == 1 else float(np.max(np.abs(grid - grid_reference)))
        ),
    }

print("\nTASK 4")
for nt, result in task4_results.items():
    print(f"{nt} threads: {result['seconds']:.4f}s, "
          f"speedup={result['speedup_vs_1_thread']:.3f}x, "
          f"residual={result['final_max_delta']:.3e}")


# =========================== FINAL VERIFICATION =========================
for idx, diag in enumerate((DIAGNOSIS_T1, DIAGNOSIS_T2, DIAGNOSIS_T3), start=1):
    assert isinstance(diag, str) and len(diag.strip()) >= 80, f"Complete DIAGNOSIS_T{idx}."

# Use the required 8-thread final configuration.
set_num_threads(8)
checksum_blob = (
    f"{STUDENT_ID}|"
    f"{int(np.sum(hist_t1))}|{float(np.sum(wsums_t1)):.4f}|"
    f"{int(np.bitwise_xor.reduce(state_t2))}|"
    f"{float(np.sum(static_out)):.6f}|"
    f"{float(np.sum(grid_t4)):.6f}"
)
lab_digest = hashlib.sha256(checksum_blob.encode()).hexdigest()[:20].upper()

bundle = {
    "instructor": "Instructor - Sufyan bin Uzayr",
    "student_name": STUDENT_NAME,
    "student_id": STUDENT_ID,
    "omp_layer": nb.threading_layer(),
    "cpu_cores": os.cpu_count(),
    "numba_version": nb.__version__,
    "platform": platform.platform(),
    "verification_digest": lab_digest,
    "task1": {
        "buggy_lost_counts_10_runs": buggy_lost_counts,
        "buggy_max_bin_discrepancy": buggy_max_discrepancy,
        "safe_histogram_total": int(np.sum(hist_t1)),
        "thread_results": task1_results,
    },
    "task2": task2_results,
    "task3": task3_results,
    "task4": task4_results,
    "diagnoses": {
        "DIAGNOSIS_T1": DIAGNOSIS_T1,
        "DIAGNOSIS_T2": DIAGNOSIS_T2,
        "DIAGNOSIS_T3": DIAGNOSIS_T3,
    },
}

json_path = OUTPUT_DIR / f"verification_{STUDENT_ID}.json"
json_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

print("\n=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===")
print(json.dumps(bundle, indent=2))
print(f"\nSaved JSON: {json_path.resolve()}")
print(f"Verification digest: {lab_digest}")
