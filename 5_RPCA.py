"""
RPCA-based synergy extraction for tensors shaped:
    (Tasks, Reps, Frames, Landmarks, Coords)  — one subject, or
    (Subjects, Tasks, Reps, Frames, Landmarks, Coords)  — multiple subjects.

Notation:
    S = subjects (optional leading axis)
    T = tasks
    R = repetitions
    F = frames (resampled clip length)
    L = landmarks
    C = coordinates

Legacy files with a singleton time-bin axis U=1 are accepted:
    (T,R,1,F,L,C) or (S,T,R,1,F,L,C) and are squeezed automatically.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def rpca_pcp(
    X: np.ndarray,
    lam: float | None = None,
    mu: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Principal Component Pursuit (RPCA) via inexact ALM.
    Decomposes X = L + S (low-rank + sparse).
    """
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    m, n = X.shape
    try:
        norm_two = np.linalg.norm(X, 2)
    except np.linalg.LinAlgError:
        norm_two = np.linalg.norm(X, "fro")
    norm_inf = np.linalg.norm(X, np.inf) / max(1.0, np.sqrt(m * n))

    lam = float(lam) if lam is not None else 1.0 / np.sqrt(max(m, n))
    mu = float(mu) if mu is not None else 1.25 / (norm_two + 1e-12)
    mu_bar = mu * 1e7

    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = X / max(norm_two, norm_inf, 1e-12)

    for _ in range(max_iter):
        # 1) Singular value thresholding for L
        U, s, Vt = np.linalg.svd(X - S + (1.0 / mu) * Y, full_matrices=False)
        s_thr = np.maximum(s - (1.0 / mu), 0.0)
        r = int(np.sum(s_thr > 0))
        if r > 0:
            L = (U[:, :r] * s_thr[:r]) @ Vt[:r, :]
        else:
            L = np.zeros_like(X)

        # 2) Soft-thresholding for S
        Q = X - L + (1.0 / mu) * Y
        S = np.sign(Q) * np.maximum(np.abs(Q) - lam / mu, 0.0)

        # 3) Dual ascent
        Z = X - L - S
        Y = Y + mu * Z
        mu = min(mu * 1.5, mu_bar)

        # 4) Convergence
        rel_err = np.linalg.norm(Z, "fro") / (np.linalg.norm(X, "fro") + 1e-12)
        if rel_err < tol:
            break

    return L, S


def _parse_index_arg(arg: str, max_len: int) -> np.ndarray:
    """
    Parse "all", "a:b", or "i,j,k" into a validated integer numpy array.
    """
    s = arg.strip().lower()
    if s == "all":
        idx = np.arange(max_len, dtype=int)
    elif ":" in s:
        a_str, b_str = s.split(":", 1)
        a = int(a_str.strip())
        b = int(b_str.strip())
        idx = np.arange(a, b, dtype=int)
    else:
        idx = np.array([int(x.strip()) for x in s.split(",") if x.strip()], dtype=int)

    if idx.size == 0:
        raise ValueError(f"Empty index selection from arg={arg!r}")
    if np.any(idx < 0) or np.any(idx >= max_len):
        raise ValueError(f"Index out of range for arg={arg!r}; valid 0..{max_len-1}")
    return idx


def _choose_axis_indices(
    n: int,
    arg: str | None,
) -> np.ndarray:
    if arg is None:
        return np.arange(n, dtype=int)
    return _parse_index_arg(arg, n)


def stack_tensor_to_matrix(
    tensor: np.ndarray,
    task_indices: Iterable[int],
    reps: Iterable[int] | None = None,
    frame_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """
    Build a (features x samples) matrix from tensor (T,R,F,L,C).

    Features = L*C
    Samples  = (#tasks * #reps * #frames)
    """
    if tensor.ndim != 5:
        raise ValueError(f"Expected tensor with 5 dims (T,R,F,L,C), got {tensor.shape}")

    T, R, F, L, C = tensor.shape
    task_idx = np.array(list(task_indices), dtype=int)
    rep_idx = np.arange(R, dtype=int) if reps is None else np.array(list(reps), dtype=int)
    frame_idx = np.arange(F, dtype=int) if frame_indices is None else np.array(list(frame_indices), dtype=int)

    block = tensor[task_idx, ...]          # (T_sel, R, F, L, C)
    block = block[:, rep_idx, ...]         # (T_sel, R_sel, F, L, C)
    block = block[:, :, frame_idx, ...]    # (T_sel, R_sel, F_sel, L, C)

    samples = int(np.prod(block.shape[:3]))
    X = block.reshape(samples, L * C).T  # (features, samples)
    return X


def fit_standardizer(X: np.ndarray) -> dict[str, np.ndarray]:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return {"mu": mu, "sd": sd}


def apply_standardizer(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return (X - stats["mu"]) / stats["sd"]


def inverse_standardizer(Xn: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return Xn * stats["sd"] + stats["mu"]


def extract_synergy_basis_from_lowrank(
    L: np.ndarray,
    k: int | None = None,
    energy: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    SVD on low-rank component to get orthonormal synergy basis Uk (features x k).
    """
    U, s, _ = np.linalg.svd(L, full_matrices=False)
    if s.size == 0:
        return np.zeros((L.shape[0], 0)), np.array([]), 0.0

    if k is None:
        cum = np.cumsum(s**2)
        frac = cum / max(cum[-1], 1e-12)
        k = int(np.searchsorted(frac, energy) + 1)
    k = max(1, min(int(k), U.shape[1]))

    U_k = U[:, :k]
    explained = float(np.sum(s[:k] ** 2) / (np.sum(s**2) + 1e-12))
    return U_k, s[:k], explained


def reconstruct_with_basis(Uk: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Orthogonal projection: Y_hat = Uk Uk^T Y."""
    return Uk @ (Uk.T @ Y)


def nrmse(Y_true: np.ndarray, Y_hat: np.ndarray) -> float:
    Y_true = np.nan_to_num(Y_true, nan=0.0)
    Y_hat = np.nan_to_num(Y_hat, nan=0.0)
    num = np.linalg.norm(Y_true - Y_hat, "fro")
    den = np.linalg.norm(Y_true, "fro") + 1e-12
    return float(num / den)


def nrmse_per_task_basis(
    tensor: np.ndarray,
    Uk: np.ndarray,
    std_stats: dict[str, np.ndarray],
    reps: np.ndarray,
    frame_idx: np.ndarray,
) -> np.ndarray:
    """
    For each task index 0..T-1, NRMSE of synergy reconstruction (Uk, train standardizer)
    on that task's data only. Shape (T,).
    """
    if tensor.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got {tensor.shape}")
    T = tensor.shape[0]
    out = np.empty(T, dtype=np.float64)
    for t in range(T):
        X = stack_tensor_to_matrix(tensor, [t], reps=reps, frame_indices=frame_idx)
        X = np.nan_to_num(X, nan=0.0)
        Xn = apply_standardizer(X, std_stats)
        Xn_hat = reconstruct_with_basis(Uk, Xn)
        X_hat = inverse_standardizer(Xn_hat, std_stats)
        out[t] = nrmse(X, X_hat)
    return out


def nrmse_per_task_rep_basis(
    tensor: np.ndarray,
    Uk: np.ndarray,
    std_stats: dict[str, np.ndarray],
    reps: np.ndarray,
    frame_idx: np.ndarray,
) -> np.ndarray:
    """
    For each (task, rep) pair, NRMSE of synergy reconstruction on that rep only.
    Shape (T, len(reps)) — e.g. 20 tasks × 3 reps => 60 trial-level scores.
    """
    if tensor.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got {tensor.shape}")
    T = tensor.shape[0]
    rep_list = [int(x) for x in np.asarray(reps).tolist()]
    out = np.empty((T, len(rep_list)), dtype=np.float64)
    for t in range(T):
        for j, r in enumerate(rep_list):
            X = stack_tensor_to_matrix(tensor, [t], reps=[r], frame_indices=frame_idx)
            X = np.nan_to_num(X, nan=0.0)
            Xn = apply_standardizer(X, std_stats)
            Xn_hat = reconstruct_with_basis(Uk, Xn)
            X_hat = inverse_standardizer(Xn_hat, std_stats)
            out[t, j] = nrmse(X, X_hat)
    return out


def load_task_names(path: Path | None, n_tasks: int) -> list[str]:
    """One task name per line; pad or truncate to n_tasks."""
    if path is None or not path.is_file():
        return [f"task_{i}" for i in range(n_tasks)]
    lines = path.read_text(encoding="utf-8").splitlines()
    names = [ln.strip() for ln in lines if ln.strip() != ""]
    if len(names) < n_tasks:
        names = names + [f"task_{i}" for i in range(len(names), n_tasks)]
    elif len(names) > n_tasks:
        names = names[:n_tasks]
    return names


def process_data(
    tensor: np.ndarray,
    train_tasks: np.ndarray,
    test_tasks: np.ndarray,
    reps: np.ndarray,
    frame_idx: np.ndarray,
    k: int | None = None,
    energy: float = 0.95,
    rpca_max_iter: int = 1000,
    rpca_tol: float = 1e-7,
) -> dict[str, np.ndarray | float]:
    """
    Train on train_tasks, perform RPCA -> synergies, reconstruct test_tasks.
    """
    X_train = stack_tensor_to_matrix(tensor, train_tasks, reps=reps, frame_indices=frame_idx)
    X_test = stack_tensor_to_matrix(tensor, test_tasks, reps=reps, frame_indices=frame_idx)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    std_stats = fit_standardizer(X_train)
    Xn_train = apply_standardizer(X_train, std_stats)
    Xn_test = apply_standardizer(X_test, std_stats)

    L_tr, S_tr = rpca_pcp(Xn_train, max_iter=rpca_max_iter, tol=rpca_tol)
    Uk, svals, expl = extract_synergy_basis_from_lowrank(L_tr, k=k, energy=energy)

    Xn_train_hat = reconstruct_with_basis(Uk, Xn_train)
    X_train_hat = inverse_standardizer(Xn_train_hat, std_stats)
    train_nrmse = nrmse(X_train, X_train_hat)

    Xn_test_hat = reconstruct_with_basis(Uk, Xn_test)
    X_test_hat = inverse_standardizer(Xn_test_hat, std_stats)
    test_nrmse = nrmse(X_test, X_test_hat)
    nrmse_each_task = nrmse_per_task_basis(tensor, Uk, std_stats, reps, frame_idx)
    nrmse_per_trial = nrmse_per_task_rep_basis(tensor, Uk, std_stats, reps, frame_idx)

    return {
        "Uk": Uk,
        "svals": svals,
        "explained_energy": float(expl),
        "X_train_true": X_train,
        "X_train_hat": X_train_hat,
        "nrmse_train": float(train_nrmse),
        "X_test_true": X_test,
        "X_test_hat": X_test_hat,
        "nrmse_test": float(test_nrmse),
        "nrmse_per_task": nrmse_each_task,
        "nrmse_per_trial": nrmse_per_trial,
        "L_train": L_tr,
        "S_train": S_tr,
    }


def _squeeze_legacy_time_axis(tensor: np.ndarray) -> np.ndarray:
    """Drop singleton time-bin axis U=1 if present: (T,R,1,F,L,C)->(T,R,F,L,C)."""
    if tensor.ndim == 6 and tensor.shape[2] == 1:
        return tensor[:, :, 0, ...]
    return tensor


def _load_one_npy_to_subject_list(path: Path, subjects_arg: str) -> list[tuple[str, np.ndarray, Path]]:
    tensor = np.load(path)

    # Legacy 7D (S,T,R,U,F,L,C), U must be 1
    if tensor.ndim == 7:
        if tensor.shape[3] != 1:
            raise ValueError(
                f"Legacy 7D (S,T,R,U,F,L,C) requires U=1 at axis 3; got shape {tensor.shape}"
            )
        tensor = tensor[:, :, :, 0, ...]

    tensor = _squeeze_legacy_time_axis(tensor)

    if tensor.ndim == 5:
        return [("0", tensor, path)]

    if tensor.ndim == 6:
        # Multi-subject (S,T,R,F,L,C)
        S = tensor.shape[0]
        subj_idx = _parse_index_arg(subjects_arg, S)
        return [(str(int(i)), tensor[int(i)], path) for i in subj_idx]

    raise ValueError(
        f"Expected 5D (T,R,F,L,C) or 6D (S,T,R,F,L,C), "
        f"or legacy 6D/7D with U=1; got ndim={tensor.ndim} shape={tensor.shape}"
    )


def _iter_subject_tensors(
    npy_path: str | None,
    npy_dir: str | None,
    subjects_arg: str,
) -> list[tuple[str, np.ndarray, Path | None]]:

    bundles: list[tuple[str, np.ndarray, Path | None]] = []

    if npy_dir is not None:
        d = Path(npy_dir).expanduser().resolve()
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
        paths = sorted(d.glob("*.npy"))
        if not paths:
            raise FileNotFoundError(f"No .npy files in {d}")
        for p in paths:
            arr = np.load(p)
            arr = _squeeze_legacy_time_axis(arr)
            if arr.ndim != 5:
                raise ValueError(
                    f"Each subject file must be 5D (T,R,F,L,C) "
                    f"(or legacy 6D with U=1); got {arr.shape} in {p}"
                )
            bundles.append((p.stem, arr, p))
        return bundles

    if npy_path is None:
        raise ValueError("Provide --npy_path or --npy_dir")

    return _load_one_npy_to_subject_list(Path(npy_path).expanduser().resolve(), subjects_arg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RPCA synergies on 5D (T,R,F,L,C) or 6D subject stack (S,T,R,F,L,C); per-subject metrics.",
        epilog=(
            "Each subject: Subject_<id>.npz; Subject_<id>_metrics.csv "
            "(subject_idx, nrmse_train, nrmse_test, explained_energy, k, train_task_names, test_task_names); "
            "Subject_<id>_metrics_per_task.csv (one NRMSE per ADL, pooling reps); "
            "Subject_<id>_metrics_per_trial.csv (subject_idx, task_name, task_split, nrmse).\n"
            "Optional task names: place <stem>.tasks.txt next to the .npy or pass --tasks_txt.\n"
            "Examples:\n"
            "  Single 5D file:  python 5_RPCA.py --npy_path one_subject.npy --out_dir out\n"
            "  Folder of 5D:    python 5_RPCA.py --npy_dir ./subject_tensors --out_dir out\n"
            "  Stacked 6D:      python 5_RPCA.py --npy_path all.npy --subjects all --out_dir out\n"
            "  (Build stack: np.save('all.npy', np.stack([a,b,c], axis=0)))"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--npy_path",
        type=str,
        default=None,
        help="Path to .npy: 5D one subject, 6D (S,T,R,F,L,C), or legacy with U=1",
    )
    src.add_argument(
        "--npy_dir",
        type=str,
        default=None,
        help="Directory of per-subject 5D tensors (*.npy); subject id = filename stem",
    )
    p.add_argument(
        "--subjects",
        type=str,
        default="all",
        help='Only when --npy_path loads 6D (S,T,R,F,L,C): subject indices, e.g. "all", "0:3"',
    )
    p.add_argument("--out_dir", type=str, default="rpca_results", help="Directory to save results")
    p.add_argument(
        "--train_tasks",
        type=str,
        default="0:16",
        help='Task indices for training: "all", "a:b", or "i,j,k" (default 0:16 for 20-task IPN-style split)',
    )
    p.add_argument(
        "--test_tasks",
        type=str,
        default="16:20",
        help='Task indices for testing: "all", "a:b", or "i,j,k" (default 16:20 with 20 tasks)',
    )
    p.add_argument("--reps", type=str, default="all", help='Rep indices: "all", "a:b", or "i,j,k"')
    p.add_argument("--frames", type=str, default="all", help='Frame indices: "all", "a:b", or "i,j,k"')
    p.add_argument("--k", type=int, default=None, help="Number of synergies; if omitted, use --energy")
    p.add_argument("--energy", type=float, default=0.95, help="Energy threshold for choosing k")
    p.add_argument("--rpca_max_iter", type=int, default=1000)
    p.add_argument("--rpca_tol", type=float, default=1e-7)
    p.add_argument(
        "--tasks_txt",
        type=str,
        default=None,
        help="Optional file with one task name per line (order matches task index 0..T-1). "
        "If omitted, uses <npy_path>.tasks.txt when present, or subject npy's sibling .tasks.txt.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    bundles = _iter_subject_tensors(args.npy_path, args.npy_dir, args.subjects)
    if len(bundles) > 1:
        print(f"Processing {len(bundles)} subject(s).")

    # Validate shared task/rep/frame layout from first tensor
    _, first, _ = bundles[0]
    T, R, F, L, C = first.shape
    print(f"Per-subject tensor shape (T,R,F,L,C)={first.shape}; features_per_frame L*C={L * C}")

    train_tasks = _parse_index_arg(args.train_tasks, T)
    test_tasks = _parse_index_arg(args.test_tasks, T)
    reps = _parse_index_arg(args.reps, R)
    frame_idx = _choose_axis_indices(F, args.frames)
    train_set = set(int(x) for x in train_tasks.tolist())
    test_set = set(int(x) for x in test_tasks.tolist())

    def _split_label(ti: int) -> str:
        if ti in train_set:
            return "train"
        if ti in test_set:
            return "test"
        return "none"

    for subj_id, tensor, bundle_npy in bundles:
        if tensor.shape != (T, R, F, L, C):
            raise ValueError(
                f"Subject {subj_id}: shape {tensor.shape} != first subject {(T, R, F, L, C)}"
            )

        res = process_data(
            tensor=tensor,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            reps=reps,
            frame_idx=frame_idx,
            k=args.k,
            energy=args.energy,
            rpca_max_iter=args.rpca_max_iter,
            rpca_tol=args.rpca_tol,
        )

        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(subj_id))
        out_npz = os.path.join(args.out_dir, f"Subject_{safe}.npz")
        np.savez_compressed(
            out_npz,
            subject_id=subj_id,
            Uk=res["Uk"],
            svals=res["svals"],
            explained_energy=res["explained_energy"],
            nrmse_per_task=res["nrmse_per_task"],
            nrmse_per_trial=res["nrmse_per_trial"],
            X_train_true=res["X_train_true"],
            X_train_hat=res["X_train_hat"],
            nrmse_train=res["nrmse_train"],
            X_test_true=res["X_test_true"],
            X_test_hat=res["X_test_hat"],
            nrmse_test=res["nrmse_test"],
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            reps=reps,
            frames=frame_idx,
        )

        k_val = int(res["Uk"].shape[1]) if isinstance(res["Uk"], np.ndarray) else 0

        tasks_path: Path | None = None
        if args.tasks_txt:
            tasks_path = Path(args.tasks_txt).expanduser().resolve()
            if not tasks_path.is_file():
                raise FileNotFoundError(f"--tasks_txt not found: {tasks_path}")
        elif bundle_npy is not None:
            cand = bundle_npy.with_suffix(".tasks.txt")
            if cand.is_file():
                tasks_path = cand

        task_names = load_task_names(tasks_path, T)

        def _safe_task_name(i: int) -> str:
            return task_names[int(i)].replace(",", ";")

        train_task_names = " | ".join(_safe_task_name(i) for i in train_tasks.tolist())
        test_task_names = " | ".join(_safe_task_name(i) for i in test_tasks.tolist())

        try:
            subject_idx_val: int | str = int(str(subj_id).strip())
        except ValueError:
            subject_idx_val = str(subj_id)

        metrics_csv = os.path.join(args.out_dir, f"Subject_{safe}_metrics.csv")
        with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "subject_idx",
                    "nrmse_train",
                    "nrmse_test",
                    "explained_energy",
                    "k",
                    "train_task_names",
                    "test_task_names",
                ]
            )
            w.writerow(
                [
                    subject_idx_val,
                    f"{res['nrmse_train']:.8f}",
                    f"{res['nrmse_test']:.8f}",
                    f"{res['explained_energy']:.8f}",
                    k_val,
                    train_task_names,
                    test_task_names,
                ]
            )
        per_task_csv = os.path.join(args.out_dir, f"Subject_{safe}_metrics_per_task.csv")
        with open(per_task_csv, "w", encoding="utf-8") as f:
            f.write("task_index,task_name,split,nrmse\n")
            for ti in range(T):
                safe_name = task_names[ti].replace(",", ";")
                f.write(
                    f"{ti},{safe_name},{_split_label(ti)},{res['nrmse_per_task'][ti]:.8f}\n"
                )

        per_trial_csv = os.path.join(args.out_dir, f"Subject_{safe}_metrics_per_trial.csv")
        n_trials = int(res["nrmse_per_trial"].size)
        with open(per_trial_csv, "w", encoding="utf-8", newline="") as f:
            wt = csv.writer(f)
            wt.writerow(["subject_idx", "task_name", "task_split", "nrmse"])
            rep_axes = [int(x) for x in reps.tolist()]
            for ti in range(T):
                base_name = _safe_task_name(ti)
                for j, r_axis in enumerate(rep_axes):
                    rep_num = int(r_axis) + 1
                    trial_name = base_name if rep_num == 1 else f"{base_name}{rep_num}"
                    wt.writerow(
                        [
                            subject_idx_val,
                            trial_name,
                            _split_label(ti),
                            f"{res['nrmse_per_trial'][ti, j]:.8f}",
                        ]
                    )

        npt = res["nrmse_per_task"]
        print(
            f"[{subj_id}] synergies: explained={100.0 * res['explained_energy']:.2f}% | k={k_val} -> {out_npz}"
        )
        print(
            f"         per-task NRMSE: min={float(np.min(npt)):.4f} max={float(np.max(npt)):.4f} "
            f"(see {per_task_csv})"
        )
        print(f"         per-trial NRMSE: {n_trials} rows -> {per_trial_csv}")
        print(f"         model summary -> {metrics_csv}")

    print(
        f"Done. Per-task / per-trial CSVs in {args.out_dir} "
        f"(Subject_<id>_metrics_per_task.csv, Subject_<id>_metrics_per_trial.csv)"
    )


if __name__ == "__main__":
    main()
