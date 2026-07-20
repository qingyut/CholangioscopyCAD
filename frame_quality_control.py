"""
Automated frame quality-control (QC) module for DSOC CADe/CADx.

Replaces manual blurred-frame exclusion. Runs upstream of YOLOv11 / ResNet18.
Three gates:
  1) Sharpness  -> variance of the Laplacian; reject if below tau
  2) Exposure   -> mean grayscale intensity must lie in [I_low, I_high]
  3) Glare      -> fraction of near-saturated pixels must be <= g

Thresholds (tau, I_low, I_high, g) are FIXED on an independent set of
manually labelled sharp/blurred frames (see `calibrate`), never on the test set.

Deps: opencv-python, numpy, scikit-learn (only for the calibration report).
Author: (add) — released at https://github.com/ZizhanT/dsoc-lesion-detection
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class QCThresholds:
    tau: float = 100.0          # variance-of-Laplacian sharpness threshold
    i_low: float = 25.0         # min mean intensity (under-exposed)
    i_high: float = 230.0       # max mean intensity (over-exposed)
    glare_frac: float = 0.15    # max fraction of near-saturated pixels
    glare_level: int = 250      # pixel value considered "saturated"


def frame_metrics(bgr: np.ndarray) -> dict:
    """Compute the three QC metrics for a single BGR frame."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_intensity = float(gray.mean())
    glare = float((gray >= 250).mean())
    return {"sharpness": sharpness, "mean_intensity": mean_intensity, "glare": glare}


def is_diagnostic_quality(bgr: np.ndarray, thr: QCThresholds) -> tuple[bool, dict]:
    """Return (keep?, metrics). keep=True means the frame passes QC."""
    m = frame_metrics(bgr)
    m["glare"] = float((cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) >= thr.glare_level).mean())
    keep = (
        m["sharpness"] >= thr.tau
        and thr.i_low <= m["mean_intensity"] <= thr.i_high
        and m["glare"] <= thr.glare_frac
    )
    return keep, m


def filter_video(path: str, thr: QCThresholds):
    """Yield only diagnostic-quality frames from a video, with their index."""
    cap = cv2.VideoCapture(path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        keep, _ = is_diagnostic_quality(frame, thr)
        if keep:
            yield idx, frame
        idx += 1
    cap.release()


# ---------------------------------------------------------------------------
# Calibration: fix tau (and optionally the gates) on a manually labelled set.
# labelled = list of (frame_bgr, keep_label) where keep_label==1 means the
# expert kept the frame (diagnostic quality) and 0 means it was excluded.
# ---------------------------------------------------------------------------
def calibrate(labelled, gate_thr: QCThresholds = QCThresholds()):
    """Pick the sharpness threshold tau that maximises agreement (accuracy)
    with expert keep/exclude labels, holding the exposure/glare gates fixed.
    Returns (best_tau, report_dict)."""
    from sklearn.metrics import cohen_kappa_score, accuracy_score

    sharp, labels, gate_pass = [], [], []
    for bgr, y in labelled:
        m = frame_metrics(bgr)
        g = float((cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) >= gate_thr.glare_level).mean())
        sharp.append(m["sharpness"])
        labels.append(int(y))
        gate_pass.append(
            gate_thr.i_low <= m["mean_intensity"] <= gate_thr.i_high
            and g <= gate_thr.glare_frac
        )
    sharp = np.asarray(sharp)
    labels = np.asarray(labels)
    gate_pass = np.asarray(gate_pass)

    best = (-1.0, None)
    for tau in np.percentile(sharp, np.arange(1, 100)):
        pred = ((sharp >= tau) & gate_pass).astype(int)
        acc = accuracy_score(labels, pred)
        if acc > best[0]:
            best = (acc, tau)
    best_acc, best_tau = best
    pred = ((sharp >= best_tau) & gate_pass).astype(int)
    report = {
        "best_tau": float(best_tau),
        "agreement_accuracy": float(best_acc),
        "cohen_kappa": float(cohen_kappa_score(labels, pred)),
        "retained_fraction": float(pred.mean()),
        "n_frames": int(len(labels)),
    }
    return best_tau, report


if __name__ == "__main__":
    # Minimal usage sketch (replace with your paths):
    #   thr = QCThresholds(tau=BEST_TAU_FROM_CALIBRATION)
    #   for idx, frame in filter_video("case001.mp4", thr):
    #       run_cade_cadx(frame)
    #
    #   # Calibration on manually labelled frames -> fills the [–] placeholders
    #   # in Methods 2.4.1 / Results (agreement %, kappa, timing):
    #   best_tau, report = calibrate(labelled_frames)
    #   print(report)
    print("Import this module; see docstring and Response to Reviewer 1, Rev. 3.")
