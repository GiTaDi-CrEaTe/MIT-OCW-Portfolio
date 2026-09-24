# Peer Review Log

*Independent study based on publicly available MIT OpenCourseWare materials. Not affiliated with or endorsed by MIT.*

This document records external critiques of the methods, my responses, and the changes made as a result. I am actively seeking reviewers who will try to break the findings, not endorse them.

## How to Submit a Review

Instructions for potential reviewers:
- Open an issue on the repository
- Focus on the mathematical methods and code, not presentation
- The most useful feedback identifies a flaw I have not considered

## Review Format

Each review is documented as:
1. Reviewer concern
2. My response
3. Experiment added (if any)
4. Result
5. What changed

## Self-Critique #1: Threshold Selection in CRI

**Concern:** The critical thresholds ($\tau_k$) in the CRI formula were chosen by hand. For QR orthogonalization, I used $10^{-2}$. For SVD, $0.10$. For gradient verification, $10^{-3}$. Why these specific values? A different set of thresholds would produce different CRI scores. This makes CRI potentially arbitrary.

**My Response:** This was a serious vulnerability in the initial formulation. The thresholds were initially chosen based on what I considered "acceptable" error in each domain -- for example, orthogonality error above $10^{-2}$ means the $Q$ matrix is no longer usable for downstream computations. But "acceptable" was subjective.

**Experiment Added:** Built a holdout validation suite (`capstone/cri_validation.py`). Calibrated thresholds on Dataset A (the original six course experiments, $N=36$), then tested CRI's ability to predict failure on Dataset B (unseen algorithms and matrices never seen during calibration, $N=38$). Measured AUROC, F1, precision, recall, FPR, and Brier calibration score.

**Result:** On held-out Dataset B, the calibrated CRI achieved an AUROC of 0.9494, an F1 score of 0.8966, Precision of 0.8667, Recall of 0.9286, and an FPR of 0.0833. The Brier score was 0.0666.

