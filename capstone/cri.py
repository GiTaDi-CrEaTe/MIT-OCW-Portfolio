"""
Computational Reliability Index (CRI)
=====================================

Research Question:
  Can computational failure across different mathematical domains be characterized
  by a unified reliability model?

Mathematical Formulation:
  For any algorithm executing on physical hardware, we define the composite risk
  state vector as:
    x = (e_num, e_dec, v_assump, r_stab)

  where:
    1. e_num (numerical error):
       Relative deviation from exact algebraic or analytical value,
       e.g., ||Q^T Q - I||_2, |sigma_hat - sigma| / sigma, ||grad_num - grad_ana|| / ||grad||.
    2. e_dec (decision error):
       Discrete combinatorial suboptimality or decision penalty,
       e.g., (cost - cost_opt) / cost_opt, classification error on critical branches.
    3. v_assump (assumption violation):
       Degree of divergence from theoretical domain premises,
       e.g., condition number kappa / kappa_limit, heuristic overestimation factor, drift rate.
    4. r_stab (instability risk):
       Inverse stability margin: r_stab = 1 - m_stab, where m_stab in [0, 1] represents
       remaining distance to machine epsilon, rank deficiency, or numerical singular points.

  Weakest-Link Principle:
    A computational pipeline fails if ANY single component breaches its critical tolerance.
    We normalize each component against its calibrated critical tolerance tau_k:
      z_k = x_k / tau_k
      z = max_k (z_k)
