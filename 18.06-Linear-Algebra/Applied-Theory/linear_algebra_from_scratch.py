"""
18.06 Applied Theory  --  Decompositions and Eigenstructure from First Principles
================================================================================

Implements, from scratch, four ideas chained together exactly as they build on
each other in the course:

    Gram-Schmidt CGS & MGS (Pset 7)  -->  QR algorithm eigensolver (Pset 9)
                                     |
                    +----------------+----------------+
                    v                                 v
       From-scratch SVD (Pset 12)          PageRank power iteration (Pset 11)

`numpy` is used strictly as an array container and for basic arithmetic
(dot products, norms). Every *algorithm* -- Gram-Schmidt, the QR iteration,
the SVD construction, power iteration -- is hand-written. `numpy.linalg` is
used ONLY inside the self-tests, as a ground-truth oracle to check this code
against, never as part of the implementation itself.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# 1. Gram-Schmidt orthogonalization -> QR decomposition
# ---------------------------------------------------------------------------

