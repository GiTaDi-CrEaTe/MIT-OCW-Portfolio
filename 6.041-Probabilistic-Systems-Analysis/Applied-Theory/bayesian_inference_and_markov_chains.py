"""
6.041 Applied Theory  --  Bayesian Inference, MLE, and Markov Chains from Scratch
================================================================================

Three pieces, chained to make a single argument: how belief should update
under evidence (Bayes), how that relates to the "just count and divide"
frequentist estimate (MLE), and how probabilistic models evolve over time
(Markov chains). Only `numpy` and `random` are used, purely as array/RNG
tools -- every probability computation is hand-derived.
"""

import numpy as np
import random

np.set_printoptions(precision=4, suppress=True)


# ===========================================================================
# PART 1  --  Bayesian inference on a discrete hypothesis space (Pset 7)
# ===========================================================================

