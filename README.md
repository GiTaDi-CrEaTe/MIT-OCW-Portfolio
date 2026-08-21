# MIT-OCW-Portfolio

A self-directed study of six MIT OpenCourseWare subjects spanning the theoretical foundations of computer science, artificial intelligence, and data science — discrete math, linear algebra, algorithms, probability, machine learning, and AI search/reasoning.

## Why this exists

Course grades and certificates don't show *how* someone thinks. This repository does. For each subject I worked through the real OCW syllabus, then rebuilt the core theoretical machinery **from first principles in raw Python/NumPy** — no `scikit-learn`, no `PyTorch`, no black-box solvers. If a pset topic was "prove convergence of gradient descent," the corresponding code here derives the gradient by hand and demonstrates the convergence empirically. The goal was never to get an answer; it was to be unable to hide from the math.

## How this repository is organized

Each course folder follows the same three-part structure:

```
<course-number>-<course-name>/
├── README.md                     # what the course covers, why it matters, what I took from it
├── Psets/
│   └── pset_roadmap.md           # digitized syllabus: topic-by-topic pset breakdown, difficulty notes
└── Applied-Theory/
    └── *.py                      # working implementation proving the theory, built from scratch
```

Every script in `Applied-Theory/` is:
- **Self-contained** — runs with only `numpy` (sometimes `matplotlib` for a diagnostic plot) installed.
- **Commented at the derivation level**, not the syntax level — comments explain *why* a mathematical step is taken, not what a Python line does.
- **Verified against a ground truth** — most scripts include an internal sanity check (e.g. comparing a from-scratch eigen-solver against `numpy.linalg.eig`, or a from-scratch gradient against a finite-difference approximation) so correctness isn't just claimed, it's tested in the code itself.

## Courses in this portfolio

| Course | Title | Core Theme |
|---|---|---|
| [6.042J](./6.042-Mathematics-for-Computer-Science) | Mathematics for Computer Science | Discrete math, induction, number theory, graph theory |
| [18.06](./18.06-Linear-Algebra) | Linear Algebra | Vector spaces, decompositions, eigenstructure |
| [6.006](./6.006-Introduction-to-Algorithms) | Introduction to Algorithms | Data structures, graph algorithms, asymptotic analysis |
| [6.041](./6.041-Probabilistic-Systems-Analysis) | Probabilistic Systems Analysis | Probability, estimation, stochastic processes |
| [6.036](./6.036-Introduction-to-Machine-Learning) | Introduction to Machine Learning | Optimization, supervised learning, neural networks |
| [6.034](./6.034-Artificial-Intelligence) | Artificial Intelligence | Search, adversarial reasoning, constraint satisfaction |

## A note on sequencing

The courses are listed above in the order I actually studied them, which matters: 6.042 and 18.06 are the load-bearing walls — discrete math gives the proof discipline, linear algebra gives the geometric/computational vocabulary. 6.006 and 6.041 build on both to get to algorithmic and stochastic reasoning. 6.036 and 6.034 are where those tools get pointed at learning and intelligent behavior. Building the repository in this order was deliberate: I did not want to implement backpropagation before I could derive it as repeated application of the chain rule, and I did not want to implement A* before I understood why a consistent heuristic guarantees optimality.

## What this repository is not

It is not a claim of novel research. It is a record of disciplined, from-scratch engagement with foundational material — the kind of raw mathematical fluency that original research eventually requires.
