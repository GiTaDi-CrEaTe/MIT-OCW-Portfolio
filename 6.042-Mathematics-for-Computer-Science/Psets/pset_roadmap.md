# 6.042J — Problem Set Roadmap

A digitized syllabus tracking the topics covered, the corresponding problem set focus, and my own notes on what was conceptually hardest. Structured to mirror the real OCW unit breakdown (Proofs → Structures → Counting → Probability).

## Unit 1 — Proofs

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 1 | Propositional logic, predicate logic | Truth tables, quantifier manipulation, De Morgan's laws for quantifiers | Easy to compute, harder to write *readably* — spent most effort on proof hygiene, not correctness. |
| Pset 2 | Direct proof, proof by contrapositive, proof by contradiction | Choosing the right proof strategy per claim shape | The contrapositive vs. contradiction distinction seems cosmetic until you hit a claim where contradiction produces a much messier argument. |
| Pset 3 | Mathematical induction | Weak induction, strong induction, well-ordering principle, structural induction on trees | The well-ordering principle equivalence to induction was the first "wait, these are the same tool" moment in the course. |
| Pset 4 | Induction applications | Correctness of recursive algorithms, invariants | Directly reused in 6.006 for loop-invariant proofs. |

## Unit 2 — Structures

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 5 | Number theory I | Divisibility, GCD, the Euclidean Algorithm | Foundation for the Applied-Theory script in this folder. |
| Pset 6 | Number theory II | Modular arithmetic, multiplicative inverses, Fermat's Little Theorem, Euler's theorem | Fermat's Little Theorem is the correctness argument underneath both RSA and Miller–Rabin — worth over-studying. |
| Pset 7 | Relations and functions | Equivalence relations, partial orders, bijections | Groundwork for later cardinality arguments. |
| Pset 8 | Graph theory I | Graph definitions, walks, trees, Euler tours/circuits | Directly reused in 6.006's graph-algorithms unit and again in 6.034's search unit. |
| Pset 9 | Graph theory II | Graph coloring, planarity, matching | Bipartite matching resurfaces in 6.034's CSP unit as a special case of constraint satisfaction. |

## Unit 3 — Counting

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 10 | Sums, products, asymptotics | Bounding sums, big-O intuition before it's formalized in 6.006 | This pset is where "counting carefully" starts to feel like algorithm analysis. |
| Pset 11 | Counting I | Permutations, combinations, the pigeonhole principle | Pigeonhole shows up again, disguised, in hashing collision arguments (6.006). |
| Pset 12 | Counting II | Binomial coefficients, inclusion-exclusion | Inclusion-exclusion is the hardest counting tool to apply correctly under time pressure. |

## Unit 4 — Discrete Probability

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 13 | Events and probability spaces | Discrete probability axioms, conditioning | Direct on-ramp into 6.041, which picks up exactly here and formalizes it. |
| Pset 14 | Random variables, expectation | Linearity of expectation, indicator random variables | Linearity of expectation without independence assumptions is the single most reused trick across this whole portfolio. |

## Applied-Theory connection

The RSA implementation in `Applied-Theory/number_theory_cryptography.py` draws directly on Psets 5–6: the Euclidean Algorithm (for GCDs and, extended, for modular inverses), Fermat's Little Theorem (correctness of Miller–Rabin and of RSA's decryption step), and modular exponentiation (efficient computation using the repeated-squaring idea introduced when bounding computation costs in Pset 10).
