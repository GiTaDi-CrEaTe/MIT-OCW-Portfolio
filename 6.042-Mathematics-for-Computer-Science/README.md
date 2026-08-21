# 6.042J — Mathematics for Computer Science

**MIT OCW subject:** 6.042J / 18.062J, taught jointly by EECS and Mathematics.

## What this course is

6.042 is the discrete-math backbone every other course in this repository leans on. It covers propositional and predicate logic, mathematical induction (including strong induction and well-ordering), graph theory, number theory (divisibility, GCDs, modular arithmetic, the foundations of RSA), counting and combinatorics, and an introduction to discrete probability. The through-line of the course is *proof technique* — every topic is really an excuse to practice building airtight arguments.

## Why it matters for this portfolio

Every later course assumes this material without restating it:
- 6.006's correctness proofs (loop invariants, exchange arguments) are induction in disguise.
- 6.041's discrete probability chapters are a direct continuation of 6.042's counting unit.
- RSA and hashing (used in 6.006 and touched on again here) depend entirely on the number theory covered in weeks 7–9 of 6.042.

## What I focused on

The `Applied-Theory/` implementation in this folder builds public-key cryptography (RSA) entirely from the number-theoretic primitives taught in the course: modular exponentiation by repeated squaring, the Extended Euclidean Algorithm for modular inverses, and the Miller–Rabin primality test (a randomized algorithm whose correctness proof itself depends on Fermat's Little Theorem and the structure of the multiplicative group mod a prime). I picked RSA specifically because it is the single cleanest demonstration that abstract number theory — statements about divisibility and congruence — has direct, load-bearing computational consequences.

## Folder contents

- [`Psets/pset_roadmap.md`](./Psets/pset_roadmap.md) — topic-by-topic breakdown of the syllabus and the problem sets I worked through.
- [`Applied-Theory/number_theory_cryptography.py`](./Applied-Theory/number_theory_cryptography.py) — RSA key generation, encryption, and decryption built from scratch, with a Miller–Rabin primality tester and Extended Euclid implementation, and an internal correctness/round-trip check.
