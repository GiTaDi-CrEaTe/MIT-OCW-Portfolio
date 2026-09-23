# 6.042J --  Mathematics for Computer Science
*Foundations Lab: Discrete Exactness and Number-Theoretic Invariants*

---

### Core Question
Why does modular arithmetic in integer rings $\mathbb{Z}/n\mathbb{Z}$ provide exact computational guarantees that continuous floating-point algorithms cannot match?

### The Mathematical Guarantee
- **Bézout's Identity:** If $\gcd(a, m) = 1$, there exist integers $x, y$ such that $a x + m y = 1$, guaranteeing that $a$ has a unique modular inverse $x \equiv a^{-1} \pmod m$.
- **Euler's Totient Theorem:** For any $m$ coprime to $n = p \cdot q$, $m^{\phi(n)} \equiv 1 \pmod n$, where $\phi(n) = (p-1)(q-1)$.
- **RSA Correctness:** When $e \cdot d \equiv 1 \pmod{\phi(n)}$, then $(m^e)^d \equiv m^{1 + k\phi(n)} \equiv m \pmod n$ for all $m < n$.

### What the Code Investigates
[`number_theory_cryptography.py`](./Applied-Theory/number_theory_cryptography.py) builds the entire public-key cryptographic pipeline from first principles:
1. **Euclidean & Extended Euclidean Algorithm:** Computes $\gcd(a, b)$ and Bézout coefficients iteratively without recursion depth limits.
2. **Modular Exponentiation by Repeated Squaring:** Computes $b^e \pmod m$ in $O(\log e)$ operations without materializing astronomically large intermediate integers.
3. **Miller-Rabin Randomized Primality Test:** Uses Fermat's Little Theorem and roots of unity to test large odd integers with failure probability bounded by $4^{-k}$.
4. **RSA Key Generation, Encryption, and Decryption:** Implements the complete theory-to-implementation pipeline.

### Empirical Findings & Failure Modes
- **Zero Roundoff Error:** In contrast to floating-point linear algebra, integer arithmetic on 512-bit keys exhibits **zero precision drift** ($|m_{\text{recovered}} - m_{\text{original}}| = 0$).
- **Non-Constant Time Arithmetic:** In Python, arbitrary-precision integer arithmetic executes in variable time, revealing that mathematical correctness does not guarantee cryptographic side-channel immunity.
- **Failure Mode with Composite Moduli in Inverses:** Attempting to find modular inverse when $\gcd(a, m) > 1$ raises `ValueError`, verifying Bézout's precondition.

### Capstone & Cross-Course Connection
In the [Capstone Synthesis](../capstone/README.md#experiment-6--cross-course-synthesis-connecting-the-six-disciplines), 6.042's exact integer arithmetic is directly contrasted with 18.06's floating-point cancellation. While 6.042 integer arithmetic guarantees $(m^e)^d \equiv m \pmod n$ with $0$ error, float64 addition fails basic associativity: $(10^{16} + 1.0) - 10^{16} - 1.0 = -1.0$.
