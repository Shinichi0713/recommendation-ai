# What Is Rank?

The **rank of a matrix** can be described in one sentence as:

> **“The effective number of dimensions of information that the matrix contains.”**

At first glance, a matrix appears to be a large table containing $$n \times m$$ numbers. However, some rows or columns may contain **redundant information that can be expressed as combinations of others**.

The **rank** represents the number of **independent pieces of information that remain after removing such redundancy**.

---

# 1. Intuitive Understanding: The “Thickness” of Information

It becomes easier to understand the meaning of rank when we view a matrix as a **device that transforms space (a linear transformation)**.

* **Rank 3 (in 3D space):**
  The space is preserved as a three-dimensional object (full rank).

* **Rank 2:**
  A 3D object is compressed into a **plane (2D)**.

* **Rank 1:**
  All points are collapsed onto a **single line (1D)**.

* **Rank 0:**
  Everything collapses to the **origin (0D)** (this occurs only for the zero matrix).

---

# 2. Three Mathematical Definitions (All Equivalent)

The rank of a matrix has three equivalent definitions depending on perspective.

* **Column rank:**
  The maximum number of **linearly independent column vectors**.

* **Row rank:**
  The maximum number of **linearly independent row vectors**.

* (A remarkable fact is that the number of independent rows and columns is always the same.)

* **Dimension of the image:**
  The dimension of the space produced by the transformation $$A\mathbf{x}$$.

---

# 3. Why Is Rank Important?

Knowing the rank allows us to determine instantly whether certain mathematical problems are **solvable or not**.

---

## ① Existence of Solutions to Linear Systems

Consider the equation

$$
A\mathbf{x} = \mathbf{b}
$$

Whether this equation has a solution can be determined by comparing

* the rank of $$A$$, and
* the rank of the **augmented matrix** $$(A|\mathbf{b})$$.

If the rank is insufficient, it means that the system either

* contains contradictions, or
* lacks sufficient information.

---

## ② Existence of an Inverse Matrix

For an $$n \times n$$ square matrix:

> **“Having rank $$n$$ (full rank)” is equivalent to “having an inverse matrix.”**

If the rank is less than $$n$$, the transformation **collapses space in some direction**, making it impossible to reverse the transformation.

---

# 4. Computing the Rank: Row Reduction

When we perform **elementary row operations (Gaussian elimination)** to transform a matrix into row echelon form, the rank is simply

> **the number of rows that are not entirely zero.**

Example:

$$
\begin{pmatrix}
1 & 2 & 3 \
0 & 1 & 1 \
0 & 0 & 0
\end{pmatrix}
$$

Since the third row becomes all zeros, the **rank is 2**.

---

# 5. Difference Between $$\dim$$ and Rank

The concepts **$$\dim$$ (dimension)** and **$$\text{rank}$$** are very similar but refer to **different objects**.

In short:

* **$$\dim$$** refers to the size of the **space (the container)**.
* **$$\text{rank}$$** refers to the size of the **action (the transformation)**.

---

# 1. $$\dim$$ Is a Property of a Vector Space

$$\dim$$ represents the **minimum number of independent vectors required to span a vector space $$V$$** (the number of basis vectors).

* **Object:** Vector space
* **Meaning:** The extent or dimensionality of the space

Example:

The dimension of three-dimensional space:

$$
\dim(\mathbb{R}^3) = 3
$$

If $$W$$ is a plane within that space:

$$
\dim(W) = 2
$$

---

# 2. Rank Is a Property of a Matrix

$$\text{rank}$$ describes how much of the output space a matrix (or linear transformation) actually utilizes.

* **Object:** Matrix or linear transformation
* **Meaning:** How many dimensions of the output space can be produced

Example:

If a matrix maps 3D space into a plane, then

$$
\text{rank}(A) = 2
$$

Even a $$100 \times 100$$ matrix can have rank 1 if all rows are identical, because it can only transmit information in **one dimension**.

---

# 3. The Dimension Theorem

These concepts are closely connected through the **Rank–Nullity Theorem**.

For a linear transformation

$$
f: V \to W
$$

the following relationship holds:

$$
\dim(V) = \text{rank}(f) + \dim(\ker f)
$$

where

* $$\dim(V)$$ — the size of the original space
* $$\text{rank}(f)$$ — the number of dimensions that survive the transformation (dimension of the image)
* $$\dim(\ker f)$$ — the number of dimensions collapsed to zero (dimension of the kernel)

### Intuition

Imagine pressing a **3-dimensional lump of clay** onto a flat table:

* Original clay: $$\dim = 3$$
* Resulting disk: $$\text{rank} = 2$$
* Lost vertical direction: $$\dim(\ker f) = 1$$

---

# 4. Summary of the Difference

| Item         | $$\dim$$ (Dimension)       | Rank                                                    |
| ------------ | -------------------------- | ------------------------------------------------------- |
| Main object  | Vector space $$V$$         | Matrix $$A$$                                            |
| Definition   | Number of basis vectors    | Number of independent columns (or rows)                 |
| Intuition    | Dimensionality of the room | How many dimensions of influence the transformation has |
| Relationship | —                          | $$\text{rank}(A) = \dim(\text{Im} A)$$                  |

---

# Example Problem

## Relationship Between Rank and Dimension

Let $$A = (a_{ij})$$ be an $$m \times n$$ matrix, and let its column vectors be

$$
\mathbf{a}_1', \dots, \mathbf{a}_n'
$$

Prove that

$$
\text{rank}(A) = \dim S[\mathbf{a}_1', \dots, \mathbf{a}_n']
$$

---

# Proof

This statement shows that the **rank of a matrix** is exactly the **dimension of the space generated by its column vectors**.

---

## 1. Definitions and Setup

Let

* $$A$$ be an $$m \times n$$ matrix
* $$\mathbf{a}_1', \dots, \mathbf{a}_n' \in \mathbb{R}^m$$ be its column vectors
* $$S[\mathbf{a}_1', \dots, \mathbf{a}_n']$$ be the subspace generated by their linear combinations

This space is called the **column space**, denoted

$$
V_{col}
$$

---

## 2. Definition of Rank (Column Rank)

One definition of rank is:

> **The maximum number of linearly independent column vectors.**

Let that number be $$k$$.

$$
\text{rank}(A) = k
$$

---

## 3. Definition of Dimension

The dimension of a subspace is

> **the number of vectors in a basis.**

A **basis** is a set of vectors that

* are linearly independent, and
* can generate all vectors in the space.

---

## 4. Showing That the $$k$$ Vectors Form a Basis

Because $$\text{rank}(A) = k$$, we can select

$$
{\mathbf{a}*{i_1}', \dots, \mathbf{a}*{i_k}'}
$$

such that they are linearly independent.

1. **Linear independence**
   By definition, these $$k$$ vectors are independent.

2. **Spanning property**
   Since $$k$$ is the maximum number of independent columns, the remaining $$n-k$$ columns must be linear combinations of these $$k$$ vectors.

Therefore, the entire column space can be generated using only these $$k$$ vectors.

---

## 5. Conclusion

Thus these $$k$$ vectors form a **basis** of

$$
S[\mathbf{a}_1', \dots, \mathbf{a}_n']
$$

Therefore

$$
\dim S[\mathbf{a}_1', \dots, \mathbf{a}_n'] = k
$$

and since $$k = \text{rank}(A)$$,

$$
\text{rank}(A) =
\dim S[\mathbf{a}_1', \dots, \mathbf{a}_n']
$$

---

# Intuition: Why This Matters

This proof shows that

* **Rank (as a property of a matrix)** and
* **Dimension (as a property of a space)**

are essentially **two sides of the same coin**.

* **Matrix $$A$$:** a factory that transforms input vectors.
* **$$\text{rank}(A)$$:** the factory’s capability (how many independent products it can produce).
* **$$\dim S[\dots]$$:** the size of the showroom filled with those products.

Even if many column vectors are provided as materials, if they depend on each other (linearly dependent), the showroom dimension (rank) will not increase.

---

# Influence of Matrix Rank

When the rank of a matrix decreases, the solutions of the linear system

$$
A\mathbf{x} = \mathbf{b}
$$

change in predictable ways.

The key is comparing

$$
\text{rank}(A)
$$

and

$$
\text{rank}(A|\mathbf{b})
$$

---

# 1. Does a Solution Exist?

A solution exists if the target vector $$\mathbf{b}$$ lies inside the **image of $$A$$**.

This is determined by the **Kronecker–Capelli theorem**.

If

$$
\text{rank}(A) = \text{rank}(A|\mathbf{b})
$$

then **a solution exists**.

If

$$
\text{rank}(A) < \text{rank}(A|\mathbf{b})
$$

then **no solution exists**.

This means $$\mathbf{b}$$ lies outside the range of $$A$$.

---

# 2. How Many Solutions Exist?

If a solution exists, the number of solutions depends on the relationship between

* the number of variables $$n$$
* the rank

If

$$
\text{rank}(A) = n
$$

then the solution is **unique**.

If

$$
\text{rank}(A) < n
$$

then **infinitely many solutions** exist.

The number of degrees of freedom is

$$
\dim(\ker A) = n - \text{rank}(A)
$$

---

# 3. The Source of Freedom: The Kernel

If infinitely many solutions exist, they can be written as

$$
\mathbf{x}
==========

\mathbf{x}_p
+
(\text{an element of } \ker A)
$$

where

* $$\mathbf{x}_p$$ is a particular solution
* any vector in the kernel maps to zero under $$A$$

Thus adding kernel vectors does not change the result.

Therefore:

> **The larger the kernel dimension, the greater the number of solution variations.**

---

# Summary

The rank of a matrix quantifies its **ability to preserve spatial information**.

* If **Rank = dimension**, all information is preserved.
* If **Rank < dimension**, some directions are **collapsed**.

The number of collapsed dimensions corresponds exactly to

$$
\dim(\ker f)
$$

—the dimension of the kernel.
