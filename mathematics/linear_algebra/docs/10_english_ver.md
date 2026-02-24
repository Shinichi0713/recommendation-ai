A matrix is a linear map.
However, even if one hears this statement, it is often difficult to immediately grasp what it truly means.

Today, we will explain the concept of a **linear map**, which is one of the most important ideas in linear algebra.

---

# Linear Maps

In one sentence, a **linear map** is **a straight transformation that does not break the rules of space (the parallelism and equal spacing of a grid).**

The word “map” may sound abstract, but every computation performed using matrices corresponds to a linear map. Let us unpack what this means from three perspectives.

---

## 1. Geometric Image: Deforming a Grid

Imagine a sheet of 2D graph paper. A linear map corresponds to operations such as “stretching,” “rotating,” or “compressing” that sheet. However, it must strictly obey the following rules:

1. **Lines remain lines:** You may not bend or fold them.
2. **The origin does not move:** The center point $(0,0)$ always remains $(0,0)$.
3. **Parallel lines remain parallel:** Equally spaced grid lines remain equally spaced after transformation.

---

## 2. Mathematical Definition: Two Properties

Formally, a map $f$ is called linear if it satisfies the following two conditions (linearity):

**Additivity**

$$
f(\mathbf{u} + \mathbf{v}) = f(\mathbf{u}) + f(\mathbf{v})
$$

(Add first, then transform—or transform first, then add—the result is the same.)

**Homogeneity (Scalar Multiplication)**

$$
f(c\mathbf{u}) = c f(\mathbf{u})
$$

(Multiply by $c$ first, then transform—or transform first, then multiply by $c$—the result is the same.)

These properties imply a highly predictable behavior:
**The scale and combination of inputs are proportionally reflected in the outputs.**

---

## 3. What Types of Operations Are Included?

Matrix multiplication can implement the following transformations:

| Type           | Description                                 |
| -------------- | ------------------------------------------- |
| **Scaling**    | Stretching space vertically or horizontally |
| **Rotation**   | Rotating around the origin                  |
| **Shearing**   | Distorting a rectangle into a parallelogram |
| **Projection** | Flattening 3D into 2D (casting a shadow)    |
| **Reflection** | Mirroring across a line                     |

---

## 4. Why Are Linear Maps Important?

Analyzing complex real-world phenomena (which are nonlinear) is extremely difficult. However, when viewed in sufficiently small regions, many phenomena can be approximated as linear.

* **3D Graphics:** Character motion and camera transformations are linear maps implemented by matrices.
* **AI / Machine Learning:** The main computation in each neural network layer is a large linear map.
* **Physics:** Vibrations, rotations, and even quantum state evolution are described using linear maps.

---

# Mathematical Formulation

A linear map is fundamentally a **structure-preserving transformation**.

More precisely, a map $f : V \to W$ between two vector spaces is defined as linear if it satisfies the following two conditions:

## 1. The Two Defining Conditions

For any $\mathbf{u}, \mathbf{v} \in V$ and any scalar $c$:

### (1) Additivity

$$
f(\mathbf{u} + \mathbf{v}) = f(\mathbf{u}) + f(\mathbf{v})
$$

Meaning: transforming the sum equals summing the transformations.

### (2) Homogeneity

$$
f(c\mathbf{u}) = c f(\mathbf{u})
$$

Meaning: scaling before transformation equals scaling after transformation.

These two properties are sometimes combined into a single expression:

$$
f(c\mathbf{u} + d\mathbf{v}) = c f(\mathbf{u}) + d f(\mathbf{v})
$$

(This is known as the **principle of superposition**.)

---

## 2. Important Consequences

Two essential properties follow directly from the definition:

1. **The origin always maps to the origin:**

$$
f(\mathbf{0}) = \mathbf{0}
$$

This follows by setting $c=0$ in homogeneity. A linear map never translates space.

2. **Parallelism is preserved:**
   Parallel lines remain parallel (or collapse to a single point). Grids never become curves.

---

## 3. Relationship Between Linear Maps and Matrices

Here lies the most interesting fact:

In finite-dimensional vector spaces, **every linear map can be represented as matrix multiplication.**

For a vector $\mathbf{x}$, a linear map can be written as:

$$
f(\mathbf{x}) = A\mathbf{x}
$$

Thus, the abstract concept of a linear map becomes a concrete numerical object—a matrix.

---

## 4. Why Is It Called “Linear”?

The term “linear” comes from the fact that the graph forms a straight line.

For example:

$$
f(x) = ax
$$

is linear because it passes through the origin.

By contrast:

* $f(x) = x^2$ (a curve) is not linear.
* $f(x) = x + 1$ (does not pass through the origin) is not linear.

---

# Summary of the Essence

A linear map is **a transformation that preserves vector addition and scalar multiplication while mapping vectors to another space.**

---

# Basis

A basis is the **minimal and sufficient set of vectors** required to represent every vector in a space.

For example, in a map, “1 km east” and “1 km north” correspond to coordinate axes.

---

## 1. Conditions for a Basis

A set of vectors is a basis if:

1. **Linear independence:** No vector can be expressed as a combination of others.
2. **Spanning:** All points in the space can be reached through linear combinations.

In 2D, two non-parallel vectors form a basis.

The standard basis is:

$$
\mathbf{e}_1 = \begin{pmatrix} 1 \ 0 \end{pmatrix},
\quad
\mathbf{e}_2 = \begin{pmatrix} 0 \ 1 \end{pmatrix}
$$

---

## 2. Expressing a Linear Map via Basis Images

A linear map is completely determined by **where it sends the basis vectors.**

Since any vector can be written as a linear combination of basis vectors, knowing the images of the basis determines the entire map.

For basis ${\mathbf{v}_1, \mathbf{v}_2}$:

$$
A = \big( f(\mathbf{v}_1) \quad f(\mathbf{v}_2) \big)
$$

Each column of $A$ is the transformed basis vector.

---

## 3. Example: Rotation by $90^\circ$

Consider a counterclockwise $90^\circ$ rotation.

$$
\mathbf{e}_1 = \begin{pmatrix} 1 \ 0 \end{pmatrix}
\to
\begin{pmatrix} 0 \ 1 \end{pmatrix}
$$

$$
\mathbf{e}_2 = \begin{pmatrix} 0 \ 1 \end{pmatrix}
\to
\begin{pmatrix} -1 \ 0 \end{pmatrix}
$$

Thus the rotation matrix is:

$$
R =
\begin{pmatrix}
0 & -1 \
1 & 0
\end{pmatrix}
$$

---

## 4. Change of Basis (Representation Matrix)

The transformation itself remains the same, but its matrix representation changes depending on the chosen basis.

* In a poorly chosen basis, the matrix looks complicated.
* In a well-chosen basis (e.g., eigenvectors), the matrix becomes diagonal, revealing the transformation clearly.

---

# Why Linear Maps and Matrices Are Equivalent

Any $n$-dimensional vector can be uniquely written as:

$$
\mathbf{x} =
x_1 \mathbf{e}_1 + \dots + x_n \mathbf{e}_n
$$

Applying linearity:

$$
f(\mathbf{x}) =
x_1 f(\mathbf{e}_1) + \dots + x_n f(\mathbf{e}_n)
$$

Define:

$$
A =
\begin{pmatrix}
f(\mathbf{e}_1) & \dots & f(\mathbf{e}_n)
\end{pmatrix}
$$

Then:

$$
f(\mathbf{x}) = A\mathbf{x}
$$

This guarantees:

* **Uniqueness:** Each linear map corresponds to exactly one matrix.
* **Reproducibility:** Once basis images are known, all inputs can be computed.

---

# Example: Straightness of Linear Maps

A linear map preserves straightness:

* Parallel lines remain parallel.
* The origin stays fixed.
* Lines do not bend.

If the transformation were nonlinear (e.g., involving $x^2$), lines would curve.

---

# Final Summary

In linear algebra, a linear map is equivalent to a matrix.

A linear map does not twist space. It performs straight transformations such as scaling or rotation while preserving structural relationships like parallelism.

Even after transformation, parallel objects remain parallel.

In short:

**Linear map = matrix, and its essential property is linearity.**

---

# Reference

For further reading on linear algebra:

Why Linear Algebra Is Important for Beginners
[Why Linear Algebra Is Important](https://yoshishinnze.hatenablog.com/entry/2026/02/25/000000)
