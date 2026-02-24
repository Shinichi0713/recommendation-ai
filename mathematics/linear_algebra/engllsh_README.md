Linear algebra is a required subject at university, yet many students finish it without truly understanding why they studied it in the first place.
At least, that was the case for the author during their student years.

In this chapter, we will explain the importance of linear algebra, outline its representative computations, introduce the fields where it is applied, and finally discuss key points for effective learning.

---

# Why Linear Algebra Is Important

Linear algebra is often called the “king of scientific computing.” In short, this is because **it allows us to reduce complex, nonlinear phenomena in the real world into the “world of lines and grids” that computers are good at handling.**

Why is it so indispensable? The answer can be broken down into three decisive reasons.

---

## 1. Handling Massive Data as a Single Object

Modern data—images, audio, sensor logs—consists of millions of numbers. If we attempted to compute them individually as variables such as `x1, x2, x3...`, our programs would quickly become unmanageable.

**Role of vectors:**
Represent a single “point” or “state” as a structured list.

**Role of matrices:**
Function as a “rule” that transforms that state all at once.

Using matrices, **one million computations can be expressed in a single equation**

$$
Ax = b
$$

Computers—especially GPUs—are specifically designed to process such matrix operations at extremely high speed.

---

## 2. Approximating Difficult Problems as Linear

Real-world phenomena—fluid dynamics, economic fluctuations, rocket trajectories—are inherently nonlinear and complex. However, both humans and computers struggle to solve nonlinear problems directly.

We exploit the principle that **if we examine a sufficiently small region, even a curve looks like a straight line** (differentiation).

**Linear approximation:**
Break complex curves into small segments and solve each locally as a linear (matrix-based) problem.

This process—“divide finely and solve with matrices”—forms the foundation of simulation techniques such as the Finite Element Method (FEM).

---

## 3. Manipulating the Properties of Space

Linear algebra is not merely about tables of numbers; it is about **transformations of space**.

**Rotation, scaling, projection:**
Moving a character in 3D graphics or changing a camera viewpoint is achieved through matrix-based spatial transformations.

**Dimensionality reduction:**
Extracting the essential 2–3 dimensions from 100-dimensional data (e.g., Principal Component Analysis). This involves finding the “important axes” (eigenvectors) of the data.

---

## Summary: Why Study Linear Algebra?

The greatest advantage of learning linear algebra is that **it allows us to organize complex, high-dimensional problems into simple structures composed only of addition and multiplication.**

In programming, it is a weapon for maximizing memory efficiency and computational speed.
In theory, it is a map for navigating high-dimensional labyrinths.

---

# Important Computations in Linear Algebra

Linear algebra is the very “language” of modern scientific computing—simulation, data analysis, AI, and more. Rather than focusing solely on textbook-style manual calculations (e.g., hand-computing determinants), the practical shortcut is to prioritize **conceptual understanding of what is possible and the computational properties relevant to machines.**

Key elements are summarized below.

---

## 1. Matrix Decomposition

One of the most frequently used concepts in scientific computing. A large matrix is decomposed into a product of matrices with convenient properties.

**LU Decomposition:**
A standard method for efficiently solving systems of linear equations.

**QR Decomposition:**
The foundation of least squares methods and eigenvalue computations.

**Singular Value Decomposition (SVD):**
One of the most powerful tools. Used for dimensionality reduction (PCA), noise removal, and determining matrix rank.

---

## 2. Eigenvalues and Eigenvectors

Essential for understanding stability and oscillation in physical systems, as well as features in data.

**Principal Component Analysis (PCA):**
Identifies the principal directions of multidimensional data.

**Spectral decomposition:**
Expresses a matrix in terms of its eigenvalues and eigenvectors.

**Diagonalization:**
Simplifies complex matrix operations, especially matrix powers such as

$$
A^n
$$

---

## 3. Vector Spaces and Bases

This concerns the perspective from which we view data.

**Change of basis:**
Switching coordinate systems can dramatically simplify computation. (The Fourier transform can be viewed as a type of basis transformation.)

**Orthogonality:**
Using orthogonal bases (inner product equals zero) improves numerical stability and removes redundancy in information.

---

## 4. Numerical Linear Algebra Practices

Beyond theory, it is crucial to understand computational realities.

**Condition number:**
A measure of how “ill-behaved” a matrix is. A large condition number means small input errors can cause large deviations in the solution.

**Sparse matrices:**
Matrices where most elements are zero. In very large-scale problems (tens of thousands of dimensions or more), special data structures and algorithms that avoid storing zeros are essential.

**Norm:**
A measure of the “size” of a vector or matrix, used to evaluate error.

---

## 5. Least Squares Method

A technique for finding the model that best fits observed data.

Theoretically, this involves solving the normal equation

$$
A^T A x = A^T b
$$

However, in practice, **QR decomposition** or **SVD** is typically used for improved numerical stability.

---

# Application Fields of Linear Algebra

Linear algebra is an invisible infrastructure of modern society. Beyond AI, matrix and vector computations support a wide range of domains.

---

## 1. Computer Graphics (CG) and Games

Character movement and viewpoint changes on screen are all implemented via coordinate transformations using matrices.

**Rotation, scaling, translation:**
Each vertex (vector) of a 3D model is multiplied by transformation matrices to compute positions instantly.

**Projection:**
Mapping 3D data onto a 2D screen (perspective projection) is a matrix operation.

**Shading:**
Light calculations use the inner product between surface normals and light direction vectors.

---

## 2. Structural Analysis and Simulation (CAE)

Bridge design, car crash testing, and aircraft aerodynamics are impossible without linear algebra.

**Finite Element Method (FEM):**
Complex objects are divided into small elements (meshes), and their relationships are represented as huge matrices.

**Systems of linear equations:**
Ultimately, forces and displacements are solved as large-scale systems such as

$$
Ax = b
$$

often involving millions of variables.

---

## 3. Network Analysis (e.g., Google Search)

Matrices are used to determine the “importance” of web pages.

**PageRank algorithm:**
Websites are nodes, links are edges, and a massive adjacency matrix is constructed.

**Eigenvalue problem:**
Solving “pages with many incoming links are important” reduces to finding the eigenvector corresponding to the largest eigenvalue of this matrix.

---

## 4. Quantum Mechanics and Quantum Computing

Quantum mechanics is formulated in the language of linear algebra.

**Quantum states:**
Represented as complex-valued vectors.

**Observation and evolution:**
Physical operations correspond to multiplying by special matrices such as Hermitian or unitary matrices.

**Quantum computing:**
Gate operations are matrix operations applied to qubits (vectors).

---

## 5. Image and Audio Compression (Signal Processing)

Smooth streaming on platforms and JPEG compression rely on linear algebra.

**Fourier transform:**
Decomposes signals into frequency components (basis vectors).

**Singular Value Decomposition (SVD):**
By truncating small singular values of an image matrix, we achieve dramatic data compression with minimal perceptual change.

---

## 6. Statistics and Financial Engineering

Used to discover correlations and predict risk in large datasets.

**Covariance matrix:**
Represents relationships among multiple variables.

**Portfolio optimization:**
Risk minimization and return maximization are formulated using quadratic forms of matrices.

---

# Keys to Learning

Below are key points for understanding linear algebra effectively.

---

## What Is a Mapping?

In linear algebra, a mapping can be defined succinctly as **a rule that transforms one vector into another vector.**

For programmers, it can be understood as **a function that takes a vector as input and returns a new vector.**

If mapping $f$ transforms vector $\vec{x}$ into $\vec{y}$, we write:

$$
f(\vec{x}) = \vec{y}
$$

A linear mapping performs this operation via matrix multiplication.

Here, matrix $A$ is not merely a table of numbers but **an instruction manual describing how the entire space is deformed.**

---

## The Two Absolute Rules of Linear Mappings

A mapping is called linear only if it preserves straightness in space:

**Additivity:**

$$
f(\vec{u} + \vec{v}) = f(\vec{u}) + f(\vec{v})
$$

**Homogeneity (scalar multiplication):**

$$
f(c \vec{u}) = c f(\vec{u})
$$

Intuitively, such mappings preserve the grid structure: equally spaced lines remain straight and evenly spaced, and the origin remains fixed. No bending or shifting of the origin occurs.

---

## Concrete Examples of Mappings (2×2 Case)

* **Scaling:** Stretching or shrinking space.
* **Rotation:** Rotating around the origin.
* **Reflection:** Mirroring across a line.
* **Projection:** Flattening 3D space onto a 2D plane (losing information).

---

## Why the Mapping Perspective Matters

Thinking “compute matrix $A$” is tedious. Thinking “transform space via mapping $f$” clarifies major concepts:

**Inverse matrix:**
Can we reverse the transformation? (If space is crushed, inversion is impossible.)

**Kernel:**
Which vectors are mapped to the origin?

**Eigenvectors:**
Which directions remain unchanged in direction (only scaled in magnitude)?

---

## A. Visualize the Process

Following formulas alone eventually hits a limit.
Use Python to visualize the state of data before and after matrix transformations.
Visualization deepens conceptual understanding.

---

## B. Linear Algebra Maps Blocks of Numbers

Linear algebra can be seen as **mapping blocks of numbers into new blocks.**

A helpful analogy:
A “recipe” (mapping) and “ingredients” (number blocks), or a **“processing machine” (mapping) and “raw material” (number blocks).**

Matrix $A$ is the “machine” that takes in a block and outputs a transformed block:

$$
A (\text{original block}) = \text{new block}
$$

---

### Examples of Processing

**1. Scaling machine**
Doubles or halves the contents of the block.
Practical example: resizing images or adjusting speaker volume.

**2. Rotation machine**
Changes the orientation of space.
Practical example: rotating a smartphone display when tilted.

**3. Information extraction machine (projection)**
Flattens a 3D block into a “shadow.”
Practical example: extracting “purchase intent” from massive customer datasets.

---

## C. Learn Together with Programming

Manual calculation up to $3 \times 3$ matrices is sufficient.
For larger problems, use Python libraries such as NumPy to observe how large datasets are transformed in practice.

This bridges theory with real-world application.
