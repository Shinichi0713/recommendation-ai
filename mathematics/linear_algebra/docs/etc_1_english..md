Below is a complete English translation of your text, preserving all information and structure.

---

When AI is used, tensors are always involved in the computation.
Tensor computation is often explained with the nuance that it is essentially performing matrix addition and multiplication.

So then—what exactly *is* a “tensor”?

Today, I would like to explain this to readers who have had that very question.

---

## What Is a Tensor?

In one sentence, a tensor is **a general term for multidimensional arrays that encompass scalars, vectors, and matrices.**

It is easiest to understand tensors as a hierarchical structure whose name changes depending on how many “axes (dimensions)” the data has.

The number of axes along which values are arranged is called the **order (or rank)** of the tensor.

* **0th-order tensor: Scalar**

A value with no direction—just a number.

Examples: temperature, mass.

* **1st-order tensor: Vector**

A sequence of numbers arranged along one direction.

Examples: wind velocity (magnitude and direction), coordinates.

* **2nd-order tensor: Matrix**

A table of numbers with two axes: vertical and horizontal.

Examples: linear transformations, pixel intensities of a grayscale image.

* **3rd-order or higher tensor: Tensor**

A block of numbers with three or more axes, such as depth or time.

Examples: color images (height × width × RGB), videos (height × width × color × time).

---

### Why Not Just Stop at “Matrix”? Why Call It a “Tensor”?

Beyond being a mere container of numbers, tensors in mathematics and physics follow important rules.

* **Handling multi-directional changes simultaneously**

For example, when you squeeze a sponge, not only compressive forces but also twisting forces arise internally.
To describe this properly, a single direction (vector) is insufficient—you need a tensor that captures the state in all directions simultaneously.

* **Coordinate-independent properties**

Tensors are defined so that their physical essence (such as spacetime curvature or stress) remains correctly described regardless of the angle or coordinate system from which they are observed.

---

## The Relationship Between Tensors and Matrices

In short, **a tensor is a generalization of a matrix to higher dimensions.**

A matrix has two axes (rows and columns), whereas a tensor can have three, four, or more axes.

---

### 1. Hierarchical Structure by Order

The terminology changes according to the number of axes (dimensions).
In mathematics and data science, this number is called the  **order (or rank)** .

* **0th-order tensor: Scalar**
  A single value (e.g., $5$). No direction.
* **1st-order tensor: Vector**
  A one-dimensional sequence (e.g., $[1, 2, 3]$). One axis.
* **2nd-order tensor: Matrix**
  A two-dimensional array with row and column axes.
* **3rd-order or higher tensor: Multidimensional array**

---

### 2. The Crucial Differences Between Matrices and Tensors

Two key points are important for understanding their relationship:

#### ① Structural Difference as Data

A matrix is planar data, while a 3rd-order or higher tensor resembles a “stack of matrices.”

Example: **Color image data**

An image has vertical pixels, horizontal pixels, and color channels (RGB).
This is treated as a **3rd-order tensor** of shape (height × width × 3).

---

#### ② Difference in Coordinate Transformation Rules (Mathematical/Physical Definition)

A tensor in strict mathematics and physics is not merely a box of numbers.
It has the defining property that **its components transform according to specific rules when the coordinate system changes.**

* A matrix strongly represents a **linear transformation** (a tool that maps vectors to other vectors).
* A tensor more generally describes phenomena such as spatial distortion (stress tensors) or distributed physical quantities in a coordinate-independent way.

---

### 3. Why Do We Often Hear the Word “Tensor”?

In recent years, the term “tensor” has become common in AI and deep learning, largely due to libraries such as  **TensorFlow** , developed by  **Google** .

* **Processing multidimensional data**
  AI processes large amounts of images, videos, and text. These are most efficiently handled as multidimensional arrays (tensors), such as (number of samples × height × width × color).
* **Extension of matrix operations**
  Tensor operations extend matrix algebra into higher dimensions and form the mathematical foundation of modern AI.

---

## Rules of Computation

Are matrix and tensor computations the same?

The conclusion is:

**Matrix computation is a special case of tensor computation, but tensor computation is not equivalent to matrix computation.**

Since a matrix is merely a 2nd-order tensor, higher-order tensors introduce new rules and complexities.

Three main differences:

---

### 1. Explosion of Multiplication Patterns

Matrix multiplication essentially follows one pattern: row × column.

For tensors, you must decide **which axes to multiply together.**

* **Matrix (2nd-order):** Only two axes (row and column), so multiplication direction is fixed.
* **3rd-order tensor:** Three axes (height, width, depth). You may contract A’s depth with B’s height, or A’s width with B’s depth, etc.
  This is mathematically called  **contraction** .

---

### 2. Tensor Product (Extension of Matrix Multiplication)

Beyond matrix multiplication, tensors also allow the **tensor product** (a generalization of the outer product).

* **Dimensions increase:**
  Vector (1st-order) × Vector (1st-order) → Matrix (2nd-order).
  Matrix × Vector via tensor product → 3rd-order tensor.

Unlike matrix multiplication, the order increases as you compute.

---

### 3. Difficulty of Decomposition (Lack of Inverse and Eigenvalues)

Tools common in matrix algebra do not generalize easily.

* **No simple inverse tensor:**
  An “inverse tensor” cannot be defined as straightforwardly as a matrix inverse.
* **Eigenvalue problem:**
  Matrix eigenvalues have elegant properties.
  Tensor eigenvalues have multiple definitions and are computationally very difficult (known to be NP-hard).

---

### 4. Why Are Tensor Calculations Often Treated as Matrix Calculations?

In programming (NumPy, PyTorch), tensors are often **flattened/unfolded** into large matrices.

A 3rd-order block is rearranged into a giant matrix so that existing optimized matrix algorithms can be reused.

---

## Example: Tensor Computation

As an example from AI (deep learning), consider  **convolution** , one of the most fundamental operations.

We simulate applying a filter (weight tensor) to a color image tensor (height × width × RGB) to produce one feature map.

(The Python code remains exactly as written above.)

---

### Result

When executing the code, we observe:

The crucial point of tensor computation is that the three dimensions (height, width, channels) are processed simultaneously as a single block.

Matrix computation typically involves two axes (rows and columns).
In the AI example, three axes interact simultaneously:

* Axis 1: Image height ($H$)
* Axis 2: Image width ($W$)
* Axis 3: Color channel ($C$)

---

### Decisive Difference from Matrix Computation

If treated purely as matrix operations:

* Compute R matrix
* Compute G matrix
* Compute B matrix
* Sum them afterward

In tensor computation, the RGB information at each location is processed together in one multidimensional array.

---

### Tensor-Specific Points in This Computation

* **3rd-order tensor × 3rd-order tensor**

Input data ($10 × 10 × 3$) interacts with a filter ($3 × 3 × 3$).
Color and spatial structure are processed simultaneously.

* **Tensor contraction**

`np.sum(region * kernel)` multiplies corresponding elements and sums them, extracting a scalar from high-order data.

* **Nonlinear transformation**

Applying `maximum(0, output_map)` (ReLU) introduces nonlinearity—something pure linear algebra cannot achieve alone.

---

## Overall Summary

We have examined tensors from conceptual foundations to practical AI usage.

### Why Tensors Instead of Matrices?

* **Preserving multifaceted information:**
  Images (height, width, RGB) and videos (height, width, color, time) must be processed as unified blocks.
* **Interactions across dimensions:**
  Features such as “red edges” or motion over time span multiple axes.
* **Coordinate invariance:**
  In physics, tensors serve as a universal language ensuring laws remain consistent across coordinate systems.

---

### Reality in AI Computation

AI (deep learning) is fundamentally about **transforming and compressing high-order tensors.**

* **Convolution:** Extracting important features via interaction between input and weight tensors.
* **Tensor contraction:** Reducing order along specific axes.
* **Batch processing:** Real systems use 4th-order tensors (e.g., batch × height × width × channel) for efficiency.

---

### Relationship with Matrix Computation

* **Containment:** Matrix computation is a special 2D case of tensor computation.
* **Implementation trick:** Internally, tensors are often flattened into large matrices to leverage optimized linear algebra routines.

---

Finally, I would like to introduce a book for learning the computation and principles of neural networks.
It explains from the basics the logic of processing blocks of data using tensors.
Please consider it as a reference.
