

## 実対称行列とエルミット行列

### 実対称行列（real symmetric matrix）

**定義**  

実数成分の正方行列 $A \in \mathbb{R}^{n \times n}$ が**実対称行列**であるとは、
$$
A^\mathsf{T} = A
$$
が成り立つことをいいます。  
つまり、転置しても元の行列と同じです。

**主な性質**

- 固有値はすべて**実数**です。
- 異なる固有値に対応する固有ベクトルは**直交**します。
- 適当な**直交行列** $P$（$P^\mathsf{T}P = I$）によって対角化できます：
  $$
  P^\mathsf{T} A P = \mathrm{diag}(\lambda_1, \dots, \lambda_n)
  $$
- 実対称行列は、実数体上の「エルミット行列」に相当します。

**応用**  
- 2次形式（$x^\mathsf{T} A x$）の符号判定（凸最適化など）
- 主成分分析（PCA）における共分散行列
- 力学系のエネルギー関数やヘッセ行列


### エルミット行列（Hermitian matrix）

**定義**  

複素数成分の正方行列 $H \in \mathbb{C}^{n \times n}$ が**エルミット行列**であるとは、
$$
H^H = H
$$
が成り立つことをいいます。  
ここで $H^H$ は**随伴行列（共役転置）** です。

**主な性質**

- 固有値はすべて**実数**です。
- 異なる固有値に対応する固有ベクトルは**直交**します（複素内積に関して）。
- 適当な**ユニタリ行列** $U$（$U^H U = I$）によって対角化できます：
  $$
  U^H H U = \mathrm{diag}(\lambda_1, \dots, \lambda_n)
  $$
- 実数行列の場合、エルミット行列は実対称行列と同じです。

**応用**  
- 量子力学における物理量（オブザーバブル）の表現
- ハミルトニアン（エネルギー演算子）
- 信号処理における自己相関行列（複素信号の場合）

### 歪エルミット行列

歪エルミット行列（skew-Hermitian matrix, anti-Hermitian matrix）は、エルミート行列の「虚数単位倍」に相当する行列で、**随伴行列がマイナス自身になる**行列です。

**定義**  

$n$ 次複素正方行列 $A \in \mathbb{C}^{n \times n}$ が**歪エルミット行列**であるとは、
$$
A^H = -A
$$
が成り立つことをいいます。  
ここで $A^H$ は $A$ の随伴行列（共役転置）です。

成分で書くと、
$$
\overline{A_{ji}} = -A_{ij}
$$
がすべての $i,j$ について成り立ちます。

**主な性質**

1. **固有値は純虚数**  
   歪エルミット行列の固有値はすべて**純虚数**（実部 0 の複素数）になります。

2. **ユニタリ行列との関係**  
   エルミート行列 $H$ に対して
   $$
   A = iH
   $$
   とおくと、$A$ は歪エルミット行列になります。  
   逆に、歪エルミット行列 $A$ に対して
   $$
   H = -iA
   $$
   はエルミート行列です。

3. **指数関数との関係**  
   歪エルミット行列 $A$ の指数関数 $e^{A}$ は**ユニタリ行列**になります：
   $$
   (e^{A})^H = e^{-A} = (e^{A})^{-1}
   $$
   これは量子力学における時間発展演算子 $e^{-iHt/\hbar}$ と同様の構造です（$H$ がエルミートなら $iH$ は歪エルミット）。

4. **内積との関係**  
   任意のベクトル $\mathbf{x}, \mathbf{y} \in \mathbb{C}^n$ に対して、
   $$
   \langle A\mathbf{x}, \mathbf{y} \rangle = - \langle \mathbf{x}, A\mathbf{y} \rangle
   $$
   が成り立ちます（反エルミート性）。

**応用** 

- **量子力学**  
  エルミート演算子 $H$ に対して $iH$ は歪エルミットであり、時間発展演算子 $e^{-iHt/\hbar}$ はユニタリになります。

- **リー代数・リー群**  
  ユニタリ群 $U(n)$ のリー代数は、歪エルミット行列全体のなすベクトル空間です。

- **微分方程式・力学系**  
  線形微分方程式 $\dot{x} = A x$ で $A$ が歪エルミットなら、解のノルムが保存される（エネルギー保存系）などの性質を持ちます。

__簡単な例__

- 実数上の例（歪対称行列）：
  $$
  A = \begin{pmatrix}
  0 & -1 \\
  1 & 0
  \end{pmatrix}
  $$
  は $A^\mathsf{T} = -A$ を満たします。

- 複素数上の例：
  $$
  A = \begin{pmatrix}
  0 & -i \\
  i & 0
  \end{pmatrix}
  $$
  は $A^H = -A$ を満たす歪エルミット行列です。

__定理:__

エルミット行列、特に実対称行列の固有値はすべて実数である。



---


$H \in \mathbb{C}^{n \times n}$ をエルミート行列とする、すなわち
$$
H^\dagger = H
$$
が成り立つとする。このとき、$H$ の固有値はすべて**実数**である。

証明

1. **固有値・固有ベクトルの設定**  
   $\lambda \in \mathbb{C}$ を $H$ の任意の固有値、$\mathbf{v} \in \mathbb{C}^n \setminus \{\mathbf{0}\}$ を対応する固有ベクトルとする：
   $$
   H\mathbf{v} = \lambda \mathbf{v}.
   $$

2. **内積を用いた式を考える**  
   複素標準内積を $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{y}^\dagger \mathbf{x}$ と定義する。  
   まず、左から $\mathbf{v}^\dagger$ をかけて
   $$
   \mathbf{v}^\dagger H \mathbf{v} = \mathbf{v}^\dagger (\lambda \mathbf{v}) = \lambda \mathbf{v}^\dagger \mathbf{v} = \lambda \|\mathbf{v}\|^2.
   $$

3. **随伴を用いて別の表現を得る**  
   一方、$H^\dagger = H$ より、
   $$
   \mathbf{v}^\dagger H \mathbf{v} = \mathbf{v}^\dagger H^\dagger \mathbf{v} = (H\mathbf{v})^\dagger \mathbf{v}.
   $$
   ここで $H\mathbf{v} = \lambda \mathbf{v}$ だから、
   $$
   (H\mathbf{v})^\dagger \mathbf{v} = (\lambda \mathbf{v})^\dagger \mathbf{v} = \overline{\lambda} \mathbf{v}^\dagger \mathbf{v} = \overline{\lambda} \|\mathbf{v}\|^2.
   $$

4. **両辺を比較する**  
   以上より、
   $$
   \lambda \|\mathbf{v}\|^2 = \overline{\lambda} \|\mathbf{v}\|^2.
   $$
   $\mathbf{v} \neq \mathbf{0}$ より $\|\mathbf{v}\|^2 > 0$ なので、両辺を $\|\mathbf{v}\|^2$ で割って
   $$
   \lambda = \overline{\lambda}.
   $$
   これは $\lambda$ が実数であることを意味する。

5. **結論**  
   任意の固有値 $\lambda$ について $\lambda = \overline{\lambda}$ が成り立つので、$H$ の固有値はすべて実数である。

---

__定理:__

歪エルミット行列の固有値はすべて純虚数であることを示せ


---

証明

1. **固有値・固有ベクトルの設定**  
   $\lambda \in \mathbb{C}$ を $A$ の任意の固有値、$\mathbf{v} \in \mathbb{C}^n \setminus \{\mathbf{0}\}$ を対応する固有ベクトルとする：
   $$
   A\mathbf{v} = \lambda \mathbf{v}.
   $$

2. **内積を用いた式を考える**  
   複素標準内積を $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{y}^\dagger \mathbf{x}$ と定義する。  
   まず、左から $\mathbf{v}^\dagger$ をかけて
   $$
   \mathbf{v}^\dagger A \mathbf{v} = \mathbf{v}^\dagger (\lambda \mathbf{v}) = \lambda \mathbf{v}^\dagger \mathbf{v} = \lambda \|\mathbf{v}\|^2.
   $$

3. **随伴を用いて別の表現を得る**  
   一方、$A^\dagger = -A$ より、
   $$
   \mathbf{v}^\dagger A \mathbf{v} = \mathbf{v}^\dagger A^\dagger \mathbf{v} = (A\mathbf{v})^\dagger \mathbf{v}.
   $$
   ここで $A\mathbf{v} = \lambda \mathbf{v}$ だから、
   $$
   (A\mathbf{v})^\dagger \mathbf{v} = (\lambda \mathbf{v})^\dagger \mathbf{v} = \overline{\lambda} \mathbf{v}^\dagger \mathbf{v} = \overline{\lambda} \|\mathbf{v}\|^2.
   $$
   したがって、
   $$
   \mathbf{v}^\dagger A \mathbf{v} = \overline{\lambda} \|\mathbf{v}\|^2.
   $$

4. **歪エルミット性からもう一つの関係を得る**  
   今度は $A^\dagger = -A$ を直接使うと、
   $$
   \mathbf{v}^\dagger A \mathbf{v} = - \mathbf{v}^\dagger A^\dagger \mathbf{v} = - (A\mathbf{v})^\dagger \mathbf{v} = - \overline{\lambda} \|\mathbf{v}\|^2.
   $$

5. **両方の式を比較する**  
   ステップ 3 とステップ 4 より、
   $$
   \overline{\lambda} \|\mathbf{v}\|^2 = - \overline{\lambda} \|\mathbf{v}\|^2.
   $$
   $\mathbf{v} \neq \mathbf{0}$ より $\|\mathbf{v}\|^2 > 0$ なので、両辺を $\|\mathbf{v}\|^2$ で割って
   $$
   \overline{\lambda} = - \overline{\lambda}.
   $$
   よって
   $$
   2\overline{\lambda} = 0 \quad \Rightarrow \quad \overline{\lambda} = 0.
   $$
   したがって $\lambda = 0$ となるか、あるいは $\lambda$ が純虚数であるかのどちらかです。

   しかし、もし $\lambda = 0$ であれば、それは実数であり、実数は純虚数でもあります（実部 0）。  
   一般には $\lambda$ は 0 とは限りませんが、いずれにせよ $\overline{\lambda} = -\overline{\lambda}$ から
   $$
   \overline{\lambda} = 0
   $$
   が導かれるので、$\lambda$ の実部は 0、すなわち $\lambda$ は純虚数です。

6. **結論**  
   任意の固有値 $\lambda$ について $\overline{\lambda} = -\overline{\lambda}$ が成り立つので、$\lambda$ の実部は 0 です。  
   よって $A$ の固有値はすべて純虚数（または 0）です。

---





