

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

歪エルミット行列の固有値はすべて純虚数であることを示せ。


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

__定理:__

エルミット行列、特に実対称行列の相異なる固有値に属する固有ベクトルは互いに直交する。


---

__証明すべき命題__

$H \in \mathbb{C}^{n \times n}$ をエルミート行列とする、すなわち
$$
H^\dagger = H
$$
が成り立つとする。  
$\lambda, \mu$ を $H$ の相異なる固有値（$\lambda \neq \mu$）とし、$\mathbf{v}, \mathbf{w}$ をそれぞれ対応する固有ベクトルとする：
$$
H\mathbf{v} = \lambda \mathbf{v}, \quad H\mathbf{w} = \mu \mathbf{w}, \quad \mathbf{v}, \mathbf{w} \neq \mathbf{0}.
$$
このとき、$\mathbf{v}$ と $\mathbf{w}$ は**直交**する：
$$
\langle \mathbf{v}, \mathbf{w} \rangle = 0.
$$

__証明__

1. **内積の定義と仮定**  
   複素標準内積を $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{y}^\dagger \mathbf{x}$ と定義する。  
   固有ベクトルの条件は
   $$
   H\mathbf{v} = \lambda \mathbf{v}, \quad H\mathbf{w} = \mu \mathbf{w}.
   $$

2. **$\langle H\mathbf{v}, \mathbf{w} \rangle$ を2通りに計算する**

   - 一つ目の計算：
     $$
     \langle H\mathbf{v}, \mathbf{w} \rangle = \langle \lambda \mathbf{v}, \mathbf{w} \rangle = \lambda \langle \mathbf{v}, \mathbf{w} \rangle.
     $$

   - 二つ目の計算：  
     エルミート性 $H^\dagger = H$ より、
     $$
     \langle H\mathbf{v}, \mathbf{w} \rangle = \mathbf{w}^\dagger H\mathbf{v} = \mathbf{w}^\dagger H^\dagger \mathbf{v} = (H\mathbf{w})^\dagger \mathbf{v} = \langle \mathbf{v}, H\mathbf{w} \rangle.
     $$
     ここで $H\mathbf{w} = \mu \mathbf{w}$ だから、
     $$
     \langle \mathbf{v}, H\mathbf{w} \rangle = \langle \mathbf{v}, \mu \mathbf{w} \rangle = \mu \langle \mathbf{v}, \mathbf{w} \rangle.
     $$

3. **2つの式を比較する**  
   以上より、
   $$
   \lambda \langle \mathbf{v}, \mathbf{w} \rangle = \mu \langle \mathbf{v}, \mathbf{w} \rangle.
   $$
   移項して
   $$
   (\lambda - \mu) \langle \mathbf{v}, \mathbf{w} \rangle = 0.
   $$

4. **結論**  
   $\lambda \neq \mu$ より $\lambda - \mu \neq 0$ なので、両辺を $\lambda - \mu$ で割って
   $$
   \langle \mathbf{v}, \mathbf{w} \rangle = 0.
   $$
   すなわち、$\mathbf{v}$ と $\mathbf{w}$ は直交します。

__実対称行列の場合__

実対称行列 $A \in \mathbb{R}^{n \times n}$ は
$$
A^\mathsf{T} = A
$$
を満たします。実数行列では随伴行列は単なる転置なので、これは
$$
A^\dagger = A
$$
と同値です。したがって実対称行列はエルミート行列の特別な場合であり、上記の証明から、相異なる固有値に対応する固有ベクトルは直交します。

---

__定理:__

$n$次実正方行列 $A$ について、次の２つの条件は同値である。

(1) $A$は対称行列である。

(2) $A$は適当な直交用列 $P$ によって対角化できる。すなわち

$$
P^{-1}AP = \begin{pmatrix} \lambda_1 & 0 & 0 \\ 0 & \lambda_2 & 0 \\ 0 & 0 & \ddots \end{pmatrix}
$$




---

__(1) ⇒ (2) の証明（実対称行列は直交行列で対角化可能）__

**主張**：$A^\mathsf{T} = A$ ならば、ある直交行列 $P$（$P^\mathsf{T}P = I$）が存在して
$$
P^{-1}AP = P^\mathsf{T}AP = \mathrm{diag}(\lambda_1, \dots, \lambda_n)
$$
と対角化できる。

**証明の流れ**（概略）：

1. **固有値はすべて実数**  
   実対称行列の固有値はすべて実数であることは、エルミート行列の固有値が実数であることの証明と同様に、内積を用いて示せます。  
   すなわち、固有値 $\lambda$ と固有ベクトル $\mathbf{v}$ に対して
   $$
   \lambda \|\mathbf{v}\|^2 = \mathbf{v}^\mathsf{T} A \mathbf{v} = (A\mathbf{v})^\mathsf{T} \mathbf{v} = \lambda \|\mathbf{v}\|^2
   $$
   から $\lambda = \overline{\lambda}$ が従い、$\lambda$ は実数です。

2. **相異なる固有値に対応する固有ベクトルは直交**  
   相異なる固有値 $\lambda \neq \mu$ に対応する固有ベクトル $\mathbf{v}, \mathbf{w}$ について、
   $$
   \lambda \langle \mathbf{v}, \mathbf{w} \rangle = \langle A\mathbf{v}, \mathbf{w} \rangle = \langle \mathbf{v}, A\mathbf{w} \rangle = \mu \langle \mathbf{v}, \mathbf{w} \rangle
   $$
   より $(\lambda - \mu)\langle \mathbf{v}, \mathbf{w} \rangle = 0$ となり、$\lambda \neq \mu$ だから $\langle \mathbf{v}, \mathbf{w} \rangle = 0$ です。

3. **各固有空間で正規直交基底を取る**  
   同じ固有値に属する固有ベクトルたちは、一般には直交しませんが、グラム・シュミットの直交化によって正規直交基底を構成できます。

4. **全体として正規直交固有ベクトル基底が取れる**  
   実対称行列は正規行列でもあるため、すべての固有ベクトルを集めて正規直交基底 $\{\mathbf{p}_1, \dots, \mathbf{p}_n\}$ を構成できます。

5. **直交行列 $P$ による対角化**  
   $P = [\mathbf{p}_1 \cdots \mathbf{p}_n]$ とおくと、$P$ は直交行列（$P^\mathsf{T}P = I$）であり、
   $$
   AP = P \mathrm{diag}(\lambda_1, \dots, \lambda_n)
   $$
   が成り立ちます。したがって
   $$
   P^{-1}AP = P^\mathsf{T}AP = \mathrm{diag}(\lambda_1, \dots, \lambda_n)
   $$
   となり、$A$ は直交行列で対角化可能です。

__(2) ⇒ (1) の証明（直交行列で対角化できるなら対称）__

**主張**：ある直交行列 $P$（$P^\mathsf{T}P = I$）と実対角行列 $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$ が存在して
$$
P^{-1}AP = P^\mathsf{T}AP = \Lambda
$$
ならば、$A$ は対称行列（$A^\mathsf{T} = A$）である。

**証明**：

仮定より
$$
P^\mathsf{T}AP = \Lambda.
$$
$\Lambda$ は対角行列なので $\Lambda^\mathsf{T} = \Lambda$ です。  
また $P$ は直交行列なので $P^{-1} = P^\mathsf{T}$ です。

1. 上式の両辺の転置を取ると、
   $$
   (P^\mathsf{T}AP)^\mathsf{T} = P^\mathsf{T} A^\mathsf{T} (P^\mathsf{T})^\mathsf{T} = P^\mathsf{T} A^\mathsf{T} P = \Lambda^\mathsf{T} = \Lambda.
   $$

2. したがって
   $$
   P^\mathsf{T} A^\mathsf{T} P = P^\mathsf{T} A P.
   $$

3. 両辺の左から $P$、右から $P^\mathsf{T}$ をかけると、
   $$
   P(P^\mathsf{T} A^\mathsf{T} P)P^\mathsf{T} = P(P^\mathsf{T} A P)P^\mathsf{T}.
   $$
   すなわち
   $$
   A^\mathsf{T} = A.
   $$
   よって $A$ は対称行列です。

以上より、

- (1) $A$ が対称行列ならば、直交行列で対角化可能。
- (2) 直交行列で対角化可能ならば、$A$ は対称行列。

が示されたので、2つの条件は同値です。

---

__例題:__

次の実対称行列を直交行列によって対角化せよ。

$$
A = \begin{pmatrix} 0 & 0 & 1 \\ 0 & -1 & 0 \\ 1 & 0 & 0 \end{pmatrix}
$$



---

__1. 固有値の計算__

固有方程式は
$$
\det(A - \lambda I) = 0
$$
です。

$$
A - \lambda I = \begin{pmatrix}
-\lambda & 0 & 1 \\
0 & -1-\lambda & 0 \\
1 & 0 & -\lambda
\end{pmatrix}
$$

行列式を計算します（第2行で展開すると楽です）：

- 第2行は $(0, -1-\lambda, 0)$ なので、$(2,2)$ 成分の余因子を考えると
  $$
  \det(A - \lambda I) = (-1-\lambda) \cdot \det\begin{pmatrix} -\lambda & 1 \\ 1 & -\lambda \end{pmatrix}.
  $$

- 2×2 行列の行列式は
  $$
  \det\begin{pmatrix} -\lambda & 1 \\ 1 & -\lambda \end{pmatrix} = (-\lambda)(-\lambda) - 1\cdot 1 = \lambda^2 - 1.
  $$

したがって
$$
\det(A - \lambda I) = (-1-\lambda)(\lambda^2 - 1) = -(1+\lambda)(\lambda - 1)(\lambda + 1) = -(\lambda+1)^2(\lambda-1).
$$

よって固有値は
$$
\lambda = -1 \quad (\text{重複度 2}),\quad \lambda = 1 \quad (\text{重複度 1}).
$$

__2. 固有ベクトルの計算__

__(a) 固有値 $\lambda = 1$ の場合__

$(A - I)\mathbf{v} = \mathbf{0}$ を解きます。

$$
A - I = \begin{pmatrix}
-1 & 0 & 1 \\
0 & -2 & 0 \\
1 & 0 & -1
\end{pmatrix}
$$

行基本変形すると（第1行＋第3行、第2行を -2 で割るなど）、
$$
\begin{pmatrix}
-1 & 0 & 1 \\
0 & 1 & 0 \\
0 & 0 & 0
\end{pmatrix}
\quad\Rightarrow\quad
\begin{cases}
-x_1 + x_3 = 0 \\
x_2 = 0
\end{cases}
$$

したがって $x_1 = x_3$, $x_2 = 0$ です。  
固有ベクトルとして
$$
\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}
$$
が取れます（長さは $\sqrt{2}$）。

__(b) 固有値 $\lambda = -1$ の場合__

$(A + I)\mathbf{v} = \mathbf{0}$ を解きます。

$$
A + I = \begin{pmatrix}
1 & 0 & 1 \\
0 & 0 & 0 \\
1 & 0 & 1
\end{pmatrix}
$$

行基本変形すると、
$$
\begin{pmatrix}
1 & 0 & 1 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{pmatrix}
\quad\Rightarrow\quad x_1 + x_3 = 0.
$$

したがって $x_3 = -x_1$ で、$x_2$ は自由です。  
基底として例えば
$$
\mathbf{v}_2 = \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix},\quad
\mathbf{v}_3 = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}
$$
が取れます。

これらは互いに直交します：
- $\mathbf{v}_2 \cdot \mathbf{v}_3 = 0$
- $\mathbf{v}_1 \cdot \mathbf{v}_2 = 1\cdot 1 + 0\cdot 0 + 1\cdot(-1) = 0$
- $\mathbf{v}_1 \cdot \mathbf{v}_3 = 0$

__3. 正規直交基底の構成__

各ベクトルを正規化します。

- $\mathbf{v}_1$：長さ $\sqrt{1^2 + 0^2 + 1^2} = \sqrt{2}$  
  $$
  \mathbf{p}_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}
  $$

- $\mathbf{v}_2$：長さ $\sqrt{1^2 + 0^2 + (-1)^2} = \sqrt{2}$  
  $$
  \mathbf{p}_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix}
  $$

- $\mathbf{v}_3$：長さ $\sqrt{0^2 + 1^2 + 0^2} = 1$  
  $$
  \mathbf{p}_3 = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}
  $$

これらは正規直交系です。

__4. 直交行列 $P$ と対角化__

$$
P = [\mathbf{p}_1 \ \mathbf{p}_2 \ \mathbf{p}_3] = \begin{pmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\
0 & 0 & 1 \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0
\end{pmatrix}
$$

$P$ は直交行列なので $P^{-1} = P^\mathsf{T}$ です。  
固有値の順序に合わせて
$$
\Lambda = \begin{pmatrix}
1 & 0 & 0 \\
0 & -1 & 0 \\
0 & 0 & -1
\end{pmatrix}
$$
とおくと、
$$
P^\mathsf{T} A P = \Lambda
$$
が成り立ちます（検算可能です）。

__5. 結論__

実対称行列
$$
A = \begin{pmatrix} 0 & 0 & 1 \\ 0 & -1 & 0 \\ 1 & 0 & 0 \end{pmatrix}
$$
は、直交行列
$$
P = \begin{pmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\
0 & 0 & 1 \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0
\end{pmatrix}
$$
によって
$$
P^{-1}AP = P^\mathsf{T}AP = \begin{pmatrix}
1 & 0 & 0 \\
0 & -1 & 0 \\
0 & 0 & -1
\end{pmatrix}
$$
と対角化されます。

---


__定理:__

$n$次複素正方行列 $A$ について、次の2つの条件は同値である。

(1) $A$はエルミット行列である。

(2) $A$は適当なユニタリ行列 $U$ によって、対角成分が実数からなる行列に対角化できる。すなわち

$$
U^(-1)AU = \begin{pmatrix} \lambda_1 & 0 & 0 \\ 0 & \lambda_2 & 0 \\ 0 & 0 & \ddots \end{pmatrix}
$$

($\lambda_1$, $\lambda_2, ...$, $\lambda_n$は実数)



---

__(1) ⇒ (2) の証明（エルミート行列はユニタリ行列で実対角化可能）__

**主張**：$A^\dagger = A$ ならば、あるユニタリ行列 $U$（$U^\dagger U = I$）と実対角行列 $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$ が存在して
$$
U^{-1}AU = U^\dagger A U = \Lambda
$$
と書ける。

**証明の流れ**（概略）：

1. **固有値はすべて実数**  
   エルミート行列の固有値はすべて実数であることは、内積を用いて示せます。  
   固有値 $\lambda$ と固有ベクトル $\mathbf{v} \neq \mathbf{0}$ に対して
   $$
   \lambda \|\mathbf{v}\|^2 = \langle A\mathbf{v}, \mathbf{v} \rangle = \langle \mathbf{v}, A\mathbf{v} \rangle = \overline{\lambda} \|\mathbf{v}\|^2
   $$
   より $\lambda = \overline{\lambda}$ となり、$\lambda$ は実数です。

2. **相異なる固有値に対応する固有ベクトルは直交**  
   相異なる固有値 $\lambda \neq \mu$ に対応する固有ベクトル $\mathbf{v}, \mathbf{w}$ について、
   $$
   \lambda \langle \mathbf{v}, \mathbf{w} \rangle = \langle A\mathbf{v}, \mathbf{w} \rangle = \langle \mathbf{v}, A\mathbf{w} \rangle = \mu \langle \mathbf{v}, \mathbf{w} \rangle
   $$
   より $(\lambda - \mu)\langle \mathbf{v}, \mathbf{w} \rangle = 0$ となり、$\lambda \neq \mu$ だから $\langle \mathbf{v}, \mathbf{w} \rangle = 0$ です。

3. **各固有空間で正規直交基底を取る**  
   同じ固有値に属する固有ベクトルたちは、一般には直交しませんが、グラム・シュミットの直交化によって正規直交基底を構成できます。

4. **全体として正規直交固有ベクトル基底が取れる**  
   エルミート行列は正規行列でもあるため、すべての固有ベクトルを集めて正規直交基底 $\{\mathbf{u}_1, \dots, \mathbf{u}_n\}$ を構成できます。

5. **ユニタリ行列 $U$ による対角化**  
   $U = [\mathbf{u}_1 \cdots \mathbf{u}_n]$ とおくと、$U$ はユニタリ行列（$U^\dagger U = I$）であり、
   $$
   AU = U \mathrm{diag}(\lambda_1, \dots, \lambda_n)
   $$
   が成り立ちます。したがって
   $$
   U^{-1}AU = U^\dagger A U = \mathrm{diag}(\lambda_1, \dots, \lambda_n)
   $$
   となり、$A$ はユニタリ行列で実対角行列に対角化可能です。

__(2) ⇒ (1) の証明（ユニタリ行列で実対角化できるならエルミート）__

**主張**：あるユニタリ行列 $U$（$U^\dagger U = I$）と実対角行列 $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$ が存在して
$$
U^{-1}AU = U^\dagger A U = \Lambda
$$
ならば、$A$ はエルミート行列（$A^\dagger = A$）である。

**証明**：

仮定より
$$
U^\dagger A U = \Lambda.
$$
$\Lambda$ は実対角行列なので $\Lambda^\dagger = \Lambda$ です。  
また $U$ はユニタリ行列なので $U^{-1} = U^\dagger$ です。

1. 上式の両辺の随伴（共役転置）を取ると、
   $$
   (U^\dagger A U)^\dagger = U^\dagger A^\dagger (U^\dagger)^\dagger = U^\dagger A^\dagger U = \Lambda^\dagger = \Lambda.
   $$

2. したがって
   $$
   U^\dagger A^\dagger U = U^\dagger A U.
   $$

3. 両辺の左から $U$、右から $U^\dagger$ をかけると、
   $$
   U(U^\dagger A^\dagger U)U^\dagger = U(U^\dagger A U)U^\dagger.
   $$
   すなわち
   $$
   A^\dagger = A.
   $$
   よって $A$ はエルミート行列です。


以上より、

- (1) $A$ がエルミート行列ならば、ユニタリ行列で実対角行列に対角化可能。
- (2) ユニタリ行列で実対角行列に対角化可能ならば、$A$ はエルミート行列。

が示されたので、2つの条件は同値です。

---

## 正規行列

**正規行列（normal matrix）** とは、**自分自身の随伴行列（共役転置）と可換である**行列のことです。

### 数学的定義

$n$ 次複素正方行列 $A \in \mathbb{C}^{n \times n}$ が**正規行列**であるとは、
$$
A A^\dagger = A^\dagger A
$$
が成り立つことをいいます。  
ここで $A^\dagger$ は $A$ の随伴行列（共役転置）です。

実数行列の場合、随伴行列は単なる転置行列 $A^\mathsf{T}$ なので、正規行列の条件は
$$
A A^\mathsf{T} = A^\mathsf{T} A
$$
となります。


### 重要な性質

- **スペクトル定理**  
  正規行列はユニタリ行列で対角化可能です。  
  すなわち、あるユニタリ行列 $U$（$U^\dagger U = I$）と対角行列 $\Lambda$ が存在して
  $$
  U^{-1} A U = U^\dagger A U = \Lambda
  $$
  と書けます。

- **固有ベクトルの正規直交性**  
  正規行列の固有ベクトルからなる正規直交基底が存在します。

- **ノルム保存との関係**  
  正規行列 $A$ について、任意のベクトル $\mathbf{x}$ に対して
  $$
  \|A\mathbf{x}\| = \|A^\dagger \mathbf{x}\|
  $$
  が成り立ちます。


### 代表的な正規行列の例

1. **エルミート行列**（$A^\dagger = A$）  
   例：実対称行列（実数版のエルミート行列）

2. **ユニタリ行列**（$U^\dagger U = I$）  
   例：直交行列（実数版のユニタリ行列）

3. **歪エルミート行列**（$A^\dagger = -A$）  
   例：歪対称行列（実数版の歪エルミート行列）

4. **その他**  
   例えば、対角行列や、より一般に可換なエルミート行列の線形結合なども正規行列です。

### 利用例

正規行列は「ユニタリ行列で対角化できる」という強い性質を持つため、**対角化やスペクトル分解が重要な場面**でよく利用されます。主な利用場面は以下の通りです。

__1. 量子力学__

- **物理量（オブザーバブル）の表現**  
  位置、運動量、スピン、エネルギーなどの物理量は、エルミート演算子（無限次元版のエルミート行列）で表されます。  
  エルミート行列は正規行列の一種であり、固有値が実数で、固有ベクトルが正規直交基底をなすため、測定結果（固有値）と状態（固有ベクトル）の対応が明確になります。

- **時間発展演算子**  
  時間発展演算子 $U(t) = e^{-iHt/\hbar}$ はユニタリ行列であり、これも正規行列です。  
  正規性により、ハミルトニアン $H$ のスペクトル分解を通じて時間発展を解析できます。

__2. 信号処理・フーリエ解析__

- **フーリエ変換行列**  
  離散フーリエ変換（DFT）の行列はユニタリ行列（正規行列）であり、信号を周波数成分に分解する際に利用されます。

- **自己相関行列・共分散行列**  
  信号の自己相関行列や統計データの共分散行列はエルミート（実対称）であり、正規行列です。  
  その固有値分解（スペクトル分解）が、主成分分析（PCA）やノイズ解析などに使われます。

__3. 数値線形代数・行列関数__

- **行列の対角化と関数計算**  
  正規行列 $A$ はユニタリ行列で対角化できるため、行列のべき乗 $A^k$ や指数関数 $e^A$、その他の関数 $f(A)$ の計算が容易になります。

- **特異値分解（SVD）との関係**  
  任意の行列 $A$ に対して $A^\dagger A$ や $A A^\dagger$ はエルミート（正規）であり、その固有値分解から特異値分解が導かれます。

__4. 制御理論・安定性解析__

- **線形システムの状態遷移行列**  
  系の状態遷移行列が正規（特にユニタリやエルミート）であれば、エネルギーやノルムの保存性など、系の安定性や構造を解析しやすくなります。

- **リャプノフ方程式**  
  制御理論では、系の安定性を調べる際に現れる行列方程式に正規行列が登場し、そのスペクトル分解が安定性条件と結びつきます。

__5. 群論・幾何学__

- **ユニタリ群・直交群**  
  ユニタリ行列や直交行列は正規行列であり、これらはコンパクト李群として研究されます。  
  幾何学的には、複素（または実）ベクトル空間における「回転＋鏡映」の群に対応します。

- **リー代数との対応**  
  ユニタリ群のリー代数は歪エルミート行列（正規行列の一種）のなす空間であり、指数写像を通じて群と代数が結びつきます。

__6. 情報理論・符号理論__

- **ユニタリ誤差モデル**  
  量子情報では、ノイズや誤差をユニタリ変換としてモデル化することがあり、これらは正規行列です。

- **量子誤り訂正符号**  
  正規行列の性質を利用して、量子状態を保護する符号を設計する研究があります。



### 注意点

- すべての行列が正規行列とは限りません。  
  例えば、ジョルダン標準形が対角化できない行列（冪零行列の一部など）は正規行列ではありません。

- 正規行列は「ユニタリ対角化可能」という強い性質を持ち、量子力学や信号処理などでよく現れます。

以上が正規行列の定義と概要です。


__定理:__

$n$次複素正方行列$A$について、
- (1) $A$ は正規行列（$A A^\dagger = A^\dagger A$）
- (2) $A$ はユニタリ行列によって対角化される



---

__(1) ⇒ (2) の証明（正規行列はユニタリ対角化可能）__

**主張**：$A A^\dagger = A^\dagger A$ ならば、あるユニタリ行列 $U$（$U^\dagger U = I$）と対角行列 $\Lambda$ が存在して
$$
U^{-1} A U = U^\dagger A U = \Lambda
$$
と書ける。

**証明の流れ**（概略）：

1. **シュール分解の適用**  
   任意の複素正方行列は、ユニタリ行列 $U$ によって上三角行列 $T$ に三角化できます（シュール分解）：
   $$
   U^\dagger A U = T.
   $$
   ここで $T$ は上三角行列です。

2. **正規性から $T$ が対角行列になることを示す**  
   $A$ が正規行列であると仮定します：
   $$
   A A^\dagger = A^\dagger A.
   $$
   シュール分解 $U^\dagger A U = T$ より、
   $$
   A = U T U^\dagger.
   $$
   このとき
   $$
   A^\dagger = U T^\dagger U^\dagger.
   $$
   したがって
   $$
   A A^\dagger = U T T^\dagger U^\dagger,\quad A^\dagger A = U T^\dagger T U^\dagger.
   $$
   正規性 $A A^\dagger = A^\dagger A$ より
   $$
   U T T^\dagger U^\dagger = U T^\dagger T U^\dagger.
   $$
   両辺の左から $U^\dagger$、右から $U$ をかけると
   $$
   T T^\dagger = T^\dagger T.
   $$
   すなわち $T$ も正規行列です。

3. **上三角かつ正規な行列は対角行列**  
   $T$ は上三角行列なので、$(i,j)$ 成分について $i > j$ なら $T_{ij} = 0$ です。  
   $T$ が正規であることから、$T T^\dagger = T^\dagger T$ の $(i,i)$ 成分を比較すると、
   $$
   |T_{ii}|^2 + \sum_{k>i} |T_{ik}|^2 = |T_{ii}|^2 + \sum_{k<i} |T_{ki}|^2.
   $$
   右辺の $\sum_{k<i} |T_{ki}|^2$ は $T$ が上三角であるため 0 です。  
   したがって
   $$
   \sum_{k>i} |T_{ik}|^2 = 0
   $$
   となり、$i < k$ に対して $T_{ik} = 0$ です。  
   よって $T$ は対角行列です。

4. **結論**  
   以上より $T = \Lambda$（対角行列）であり、
   $$
   U^\dagger A U = \Lambda
   $$
   が成り立ちます。すなわち $A$ はユニタリ行列 $U$ によって対角化されます。

__(2) ⇒ (1) の証明（ユニタリ対角化可能なら正規）__

**主張**：あるユニタリ行列 $U$（$U^\dagger U = I$）と対角行列 $\Lambda$ が存在して
$$
U^{-1} A U = U^\dagger A U = \Lambda
$$
ならば、$A$ は正規行列（$A A^\dagger = A^\dagger A$）である。

**証明**：

仮定より
$$
U^\dagger A U = \Lambda.
$$
$\Lambda$ は対角行列なので $\Lambda^\dagger = \overline{\Lambda}$（成分ごとの共役）であり、対角行列同士は可換なので
$$
\Lambda \Lambda^\dagger = \Lambda^\dagger \Lambda
$$
が成り立ちます。

1. 上式の左辺と右辺をそれぞれ計算すると、
   - 左辺：
     $$
     A A^\dagger = (U \Lambda U^\dagger)(U \Lambda^\dagger U^\dagger) = U \Lambda \Lambda^\dagger U^\dagger.
     $$
   - 右辺：
     $$
     A^\dagger A = (U \Lambda^\dagger U^\dagger)(U \Lambda U^\dagger) = U \Lambda^\dagger \Lambda U^\dagger.
     $$

2. $\Lambda \Lambda^\dagger = \Lambda^\dagger \Lambda$ より、
   $$
   U \Lambda \Lambda^\dagger U^\dagger = U \Lambda^\dagger \Lambda U^\dagger.
   $$
   したがって
   $$
   A A^\dagger = A^\dagger A.
   $$
   よって $A$ は正規行列です。

以上より、

- (1) $A$ が正規行列ならば、ユニタリ行列で対角化可能。
- (2) ユニタリ行列で対角化可能ならば、$A$ は正規行列。

が示されたので、2つの条件は同値です。


---

__例題:__

次の行列が正規行列であることを示し、ユニタリ行列によって対角化せよ。

(1) $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} $  &emsp;&emsp;&emsp;&emsp; (2) $\begin{pmatrix} a & -b \\ b & a \end{pmatrix} $ ($a,b$実数)


---

__(1) $A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$__

__1. 正規行列であることの確認__

$A$ は実行列なので、随伴行列は転置行列 $A^\mathsf{T}$ です。

$$
A A^\mathsf{T} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I,
$$
$$
A^\mathsf{T} A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I.
$$

よって $A A^\mathsf{T} = A^\mathsf{T} A$ が成り立ち、$A$ は正規行列です。

__2. 固有値と固有ベクトル__

固有方程式は
$$
\det(A - \lambda I) = \det\begin{pmatrix} -\lambda & 1 \\ 1 & -\lambda \end{pmatrix} = \lambda^2 - 1 = 0
$$
より、固有値は
$$
\lambda_1 = 1,\quad \lambda_2 = -1.
$$

- $\lambda_1 = 1$ のとき：
  $$
  (A - I)\mathbf{v} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\mathbf{v} = \mathbf{0}
  $$
  より、$v_1 = v_2$。固有ベクトルとして
  $$
  \mathbf{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
  $$
  が取れます。

- $\lambda_2 = -1$ のとき：
  $$
  (A + I)\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\mathbf{v} = \mathbf{0}
  $$
  より、$v_1 = -v_2$。固有ベクトルとして
  $$
  \mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}
  $$
  が取れます。

これらは直交します：
$$
\mathbf{v}_1 \cdot \mathbf{v}_2 = 1\cdot 1 + 1\cdot(-1) = 0.
$$

__3. 正規化とユニタリ行列による対角化__

長さを正規化します：
- $\|\mathbf{v}_1\| = \sqrt{2}$ → $\mathbf{u}_1 = \dfrac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}$
- $\|\mathbf{v}_2\| = \sqrt{2}$ → $\mathbf{u}_2 = \dfrac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

ユニタリ行列（実数なので直交行列）を
$$
U = [\mathbf{u}_1 \ \mathbf{u}_2] = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
$$
とおくと、$U^\mathsf{T} U = I$ であり、
$$
U^{-1} A U = U^\mathsf{T} A U = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$
となります。


__(2) $A = \begin{pmatrix} a & -b \\ b & a \end{pmatrix}$（$a,b$ は実数）__

__1. 正規行列であることの確認__

$A$ は実行列なので、随伴行列は転置行列 $A^\mathsf{T}$ です。

$$
A A^\mathsf{T} = \begin{pmatrix} a & -b \\ b & a \end{pmatrix} \begin{pmatrix} a & b \\ -b & a \end{pmatrix}
= \begin{pmatrix} a^2 + b^2 & 0 \\ 0 & a^2 + b^2 \end{pmatrix},
$$
$$
A^\mathsf{T} A = \begin{pmatrix} a & b \\ -b & a \end{pmatrix} \begin{pmatrix} a & -b \\ b & a \end{pmatrix}
= \begin{pmatrix} a^2 + b^2 & 0 \\ 0 & a^2 + b^2 \end{pmatrix}.
$$

よって $A A^\mathsf{T} = A^\mathsf{T} A$ が成り立ち、$A$ は正規行列です。

__2. 固有値と固有ベクトル__

固有方程式は
$$
\det(A - \lambda I) = \det\begin{pmatrix} a-\lambda & -b \\ b & a-\lambda \end{pmatrix} = (a-\lambda)^2 + b^2 = 0
$$
より、固有値は
$$
\lambda = a \pm i b.
$$

- $\lambda_1 = a + i b$ のとき：
  $$
  (A - \lambda_1 I)\mathbf{v} = \begin{pmatrix} -ib & -b \\ b & -ib \end{pmatrix}\mathbf{v} = \mathbf{0}
  $$
  第1行より $-i b v_1 - b v_2 = 0$ → $v_2 = -i v_1$。  
  固有ベクトルとして
  $$
  \mathbf{v}_1 = \begin{pmatrix} 1 \\ -i \end{pmatrix}
  $$
  が取れます。

- $\lambda_2 = a - i b$ のとき：
  $$
  (A - \lambda_2 I)\mathbf{v} = \begin{pmatrix} ib & -b \\ b & ib \end{pmatrix}\mathbf{v} = \mathbf{0}
  $$
  第1行より $i b v_1 - b v_2 = 0$ → $v_2 = i v_1$。  
  固有ベクトルとして
  $$
  \mathbf{v}_2 = \begin{pmatrix} 1 \\ i \end{pmatrix}
  $$
  が取れます。

これらは複素内積に関して直交します：
$$
\langle \mathbf{v}_1, \mathbf{v}_2 \rangle = \mathbf{v}_2^\dagger \mathbf{v}_1 = \begin{pmatrix} 1 & -i \end{pmatrix} \begin{pmatrix} 1 \\ -i \end{pmatrix} = 1\cdot 1 + (-i)\cdot(-i) = 1 - 1 = 0.
$$

__3. 正規化とユニタリ行列による対角化__

長さを正規化します：
- $\|\mathbf{v}_1\|^2 = 1^2 + |-i|^2 = 1 + 1 = 2$ → $\mathbf{u}_1 = \dfrac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -i \end{pmatrix}$
- $\|\mathbf{v}_2\|^2 = 1^2 + |i|^2 = 1 + 1 = 2$ → $\mathbf{u}_2 = \dfrac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ i \end{pmatrix}$

ユニタリ行列を
$$
U = [\mathbf{u}_1 \ \mathbf{u}_2] = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ -i & i \end{pmatrix}
$$
とおくと、$U^\dagger U = I$ であり、
$$
U^{-1} A U = U^\dagger A U = \begin{pmatrix} a + i b & 0 \\ 0 & a - i b \end{pmatrix}
$$
となります。

---


__定理:__

$\lambda$を正規行列 $A$ の固有値、$\mathbf{x}$を固有値$\lambda$に属する固有ベクトルとすると、
$\bar{\lambda}$は$A^*$の固有値で$\mathbf{x}$は$A^*$の固有値$\bar{\lambda}$に属する固有ベクトルである。



---

証明

1. **正規性の仮定**  
   $A$ は正規行列なので、
   $$
   A A^* = A^* A
   $$
   が成り立ちます。

2. **$A^* \mathbf{x}$ が $\mathbf{x}$ と直交するベクトルと直交することを示す**  
   任意のベクトル $\mathbf{y} \in \mathbb{C}^n$ をとり、内積 $\langle \cdot, \cdot \rangle$ を標準内積（$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{v}^* \mathbf{u}$）とします。  
   まず、
   $$
   \langle A^*\mathbf{x}, (A - \lambda I)\mathbf{y} \rangle
   $$
   を2通りに計算します。

   - 一つ目：
     $$
     \langle A^*\mathbf{x}, (A - \lambda I)\mathbf{y} \rangle
     = \langle \mathbf{x}, A(A - \lambda I)\mathbf{y} \rangle
     = \langle \mathbf{x}, (A^2 - \lambda A)\mathbf{y} \rangle.
     $$

   - 二つ目：正規性 $A A^* = A^* A$ より、
     $$
     \langle A^*\mathbf{x}, (A - \lambda I)\mathbf{y} \rangle
     = \langle \mathbf{x}, A^*(A - \lambda I)\mathbf{y} \rangle
     = \langle \mathbf{x}, (A^*A - \lambda A^*)\mathbf{y} \rangle.
     $$

   したがって、任意の $\mathbf{y}$ に対して
   $$
   \langle \mathbf{x}, (A^2 - \lambda A)\mathbf{y} \rangle = \langle \mathbf{x}, (A^*A - \lambda A^*)\mathbf{y} \rangle.
   $$
   すなわち
   $$
   \langle \mathbf{x}, (A^2 - A^*A - \lambda A + \lambda A^*)\mathbf{y} \rangle = 0.
   $$
   正規性 $A A^* = A^* A$ より $A^2 - A^*A = A(A - A^*) = 0$ ではありませんが、より直接的に次のように進めます。

3. **別のアプローチ：$(A^* - \overline{\lambda} I)\mathbf{x}$ のノルムを計算**  
   より簡潔な方法として、$(A^* - \overline{\lambda} I)\mathbf{x}$ のノルムの2乗を計算します。

   $$
   \|(A^* - \overline{\lambda} I)\mathbf{x}\|^2
   = \langle (A^* - \overline{\lambda} I)\mathbf{x}, (A^* - \overline{\lambda} I)\mathbf{x} \rangle.
   $$

   内積を展開すると、
   $$
   = \langle A^*\mathbf{x}, A^*\mathbf{x} \rangle
     - \overline{\lambda} \langle A^*\mathbf{x}, \mathbf{x} \rangle
     - \lambda \langle \mathbf{x}, A^*\mathbf{x} \rangle
     + |\lambda|^2 \langle \mathbf{x}, \mathbf{x} \rangle.
   $$

   ここで、$\langle A^*\mathbf{x}, \mathbf{x} \rangle = \langle \mathbf{x}, A\mathbf{x} \rangle = \langle \mathbf{x}, \lambda \mathbf{x} \rangle = \lambda \|\mathbf{x}\|^2$ より、
   $$
   \overline{\lambda} \langle A^*\mathbf{x}, \mathbf{x} \rangle = \overline{\lambda} \lambda \|\mathbf{x}\|^2 = |\lambda|^2 \|\mathbf{x}\|^2,
   $$
   同様に $\lambda \langle \mathbf{x}, A^*\mathbf{x} \rangle = |\lambda|^2 \|\mathbf{x}\|^2$ です。

   また、正規性 $A A^* = A^* A$ より、
   $$
   \langle A^*\mathbf{x}, A^*\mathbf{x} \rangle
   = \langle A A^*\mathbf{x}, \mathbf{x} \rangle
   = \langle A^* A\mathbf{x}, \mathbf{x} \rangle
   = \langle A^*(\lambda \mathbf{x}), \mathbf{x} \rangle
   = \lambda \langle A^*\mathbf{x}, \mathbf{x} \rangle
   = |\lambda|^2 \|\mathbf{x}\|^2.
   $$

   したがって、
   $$
   \|(A^* - \overline{\lambda} I)\mathbf{x}\|^2
   = |\lambda|^2 \|\mathbf{x}\|^2 - |\lambda|^2 \|\mathbf{x}\|^2 - |\lambda|^2 \|\mathbf{x}\|^2 + |\lambda|^2 \|\mathbf{x}\|^2 = 0.
   $$

4. **結論**  
   $\|(A^* - \overline{\lambda} I)\mathbf{x}\| = 0$ より、
   $$
   (A^* - \overline{\lambda} I)\mathbf{x} = \mathbf{0},
   $$
   すなわち
   $$
   A^*\mathbf{x} = \overline{\lambda} \mathbf{x}.
   $$
   よって、$\overline{\lambda}$ は $A^*$ の固有値であり、$\mathbf{x}$ はその固有ベクトルです。

- この結果から、正規行列 $A$ の固有ベクトルは、$A$ と $A^*$ の両方の固有ベクトルになることがわかります。
- 特に、エルミート行列（$A^* = A$）の場合、$\overline{\lambda} = \lambda$ なので固有値は実数であり、固有ベクトルは $A$ と $A^*$ で共通です。

以上で定理が証明されました。

---


__定理:__

正規行列 $A$ の相異なる固有値に属する固有ベクトルは互いに直交する。




---

証明

以下では、$A$ を $n$ 次複素正規行列（$A A^\dagger = A^\dagger A$）とし、$\lambda, \mu$ を $A$ の相異なる固有値、$\mathbf{v}, \mathbf{w}$ をそれぞれ対応する固有ベクトルとします：
$$
A\mathbf{v} = \lambda \mathbf{v}, \quad A\mathbf{w} = \mu \mathbf{w}, \quad \mathbf{v}, \mathbf{w} \neq \mathbf{0}.
$$
このとき、$\mathbf{v}$ と $\mathbf{w}$ が直交する、すなわち
$$
\langle \mathbf{v}, \mathbf{w} \rangle = 0
$$
であることを示します。

1. **内積の定義と仮定**  
   複素標準内積を $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{y}^\dagger \mathbf{x}$ と定義します。  
   固有ベクトルの条件は
   $$
   A\mathbf{v} = \lambda \mathbf{v}, \quad A\mathbf{w} = \mu \mathbf{w}.
   $$

2. **$\langle A\mathbf{v}, \mathbf{w} \rangle$ を2通りに計算する**

   - 一つ目の計算：
     $$
     \langle A\mathbf{v}, \mathbf{w} \rangle = \langle \lambda \mathbf{v}, \mathbf{w} \rangle = \lambda \langle \mathbf{v}, \mathbf{w} \rangle.
     $$

   - 二つ目の計算：  
     正規性 $A A^\dagger = A^\dagger A$ より、$A$ は正規行列です。  
     一般に正規行列に対しては、随伴行列 $A^\dagger$ の固有値は固有ベクトル $\mathbf{v}$ に対応して $\overline{\lambda}$ であることが知られています（前問で示した定理）。  
     ここではより直接的に、内積の性質を用います：
     $$
     \langle A\mathbf{v}, \mathbf{w} \rangle = \mathbf{w}^\dagger A\mathbf{v} = \mathbf{w}^\dagger A^\dagger \mathbf{v} = (A\mathbf{w})^\dagger \mathbf{v} = \langle \mathbf{v}, A\mathbf{w} \rangle.
     $$
     ここで $A\mathbf{w} = \mu \mathbf{w}$ だから、
     $$
     \langle \mathbf{v}, A\mathbf{w} \rangle = \langle \mathbf{v}, \mu \mathbf{w} \rangle = \mu \langle \mathbf{v}, \mathbf{w} \rangle.
     $$

3. **2つの式を比較する**  
   以上より、
   $$
   \lambda \langle \mathbf{v}, \mathbf{w} \rangle = \mu \langle \mathbf{v}, \mathbf{w} \rangle.
   $$
   移項して
   $$
   (\lambda - \mu) \langle \mathbf{v}, \mathbf{w} \rangle = 0.
   $$

4. **結論**  
   $\lambda \neq \mu$ より $\lambda - \mu \neq 0$ なので、両辺を $\lambda - \mu$ で割って
   $$
   \langle \mathbf{v}, \mathbf{w} \rangle = 0.
   $$
   すなわち、$\mathbf{v}$ と $\mathbf{w}$ は直交します。

---


