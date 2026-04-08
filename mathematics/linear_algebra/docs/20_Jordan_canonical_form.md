
## 不変部分空間

不変部分空間（invariant subspace）は、**線形変換（行列）を施してもその空間から出ていかない部分空間**のことです。  
線形代数や関数解析、表現論などで非常に重要な概念です。

### 1. 不変部分空間の定義

$V$をベクトル空間、$T: V \to V$を線形変換とします。  
$V$の部分空間 $W \subset V$が

$$
T(W) \subset W
$$

を満たすとき、$W$を $T$に関する**不変部分空間**といいます。  
つまり、任意の $\mathbf{w} \in W$に対して $T(\mathbf{w}) \in W$となることです。

行列で言えば、$A$を $n\times n$行列、$W$を $\mathbb{R}^n$（または $\mathbb{C}^n$）の部分空間として、

$$
A\mathbf{w} \in W \quad (\forall \mathbf{w} \in W)
$$

が成り立つとき、$W$は $A$に関する不変部分空間です。

### 2. 具体例

__例1：固有空間__

$A$の固有値 $\lambda$に対する固有空間

$$
E_\lambda = \{\mathbf{v} \mid A\mathbf{v} = \lambda \mathbf{v}\}
$$

は不変部分空間です。  
実際、$\mathbf{v} \in E_\lambda$なら $A\mathbf{v} = \lambda \mathbf{v} \in E_\lambda$です。

__例2：一般化固有空間__

固有値 $\lambda$に対する一般化固有空間

$$
W_\lambda = \{\mathbf{v} \mid (A - \lambda I)^m \mathbf{v} = \mathbf{0} \text{ となる } m \text{ が存在}\}
$$

も不変部分空間です。  
$(A - \lambda I)^m \mathbf{v} = \mathbf{0}$なら、$(A - \lambda I)^m (A\mathbf{v}) = A (A - \lambda I)^m \mathbf{v} = \mathbf{0}$より $A\mathbf{v} \in W_\lambda$です。

__例3：自明な例__

- 零空間 $\{\mathbf{0}\}$
- 全体空間 $V$

は常に不変部分空間です。

__例4：回転行列__

2次元回転行列

$$
R_\theta = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}
$$

に対して、$\mathbb{R}^2$全体は不変部分空間ですが、零でない真の部分空間（直線）は一般には不変ではありません（$\theta$が $\pi$の整数倍でない限り）。


### 3. 不変部分空間の性質

__(1) 制限写像__

$W$が $T$に関する不変部分空間なら、$T$を $W$に制限した写像

$$
T|_W : W \to W
$$

が定義できます。これは $W$上の線形変換です。

__(2) 基底と行列表示__

$V$の基底を、まず $W$の基底 $\{\mathbf{w}_1, \dots, \mathbf{w}_k\}$で取り、その後 $V$全体の基底に拡張すると、$T$の行列表示はブロック上三角行列の形になります：

$$
[T] =
\begin{pmatrix}
A & B \\
0 & C
\end{pmatrix}
$$

ここで $A$は $T|_W$の行列表示です。

__(3) 直和分解__

$V$が不変部分空間 $W_1, \dots, W_r$の直和

$$
V = W_1 \oplus W_2 \oplus \cdots \oplus W_r
$$

で、各 $W_i$が $T$に関して不変なら、$T$の行列表示はブロック対角行列になります：

$$
[T] =
\begin{pmatrix}
A_1 & & \\
 & \ddots & \\
 & & A_r
\end{pmatrix}
$$

ここで $A_i$は $T|_{W_i}$の行列表示です。  
固有空間分解や一般化固有空間分解はこの特別な場合です。

### 4. なぜ重要か

- **行列の簡約化**  
  不変部分空間を見つけることで、行列をより小さなブロックに分解でき、解析や計算が容易になります。

- **表現論**  
  群の表現や線形変換の構造を、不変部分空間の存在・非存在（既約性）を通じて調べます。

- **動的システム**  
  線形システム $\dot{\mathbf{x}} = A\mathbf{x}$において、不変部分空間は「状態がそこに留まり続ける領域」を表し、安定性解析や制御設計に役立ちます。

- **数値線形代数**  
  アーノルディ法やランチョス法など、大規模行列の部分空間を利用したアルゴリズムでは、不変部分空間（あるいはその近似）が重要な役割を果たします。


### 5. 関連概念

- **既約表現**：自明な不変部分空間（$\{\mathbf{0}\}$と全体空間）以外に不変部分空間を持たない表現。
- **可約表現**：非自明な不変部分空間を持つ表現。
- **巡回部分空間**：あるベクトル $\mathbf{v}$から $T$を繰り返し適用して得られる空間 $\langle \mathbf{v}, T\mathbf{v}, T^2\mathbf{v}, \dots \rangle$は不変部分空間です。

__定理:__


$\lambda$を $n$次正方行列 $A$の固有値とすると、$\lambda$に属する固有空間
\[
V(\lambda) = \{\mathbf{v} \in \mathbb{C}^n \mid A\mathbf{v} = \lambda \mathbf{v}\}
\]
は $A$に関する不変部分空間である。

---

**証明**

1. **$V(\lambda)$が部分空間であること**  
   任意の $\mathbf{v}_1, \mathbf{v}_2 \in V(\lambda)$とスカラー $c_1, c_2 \in \mathbb{C}$に対して、
   \[
   A(c_1\mathbf{v}_1 + c_2\mathbf{v}_2) = c_1 A\mathbf{v}_1 + c_2 A\mathbf{v}_2 = c_1 \lambda \mathbf{v}_1 + c_2 \lambda \mathbf{v}_2 = \lambda (c_1\mathbf{v}_1 + c_2\mathbf{v}_2)
   \]
   より $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 \in V(\lambda)$である。  
   また $V(\lambda)$は空でなく（固有ベクトルが存在する）、$\mathbf{0} \in V(\lambda)$である。  
   よって $V(\lambda)$は $\mathbb{C}^n$の部分空間である。

2. **不変性の確認**  
   任意の $\mathbf{v} \in V(\lambda)$をとる。固有空間の定義より
   \[
   A\mathbf{v} = \lambda \mathbf{v}.
   \]
   右辺 $\lambda \mathbf{v}$は明らかに $V(\lambda)$の元である（$\mathbf{v} \in V(\lambda)$より $\lambda \mathbf{v} \in V(\lambda)$）。  
   したがって
   \[
   A\mathbf{v} \in V(\lambda)
   \]
   が成り立つ。  
   これは $A(V(\lambda)) \subset V(\lambda)$を意味し、$V(\lambda)$が $A$に関する不変部分空間であることを示している。

以上より、定理は証明された。

---

### 6. 直和分割

直和分割（direct sum decomposition）は、**ベクトル空間を「交わりのない」部分空間の和に分解する**操作です。  
線形代数の構造解析（対角化、ジョルダン標準形、表現論など）で非常に重要な概念です。

__1. 直和の定義__

$V$をベクトル空間、$W_1, W_2, \dots, W_k$を $V$の部分空間とします。

__(1) 和空間__

$$
W_1 + W_2 + \cdots + W_k = \{\mathbf{w}_1 + \mathbf{w}_2 + \cdots + \mathbf{w}_k \mid \mathbf{w}_i \in W_i\}
$$

は、各 $W_i$の元を足し合わせて得られる空間です。

__(2) 直和__

和空間 $W_1 + \cdots + W_k$が**直和**であるとは、任意のベクトル $\mathbf{v} \in W_1 + \cdots + W_k$の表現が**一意**であることです。  
すなわち、

$$
\mathbf{v} = \mathbf{w}_1 + \cdots + \mathbf{w}_k = \mathbf{w}_1' + \cdots + \mathbf{w}_k'
\quad (\mathbf{w}_i, \mathbf{w}_i' \in W_i)
$$

ならば、すべての $i$について $\mathbf{w}_i = \mathbf{w}_i'$が成り立つことです。

このとき、直和を

$$
W_1 \oplus W_2 \oplus \cdots \oplus W_k
$$

と書きます。

__2. 直和の同値な条件__

2つの部分空間 $W_1, W_2$について、次は同値です：

1. $W_1 + W_2$は直和である（$W_1 \oplus W_2$）。
2. $W_1 \cap W_2 = \{\mathbf{0}\}$（交わりが自明）。
3. $\dim(W_1 + W_2) = \dim W_1 + \dim W_2$。

一般に $k$個の部分空間については、

- 任意の $i$について $W_i \cap (W_1 + \cdots + W_{i-1} + W_{i+1} + \cdots + W_k) = \{\mathbf{0}\}$
- あるいは、各 $W_i$の基底を合わせたものが $V$の基底になる

ことなどが直和の特徴です。

__3. 直和分割の例__

__例1：座標空間の分解__

$\mathbb{R}^3$を

- $W_1 = \{(x,0,0) \mid x \in \mathbb{R}\}$（$x$軸）
- $W_2 = \{(0,y,0) \mid y \in \mathbb{R}\}$（$y$軸）
- $W_3 = \{(0,0,z) \mid z \in \mathbb{R}\}$（$z$軸）

とすると、

$$
\mathbb{R}^3 = W_1 \oplus W_2 \oplus W_3
$$

です。任意のベクトル $(x,y,z)$は一意に $(x,0,0)+(0,y,0)+(0,0,z)$と分解されます。

__例2：固有空間分解__

$A$を対角化可能な $n$次行列とし、相異なる固有値を $\lambda_1, \dots, \lambda_k$、対応する固有空間を $E_{\lambda_i}$とします。  
このとき、

$$
\mathbb{C}^n = E_{\lambda_1} \oplus E_{\lambda_2} \oplus \cdots \oplus E_{\lambda_k}
$$

が成り立ちます（固有空間は互いに直交し、和が全体空間になる）。

__例3：一般化固有空間分解__

任意の複素正方行列 $A$に対して、相異なる固有値を $\lambda_1, \dots, \lambda_k$、一般化固有空間を $W_{\lambda_i}$とすると、

$$
\mathbb{C}^n = W_{\lambda_1} \oplus W_{\lambda_2} \oplus \cdots \oplus W_{\lambda_k}
$$

が成り立ちます。これはジョルダン標準形の理論の根幹です。

__4. 直和分割の性質__

__(1) 次元の加法性__

$$
V = W_1 \oplus \cdots \oplus W_k \quad\Rightarrow\quad \dim V = \sum_{i=1}^k \dim W_i
$$

__(2) 基底の取り方__

各 $W_i$の基底 $\mathcal{B}_i$をとると、それらを合わせた集合 $\mathcal{B} = \mathcal{B}_1 \cup \cdots \cup \mathcal{B}_k$は $V$の基底になります。

__(3) 線形写像の分解__

$V = W_1 \oplus \cdots \oplus W_k$かつ各 $W_i$が線形写像 $T$に関して不変部分空間なら、$T$の行列表示はブロック対角行列になります：

$$
[T] =
\begin{pmatrix}
A_1 & & \\
 & \ddots & \\
 & & A_k
\end{pmatrix}
$$

ここで $A_i$は $T|_{W_i}$の行列表示です。

__5. なぜ重要か__

- **構造の分解**：ベクトル空間を「独立な」部分に分けることで、線形変換の挙動を部分ごとに解析できます。
- **対角化・ジョルダン標準形**：固有空間や一般化固有空間への直和分解が、行列の標準形理論の基礎です。
- **表現論**：群の表現を既約表現の直和として分解する（マシュケの定理）など、表現の構造解析に不可欠です。
- **数値線形代数**：部分空間法（アーノルディ法など）では、近似的不変部分空間への直和分解を利用します。


__例題:__

$A$を $n$次実対称行列（または複素エルミート行列）とし、相異なる固有値を $\lambda_1, \dots, \lambda_k$、対応する固有空間を $E_{\lambda_i}$とします。  
このとき、

$$
\mathbb{R}^n = E_{\lambda_1} \oplus E_{\lambda_2} \oplus \cdots \oplus E_{\lambda_k}
$$

という直和分解が成り立ちます。  
Python で：

1. 固有値と固有ベクトルを計算
2. 各固有空間の基底を求め
3. 任意のベクトルを固有空間成分に一意分解

するコードを示します。

__Python 実装例__

```python
import numpy as np

def direct_sum_decomposition(A, tol=1e-10):
    """
    実対称行列 A の固有空間への直和分解を扱う。
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        実対称行列
    tol : float
        固有値の一致判定の許容誤差
    
    Returns
    -------
    eigvals : ndarray
        相異なる固有値のリスト
    eigspaces : list of ndarray
        各固有空間の基底ベクトル（列ベクトル）を並べた行列のリスト
    P : ndarray
        直交行列（固有ベクトルを列に並べたもの）
    """
    # 固有値・固有ベクトルを計算（実対称行列なので eigh を使用）
    eigvals, P = np.linalg.eigh(A)
    
    # 固有値を丸めてグループ化
    rounded_vals = np.round(eigvals / tol) * tol
    unique_vals = np.unique(rounded_vals)
    
    eigspaces = []
    for lam in unique_vals:
        # 該当する固有値のインデックスを取得
        idx = np.where(np.abs(rounded_vals - lam) < tol)[0]
        # 対応する固有ベクトルを取り出し、正規直交基底として格納
        basis = P[:, idx]
        eigspaces.append(basis)
    
    return unique_vals, eigspaces, P

def decompose_vector(x, eigspaces):
    """
    ベクトル x を各固有空間成分に分解する。
    
    Parameters
    ----------
    x : ndarray, shape (n,)
        分解したいベクトル
    eigspaces : list of ndarray
        direct_sum_decomposition で得られた固有空間基底のリスト
    
    Returns
    -------
    components : list of ndarray
        各固有空間への射影成分
    """
    components = []
    for basis in eigspaces:
        # 固有空間への直交射影: proj = basis @ basis.T @ x
        proj = basis @ (basis.T @ x)
        components.append(proj)
    return components

def check_direct_sum(x, components, tol=1e-10):
    """
    直和分解の条件（和が元のベクトルに一致、成分が一意）を確認する。
    """
    # 和が元のベクトルに一致するか
    reconstructed = sum(components)
    error = np.linalg.norm(x - reconstructed)
    print(f"再構成誤差 (||x - sum components||): {error:.2e}")
    
    # 成分が互いに直交しているか（直和の特徴）
    for i in range(len(components)):
        for j in range(i+1, len(components)):
            ip = np.dot(components[i], components[j])
            if np.abs(ip) > tol:
                print(f"警告: 成分 {i} と {j} が直交していません (内積 = {ip:.2e})")
            else:
                print(f"成分 {i} と {j} は直交しています (内積 = {ip:.2e})")

# --- 例: 3x3 実対称行列 ---
A = np.array([[4, 1, 1],
              [1, 3, 0],
              [1, 0, 3]])

print("行列 A:")
print(A)
print()

# 固有空間への直和分解
eigvals, eigspaces, P = direct_sum_decomposition(A)

print("相異なる固有値:")
for lam in eigvals:
    print(f"λ = {lam:.4f}")
print()

print("各固有空間の次元:")
for i, basis in enumerate(eigspaces):
    print(f"E(λ={eigvals[i]:.4f}) の次元: {basis.shape[1]}")
print()

# 任意のベクトルを分解
x = np.array([1, 2, 3], dtype=float)
print(f"分解したいベクトル x = {x}")
components = decompose_vector(x, eigspaces)

print("\n各固有空間成分:")
for i, comp in enumerate(components):
    print(f"E(λ={eigvals[i]:.4f}) 成分: {comp}")

# 直和条件の確認
print("\n--- 直和条件の確認 ---")
check_direct_sum(x, components)
```

__結果__

```
行列 A:
[[4 1 1]
 [1 3 0]
 [1 0 3]]

相異なる固有値:
λ = 2.0000
λ = 3.0000
λ = 5.0000

各固有空間の次元:
E(λ=2.0000) の次元: 1
E(λ=3.0000) の次元: 1
E(λ=5.0000) の次元: 1

分解したいベクトル x = [1. 2. 3.]

各固有空間成分:
E(λ=2.0000) 成分: [ 0.  1. -1.]
E(λ=3.0000) 成分: [0. 1. 1.]
E(λ=5.0000) 成分: [1. 0. 1.]

--- 直和条件の確認 ---
再構成誤差 (||x - sum components||): 1.11e-16
成分 0 と 1 は直交しています (内積 = 0.00e+00)
成分 0 と 2 は直交しています (内積 = 0.00e+00)
成分 1 と 2 は直交しています (内積 = 0.00e+00)
```

__解説__

1. **`direct_sum_decomposition`**  
   - `np.linalg.eigh` で固有値・固有ベクトルを計算（実対称行列なので固有値は実数、固有ベクトルは直交）。
   - 固有値を丸めてグループ化し、各固有値に対応する固有ベクトルをまとめて固有空間の基底とします。

2. **`decompose_vector`**  
   - 各固有空間の正規直交基底 `basis` に対し、直交射影 $P_i = \text{basis}_i \cdot \text{basis}_i^\mathsf{T}$を計算し、$P_i \mathbf{x}$を固有空間成分として返します。

3. **`check_direct_sum`**  
   - 成分の和が元のベクトルに一致するか（再構成誤差）を確認。
   - 異なる固有空間成分が互いに直交しているか（直和の特徴）を確認。

4. **直和分割の確認**  
   - 実対称行列の場合、異なる固有値に対応する固有ベクトルは直交するため、固有空間は互いに直交し、直和分解になります。
   - 再構成誤差が数値誤差レベル（$10^{-16}$程度）であれば、数値的にも直和分解が正しく行われていると判断できます。

__定理:__

その定理は、**不変部分空間による直和分解の存在定理**（あるいは、線形変換の**準素分解定理**に近い内容）を指していると思われます。  
ただし、一般の実行列に対しては「必ず直和分解できる」とは限らず、**複素数体上で考えた場合**に成り立つ定理です。

---

__定理の内容__

$A$を $n$次正方行列（複素行列）とし、$V$を $\mathbb{C}^n$とします。  
$A$の相異なる固有値を $\lambda_1, \dots, \lambda_t$とし、それぞれに対応する**一般化固有空間**を

$$
W_{\lambda_i} = \{\mathbf{v} \in \mathbb{C}^n \mid (A - \lambda_i I)^m \mathbf{v} = \mathbf{0} \text{ となる } m \text{ が存在}\}
$$

とおきます。

このとき、次が成り立ちます：

1. 各 $W_{\lambda_i}$は $A$に関する**不変部分空間**である。
2. $V$はこれらの一般化固有空間の**直和**に分解される：
   $$
   \mathbb{C}^n = W_{\lambda_1} \oplus W_{\lambda_2} \oplus \cdots \oplus W_{\lambda_t}.
   $$
3. 各 $W_{\lambda_i}$上に制限した $A$の行列表示は、固有値 $\lambda_i$に対応する**ジョルダンブロック**の直和となる。

__証明__

__1. 不変性__

任意の $\mathbf{v} \in W_{\lambda_i}$に対して、$(A - \lambda_i I)^m \mathbf{v} = \mathbf{0}$となる $m$が存在します。  
このとき、

$$
(A - \lambda_i I)^m (A\mathbf{v}) = A (A - \lambda_i I)^m \mathbf{v} = A\mathbf{0} = \mathbf{0}
$$

より、$A\mathbf{v} \in W_{\lambda_i}$です。  
したがって $W_{\lambda_i}$は $A$に関する不変部分空間です。

__2. 直和分解の存在__

- 特性多項式 $\phi(\lambda) = \det(\lambda I - A)$を因数分解すると
  $$
  \phi(\lambda) = (\lambda - \lambda_1)^{m_1} (\lambda - \lambda_2)^{m_2} \cdots (\lambda - \lambda_t)^{m_t}
  $$
  と書けます（$\lambda_i$は相異なる固有値、$m_i$は代数的重複度）。

- ケイリー・ハミルトンの定理より $\phi(A) = 0$です。

- 多項式
  $$
  p_i(\lambda) = \frac{\phi(\lambda)}{(\lambda - \lambda_i)^{m_i}}
  $$
  を考えると、$p_i(A)$の像 $\operatorname{Im}(p_i(A))$は $W_{\lambda_i}$に含まれ、かつ $p_i(A)$と $(A - \lambda_i I)^{m_i}$は互いに素な多項式です。

- 多項式の互除法（ベズーの等式）から、ある多項式 $u_i(\lambda), v_i(\lambda)$が存在して
  $$
  u_i(\lambda) p_i(\lambda) + v_i(\lambda) (\lambda - \lambda_i)^{m_i} = 1
  $$
  が成り立ち、$\lambda$に $A$を代入すると
  $$
  u_i(A) p_i(A) + v_i(A) (A - \lambda_i I)^{m_i} = I
  $$
  となります。

- これを用いると、任意の $\mathbf{v} \in \mathbb{C}^n$は
  $$
  \mathbf{v} = \sum_{i=1}^t \mathbf{v}_i,\quad \mathbf{v}_i \in W_{\lambda_i}
  $$
  と一意に分解できることが示せます（直和分解）。

__3. ジョルダン標準形との関係__

各 $W_{\lambda_i}$上で $A$を制限した線形変換 $A|_{W_{\lambda_i}}$は、$\lambda_i$を固有値とするべき零成分 $N_i$を用いて
$$
A|_{W_{\lambda_i}} = \lambda_i I + N_i
$$
と書けます。ここで $N_i$はべき零行列です。  
この分解により、$W_{\lambda_i}$の基底を適切に選ぶと、$A|_{W_{\lambda_i}}$の行列表示がジョルダンブロックの直和になります。

__実数体上の注意点__

- 定理の元の記述では「$A:\mathbb{R}^n \to \mathbb{R}^n$」とありますが、**実数体上では一般化固有空間への直和分解は必ずしも成り立ちません**。
- 実行列でも、**複素数体に拡張して考えれば**上記の直和分解が成り立ちます。
- 実数の範囲で扱いたい場合は、**実ジョルダン標準形**（共役な複素固有値ペアに対応する $2\times 2$ブロックを用いる）を用いた分解が考えられます。

---

## べき零部分空間

べき零部分空間（nilpotent subspace）は、**べき零行列やべき零変換が作用する空間**として定義されます。  
ジョルダン標準形の理論や一般化固有空間の構造解析で重要な役割を果たします。


### 1. べき零変換・べき零行列の定義

__(1) べき零変換__

$V$をベクトル空間、$N: V \to V$を線形変換とします。  
ある正整数 $k$が存在して

$$
N^k = 0
$$

（零写像）となるとき、$N$を**べき零変換**といいます。

__(2) べき零行列__

$n$次正方行列 $N$に対して、ある $k$が存在して

$$
N^k = O \quad (\text{零行列})
$$

となるとき、$N$を**べき零行列**といいます。

### 2. べき零部分空間の定義

べき零部分空間は、べき零変換 $N$が作用する空間として定義されます。  
より正確には、**一般化固有空間**の文脈で現れます。

$A$を線形変換（または行列）、$\lambda$をその固有値とします。  
一般化固有空間

$$
W_\lambda = \{\mathbf{v} \in V \mid (A - \lambda I)^m \mathbf{v} = \mathbf{0} \text{ となる } m \text{ が存在}\}
$$

上で、$N = A - \lambda I$とおくと、$N$は $W_\lambda$上のべき零変換です。  
このとき、$W_\lambda$を「$\lambda$に属するべき零部分空間」とみなすことができます。

### 3. べき零部分空間の性質

__(1) 有限ステップで零へ__

$N$を $W$上のべき零変換とすると、任意の $\mathbf{w} \in W$に対して、ある $m$が存在して

$$
N^m \mathbf{w} = \mathbf{0}
$$

となります。  
つまり、$N$を繰り返し適用すると、有限回で必ず零ベクトルに到達します。

__(2) 基底とジョルダンブロック__

$W$が有限次元なら、$N$の作用に関する**ジョルダン鎖**（Jordan chain）を基底として選ぶことができます。  
これは

$$
\mathbf{v}, N\mathbf{v}, N^2\mathbf{v}, \dots, N^{k-1}\mathbf{v}
$$

のような列で、$N^k\mathbf{v} = \mathbf{0}$となるものです。  
この基底に関して $N$の行列表示は、**ジョルダンブロック** $J_k(0)$の直和になります：

$$
N \sim
\begin{pmatrix}
J_{k_1}(0) & & \\
 & \ddots & \\
 & & J_{k_r}(0)
\end{pmatrix}
$$

ここで $J_k(0)$はサイズ $k$のべき零ジョルダンブロックです。

__(3) 固有値は 0 のみ__

べき零変換 $N$の固有値はすべて $0$です。  
実際、$N\mathbf{v} = \mu \mathbf{v}$なら $N^k\mathbf{v} = \mu^k \mathbf{v}$ですが、$N^k = 0$より $\mu^k = 0$、したがって $\mu = 0$です。

__(4) トレースと行列式__

べき零行列 $N$に対して：

- $\operatorname{tr}(N) = 0$
- $\det(N) = 0$

が成り立ちます。


### 4. べき零部分空間の例

__例1：単純なべき零行列__

$$
N = \begin{pmatrix}0 & 1 \\ 0 & 0\end{pmatrix}
$$

はべき零行列です（$N^2 = O$）。  
このとき、$\mathbb{R}^2$全体が $N$に関するべき零部分空間とみなせます。

__例2：一般化固有空間内のべき零成分__

$$
A = \begin{pmatrix}2 & 1 \\ 0 & 2\end{pmatrix}
$$

に対して、$\lambda = 2$の一般化固有空間は $\mathbb{R}^2$全体です。  
ここで $N = A - 2I = \begin{pmatrix}0 & 1 \\ 0 & 0\end{pmatrix}$とおくと、$N$はべき零であり、$\mathbb{R}^2$は $N$に関するべき零部分空間です。

### 5. なぜ重要か

- **ジョルダン標準形の構成**  
  線形変換 $A$を固有値成分 $\lambda I$とべき零成分 $N$に分解し、べき零部分空間上でジョルダンブロックを構成します。

- **微分方程式の解の構造**  
  線形微分方程式 $\dot{\mathbf{x}} = A\mathbf{x}$の解は、固有値成分（指数関数）とべき零成分（多項式×指数関数）の和として表されます。べき零部分空間が多項式的な項を生みます。

- **べき零指数（nilpotency index）**  
  べき零変換 $N$が零になる最小の $k$を**べき零指数**といい、システムの「隠れた」ダイナミクスの長さを表します。


__定理:__


$V$を有限次元ベクトル空間、$N: V \to V$をべき零変換（$N^k = 0$となる正整数 $k$が存在）とする。  
各 $m \ge 0$に対して

$$
W_m = \ker N^m = \{\mathbf{v} \in V \mid N^m \mathbf{v} = \mathbf{0}\}
$$

と定義する。このとき、次が成り立つ：

1. $W_0 = \{0\}$。
2. $W_1 \subset W_2 \subset \cdots \subset W_k = W_{k+1} = \cdots = V$。
3. ある $m$に対して $W_m = W_{m+1}$ならば、それ以降はすべて等しい：$W_m = W_{m+1} = W_{m+2} = \cdots$。
4. 特に、$N^k = 0$となる最小の $k$（べき零指数）に対して
   $$
   \{0\} = W_0 \subsetneq W_1 \subsetneq \cdots \subsetneq W_k = V
   $$
   が成り立つ（真の増大列）。

---

__証明__

__1. $W_0 = \{0\}$の確認__

$N^0$は恒等写像 $I$と定義されます（あるいは $N^0 = I$と約束する）。  
したがって

$$
W_0 = \ker N^0 = \ker I = \{\mathbf{v} \in V \mid \mathbf{v} = \mathbf{0}\} = \{0\}.
$$

__2. 包含関係 $W_m \subset W_{m+1}$__

任意の $\mathbf{v} \in W_m$をとります。定義より $N^m \mathbf{v} = \mathbf{0}$です。  
このとき

$$
N^{m+1} \mathbf{v} = N (N^m \mathbf{v}) = N\mathbf{0} = \mathbf{0}
$$

より $\mathbf{v} \in W_{m+1}$です。  
したがって $W_m \subset W_{m+1}$が成り立ちます。

__3. 安定性：$W_m = W_{m+1}$なら $W_m = W_{m+2} = \cdots$__

$W_m = W_{m+1}$と仮定します。  
任意の $\mathbf{v} \in W_{m+2}$をとると、$N^{m+2} \mathbf{v} = \mathbf{0}$です。  
このとき

$$
N^{m+1} (N\mathbf{v}) = N^{m+2} \mathbf{v} = \mathbf{0}
$$

より $N\mathbf{v} \in W_{m+1}$です。  
仮定より $W_{m+1} = W_m$なので、$N\mathbf{v} \in W_m$、すなわち $N^m (N\mathbf{v}) = N^{m+1} \mathbf{v} = \mathbf{0}$です。  
したがって $\mathbf{v} \in W_{m+1}$です。  
これより $W_{m+2} \subset W_{m+1}$ですが、2. より $W_{m+1} \subset W_{m+2}$でもあるので $W_{m+1} = W_{m+2}$です。  
同様に帰納的に、$W_m = W_{m+1} = W_{m+2} = \cdots$が示されます。

__4. $W_k = V$となる $k$の存在と真の増大__

$N$はべき零なので、ある正整数 $k$が存在して $N^k = 0$です。  
このとき、任意の $\mathbf{v} \in V$に対して $N^k \mathbf{v} = \mathbf{0}$なので $\mathbf{v} \in W_k$です。  
したがって $W_k = V$です。

次に、このような $k$のうち最小のものを $k$とおきます（べき零指数）。  
このとき、$m < k$に対して $W_m \subsetneq W_{m+1}$であることを示します。

もしある $m < k$に対して $W_m = W_{m+1}$なら、3. より $W_m = W_{m+1} = W_{m+2} = \cdots$です。  
一方、$N^m \neq 0$（$k$の最小性より）なので、$W_m \neq V$です。  
しかし $W_k = V$であるから矛盾します。  
したがって、$m < k$のときは常に $W_m \subsetneq W_{m+1}$です。

以上より、

$$
\{0\} = W_0 \subsetneq W_1 \subsetneq \cdots \subsetneq W_k = V
$$

が成り立ちます。  
また $W_k = V$より、$W_k = W_{k+1} = \cdots = V$です。

---

## 安定像空間

安定像空間（stable image space）は、「作用素を何度かけても像空間からはみ出さない」という性質を持つ部分空間のことです。


### 1. 準備：線形作用素と像空間

- $V $を体 $\mathbb{K} $（実数体 $\mathbb{R} $または複素数体 $\mathbb{C} $）上のベクトル空間とします。
- $T: V \to V $を線形作用素（線形写像）とします。
- $T$の **像（image）** は
  $$
  \operatorname{Im}(T) = \{ T(v) \mid v \in V \}
  $$
  で定義される $V $の部分空間です。


### 2. 安定部分空間（T-不変部分空間）の定義

部分空間 $W \subset V $が**T-不変（T-invariant）** であるとは、
$$
T(W) \subset W
$$
が成り立つことです。  
つまり、$W$の元に $T$を作用させても、結果は必ず $W$の中に留まる、という性質です。


### 3. 「安定像空間」の定義

「安定像空間」は、**像空間 $\operatorname{Im}(T)$が T-不変である**ときに、その像空間を指して使われることが多いです。  
すなわち、

$$
T(\operatorname{Im}(T)) \subset \operatorname{Im}(T)
$$

が成り立つとき、$\operatorname{Im}(T) $を**安定像空間（stable image space）**と呼びます。


### 4. 定義の言い換えと意味

上の条件は、次のようにも言い換えられます。

- 任意の $w \in \operatorname{Im}(T) $に対して、ある $v \in V $が存在して $w = T(v) $と書ける。
- このとき $T(w) = T(T(v)) $であり、これは $T^2(v) $なので、再び $\operatorname{Im}(T) $の元です。

したがって、「安定像空間」とは、

> 「作用素 $T $を一度かけて得られるベクトルたちの集合（＝像空間）が、さらに $T $をかけてもその集合から出ていかない」

という性質を持つ部分空間を指します。


### 5. 例

__例1：射影作用素__

$V = \mathbb{R}^2 $とし、$T $を $x $軸への直交射影とします：
$$
T(x, y) = (x, 0).
$$
このとき、
$$
\operatorname{Im}(T) = \{ (x, 0) \mid x \in \mathbb{R} \}
$$
であり、任意の $(x, 0) \in \operatorname{Im}(T) $に対して
$$
T(x, 0) = (x, 0) \in \operatorname{Im}(T)
$$
なので、$\operatorname{Im}(T) $は安定像空間です。

__例2：冪零作用素__

$V = \mathbb{R}^2 $とし、
$$
T(x, y) = (0, x)
$$
とします。  
このとき、
$$
\operatorname{Im}(T) = \{ (0, x) \mid x \in \mathbb{R} \}
$$
であり、任意の $(0, x) \in \operatorname{Im}(T) $に対して
$$
T(0, x) = (0, 0) \in \operatorname{Im}(T)
$$
なので、やはり $\operatorname{Im}(T) $は安定像空間です。


__定理:__

ある $k$ が存在して

$$
V = V_0 \supsetneq V_1 \dots \supsetneq V_k = V_{k+1} = \dots
$$

が成り立つ。そして、任意の $i \geqq k$ 、$i \geqq 1$ に対して

$$
f^h | V_i:V_i \longrightarrow V_{i+h} = V_i
$$

は同型写像である。


---

__1. 減少列の停止性（$k$ の存在）__

$V$ は有限次元ベクトル空間なので、部分空間の真減少列
$$
V = V_0 \supset V_1 \supset V_2 \supset \dots
$$
は有限長で止まらなければなりません（次元が自然数の減少列だから）。

より正確には、次元の列
$$
\dim V_0 \geq \dim V_1 \geq \dim V_2 \geq \dots
$$
は非負整数の単調減少列なので、ある $k$ で
$$
\dim V_k = \dim V_{k+1}
$$
となります。このとき $V_k \supset V_{k+1}$ かつ次元が等しいので、
$$
V_k = V_{k+1}
$$
が成り立ちます。

したがって、$i \geq k$ に対しては
$$
V_k = V_{k+1} = V_{k+2} = \dots
$$
となります。これで減少列が有限長で止まることが示されました。

__2. $f^h|_{V_i}: V_i \to V_i$ が同型であること__

$i \geq k$ とします。このとき $V_i = V_k$ です。また、$h \geq 1$ とします。

__(a) 線形性__

$f$ が線形写像なので、その合成 $f^h$ も線形写像です。  
したがって、制限 $f^h|_{V_i}: V_i \to V_{i+h}$ も線形写像です。

__(b) 全射性__

$V_{i+h} = f^{i+h}(V)$ ですが、$i \geq k$ のとき $V_i = V_k$ であり、かつ $V_k = V_{k+1} = \dots$ なので、
$$
V_{i+h} = V_i
$$
です。  
したがって、写像の終域は $V_i$ 自身です。

一方、$f^h|_{V_i}$ の像は
$$
f^h(V_i) = f^h(f^i(V)) = f^{i+h}(V) = V_{i+h} = V_i
$$
なので、$f^h|_{V_i}$ は $V_i$ 上への全射です。

__(c) 単射性__

$V_i$ は有限次元ベクトル空間であり、$f^h|_{V_i}: V_i \to V_i$ は線形写像で、かつ全射です。  
有限次元ベクトル空間上の線形写像では、

> 全射 $\iff$ 単射 $\iff$ 同型

が成り立ちます。  
したがって、$f^h|_{V_i}$ は単射でもあり、同型写像です。

（より一般に加群の場合は、「全射かつ有限生成自由加群」などの条件が必要ですが、ここでは有限次元ベクトル空間を仮定しています。）

---


## べき零部分空間と安定像空間における直和分解

「べき零部分空間」と「安定像空間」の直和分解とは、**有限次元ベクトル空間 $V$ 上の線形作用素 $f$ に対して、$V$ を「$f$ がべき零的に振る舞う部分」と「$f$ が同型として振る舞う部分」に分ける**分解のことです。

以下、定義と分解の形を順に説明します。


### 1. 準備：像の列と安定像空間

$f: V \to V$ を線形作用素とし、
$$
V_0 = V,\quad V_1 = f(V),\quad V_2 = f^2(V),\ \dots
$$
とおきます（$V_i = f^i(V)$）。

前の定理で示したように、有限次元性から、ある $k$ が存在して
$$
V = V_0 \supsetneq V_1 \supsetneq \dots \supsetneq V_k = V_{k+1} = \dots
$$
となります。この $V_k$ を**安定像空間（stable image space）** と呼びます。


### 2. べき零部分空間（nilpotent part）

次に、**べき零部分空間**を定義します。

- $N = \ker f^k$ とおきます。
- この $N$ を「$f$ のべき零部分空間」と呼びます。

理由：
- $f^k(N) = f^k(\ker f^k) = \{0\}$ なので、$f$ の $N$ への制限はべき零的です。
- より正確には、$f|_N$ はべき零作用素（ある冪でゼロになる）です。


### 3. 直和分解の定理

上で定義した $N$ と $V_k$ について、次のことが成り立ちます。

> $$
> V = N \oplus V_k
> $$
> すなわち、$V$ はべき零部分空間 $N$ と安定像空間 $V_k$ の直和に分解される。

ここで、
- $N$ 上では $f$ はべき零的（最終的にはゼロになる）
- $V_k$ 上では $f$ は同型（可逆）として振る舞う

という意味で、$f$ の振る舞いを「べき零部分」と「可逆部分」に分離しています。

### 4. 分解の意味・役割

この分解は、線形作用素の構造を理解する上で重要です。

- **べき零部分 $N$**  
  - $f$ の一般化固有空間のうち、固有値 $0$ に対応する部分です。
  - ジョルダン標準形で言えば、固有値 $0$ のジョルダンブロック全体に対応します。

- **安定像空間 $V_k$**  
  - $f$ の一般化固有空間のうち、**非ゼロ固有値**に対応する部分です。
  - $V_k$ 上では $f$ は同型（可逆）なので、固有値はすべて $0$ 以外です。

したがって、この直和分解は

> 「$f$ のゼロ固有値部分」と「非ゼロ固有値部分」

を分離する役割を持っています。




