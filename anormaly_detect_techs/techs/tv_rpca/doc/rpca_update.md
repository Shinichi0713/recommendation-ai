異常検知などで利用される **RPCA (Robust Principal Component Analysis)** は、観測行列 **$M$** を「低ランク行列 **$L$**」と「スパース行列 **$S$**」に分解する手法です。一般的に、**ALM (Augmented Lagrangian Multiplier: 拡張ラグランジュ乗数法)** 、特に **IALM (Inexact ALM)** を用いて解かれます。

以下に、その更新式の導出プロセスをまとめます。

---

## 1. 最適化問題の定義

まず、RPCAの主問題は次のように定式化されます。

$$
\min_{L, S} \|L\|_* + \lambda \|S\|_1 \quad \text{subject to} \quad M = L + S
$$

ここで、**$\|L\|_*$** は核ノルム（特異値の和）、**$\|S\|_1$** は **$L_1$** ノルム、**$\lambda$** はスパース性を制御するハイパーパラメータです。

---

## 2. 拡張ラグランジュ関数の構成

制約付き最適化問題を解くために、拡張ラグランジュ関数 **$L(L, S, Y, \mu)$** を構成します。

$$
L(L, S, Y, \mu) = \|L\|_* + \lambda \|S\|_1 + \langle Y, M - L - S \rangle + \frac{\mu}{2} \|M - L - S\|_F^2
$$

ここで：

* **$Y$** はラグランジュ乗数行列
* **$\mu$** は正のペナルティパラメータ
* **$\langle A, B \rangle = \text{tr}(A^T B)$**（内積）
* **$\| \cdot \|_F$** はフロベニウスノルム

計算を簡略化するため、二乗項を整理すると次のように書き換えられます。

$$
L(L, S, Y, \mu) = \|L\|_* + \lambda \|S\|_1 + \frac{\mu}{2} \left\| M - L - S + \frac{1}{\mu} Y \right\|_F^2 - \frac{1}{2\mu} \|Y\|_F^2
$$

---

## 3. 各変数の更新式の導出

IALMでは、他の変数を固定して1つずつ更新する交互方向法（ADMM的なアプローチ）をとります。

### (1) **$S$** の更新（スパース行列）

**$L$** と **$Y$** を固定し、**$S$** について最小化します。

$$
S_{k+1} = \arg \min_S \lambda \|S\|_1 + \frac{\mu_k}{2} \left\| M - L_k - S + \frac{1}{\mu_k} Y_k \right\|_F^2
$$

この解は、**Soft-Thresholding（軟判定しきい値処理）演算子** **$\mathcal{S}_{\tau}(x)$** を用いて以下のように導かれます。

$$
S_{k+1} = \mathcal{S}_{\frac{\lambda}{\mu_k}} \left( M - L_k + \frac{1}{\mu_k} Y_k \right)
$$

ここで、**$\mathcal{S}_{\tau}(x) = \text{sgn}(x) \max(|x| - \tau, 0)$** です。

### (2) **$L$** の更新（低ランク行列）

**$S$** と **$Y$** を固定し、**$L$** について最小化します。

$$
L_{k+1} = \arg \min_L \|L\|_* + \frac{\mu_k}{2} \left\| M - S_{k+1} - L + \frac{1}{\mu_k} Y_k \right\|_F^2
$$

この解は、**SVT (Singular Value Thresholding: 特異値しきい値処理)** を用いて導かれます。

$$
L_{k+1} = \text{SVT}_{\frac{1}{\mu_k}} \left( M - S_{k+1} + \frac{1}{\mu_k} Y_k \right)
$$

具体的には、対象の行列を **$U \Sigma V^T$** と特異値分解（SVD）したとき、**$\text{SVT}_{\tau}(X) = U \mathcal{S}_{\tau}(\Sigma) V^T$** となります。

### (3) ラグランジュ乗数 **$Y$** とペナルティ項 **$\mu$** の更新

制約の誤差に基づいて、乗数とペナルティパラメータを更新します。

* **$Y_{k+1} = Y_k + \mu_k (M - L_{k+1} - S_{k+1})$**
* **$\mu_{k+1} = \rho \mu_k \quad (\rho > 1)$**

---

## 4. アルゴリズムのまとめ

最終的な更新ループは以下の通りです。

* **Step 1:** **$S_{k+1} = \mathcal{S}_{\frac{\lambda}{\mu_k}} (M - L_k + \mu_k^{-1} Y_k)$**
* **Step 2:** **$L_{k+1} = \text{SVT}_{\mu_k^{-1}} (M - S_{k+1} + \mu_k^{-1} Y_k)$**
* **Step 3:** **$Y_{k+1} = Y_k + \mu_k (M - L_{k+1} - S_{k+1})$**
* **Step 4:** **$\mu_{k+1} = \rho \mu_k$**

収束条件（例：**$\|M - L_{k+1} - S_{k+1}\|_F / \|M\|_F < \epsilon$**）を満たすまでこれらを繰り返します。これにより、ノイズや外れ値を含む **$M$** から、きれいに構造化された **$L$** と異常成分である **$S$** が分離されます。


拡張ラグランジュ関数から、各変数の更新式（近接写像）を導出するプロセスを詳しく解説します。

---

## 1. 拡張ラグランジュ関数の整理（平方完成）

まず、再構成誤差に関する項を一つにまとめます。
ターゲットとなる関数は以下でした。

$$L(L, S, Y, \mu) = \|L\|_* + \lambda \|S\|_1 + \langle Y, M - L - S \rangle + \frac{\mu}{2} \|M - L - S\|_F^2$$

ここで、内積項と二乗項を $\frac{\mu}{2} \| \cdot \|_F^2$ の形にまとめます。
一般に $\|A+B\|_F^2 = \|A\|_F^2 + 2\langle A, B \rangle + \|B\|_F^2$ であることを利用すると、次のように変形できます。

$$\frac{\mu}{2} \left\| M - L - S + \frac{1}{\mu} Y \right\|_F^2 = \frac{\mu}{2} \left( \|M - L - S\|_F^2 + \frac{2}{\mu} \langle Y, M - L - S \rangle + \frac{1}{\mu^2} \|Y\|_F^2 \right)$$

これを展開して元の式と比較すると、$\frac{1}{2\mu}\|Y\|_F^2$ の項が余分に出るため、帳尻を合わせると以下の形式（標準的なADMM形式）になります。

$$L(L, S, Y, \mu) = \|L\|_* + \lambda \|S\|_1 + \frac{\mu}{2} \left\| M - L - S + \frac{1}{\mu} Y \right\|_F^2 - \frac{1}{2\mu} \|Y\|_F^2$$

---

## 2. $S$ の更新式の導出（$L_1$ ノルム最小化）

$L$ と $Y$ を固定した場合、$S$ に関する最適化問題は次のように書けます。
定数項（$- \frac{1}{2\mu} \|Y\|_F^2$）を無視し、$X = M - L + \frac{1}{\mu} Y$ と置くと：

$$\min_S \left( \lambda \|S\|_1 + \frac{\mu}{2} \| S - X \|_F^2 \right)$$
$$\min_S \left( \frac{\lambda}{\mu} \|S\|_1 + \frac{1}{2} \| S - X \|_F^2 \right)$$

これは要素ごとに独立な最適化問題 $\min_{s_{ij}} \left( \frac{\lambda}{\mu} |s_{ij}| + \frac{1}{2} (s_{ij} - x_{ij})^2 \right)$ です。
この目的関数 $f(s)$ の劣微分を $0$ と置きます。

- $s > 0$ のとき：$\frac{\lambda}{\mu} + (s - x) = 0 \implies s = x - \frac{\lambda}{\mu}$
- $s < 0$ のとき：$-\frac{\lambda}{\mu} + (s - x) = 0 \implies s = x + \frac{\lambda}{\mu}$
- $s = 0$ のとき：$x \in [-\frac{\lambda}{\mu}, \frac{\lambda}{\mu}]$

これをまとめると、**軟判定しきい値処理 (Soft-Thresholding)** の式が得られます。

$$S_{k+1} = \text{sgn}(X) \cdot \max\left(|X| - \frac{\lambda}{\mu}, 0\right)$$

---

## 3. $L$ の更新式の導出（核ノルム最小化）

同様に、$S$ と $Y$ を固定し、$X' = M - S + \frac{1}{\mu} Y$ と置くと：

$$\min_L \left( \|L\|_* + \frac{\mu}{2} \| L - X' \|_F^2 \right)$$
$$\min_L \left( \frac{1}{\mu} \|L\|_* + \frac{1}{2} \| L - X' \|_F^2 \right)$$

核ノルム $\|L\|_*$ は行列の特異値の $L_1$ ノルムであるため、この問題は「行列の特異値に対する $L_1$ 正則化問題」に帰着します。

行列 $X'$ を特異値分解（SVD）して $X' = U \Sigma V^T$ とすると、最適解 $L$ は同じ特異ベクトル $U, V$ を持ち、特異値のみが収縮（Shrinkage）されることが証明されています（Cai et al. 2010）。

$$L_{k+1} = U \mathcal{S}_{\frac{1}{\mu}}(\Sigma) V^T$$

ここで $\mathcal{S}_{\tau}$ は先ほどのSoft-Thresholding演算子です。これが **SVT (Singular Value Thresholding)** です。

---

## 4. $Y$ の更新（デュアル変数の更新）

ラグランジュ乗数 $Y$ は、勾配法（上昇法）に基づいて更新されます。

$$Y_{k+1} = Y_k + \mu \cdot \nabla_Y L(L, S, Y, \mu)$$

拡張ラグランジュ関数の $Y$ に関する勾配は、単純に制約の残差（Equality constraint error）であるため、次式となります。

$$Y_{k+1} = Y_k + \mu (M - L_{k+1} - S_{k+1})$$

---

## まとめ：更新のポイント

導出の肝は、**複雑な行列分解問題を「近接写像（Proximal Mapping）」の形に落とし込むこと**にあります。

- $L_1$ ノルム（スパース性） $\to$ 要素ごとのSoft-Thresholding
- 核ノルム（低ランク性） $\to$ 特異値に対するSoft-Thresholding

この2つのステップを交互に繰り返すことで、非平滑な最適化問題を効率的に解くことができます。

理論的な背景として、近傍作用素の不動点反復に基づいているため、適切なステップサイズ（$\mu$）の制御により、非常に高速に収束することが知られています。

# ADMMの更新式導出

RPCA（Robust Principal Component Analysis）を **ADMM（Alternating Direction Method of Multipliers：交互方向乗数法）** を用いて解くための更新式の導出プロセスを整理します。

RPCAは、観測行列 $M$ を低ランク行列 $L$ とスパース行列 $S$ に分解する問題であり、以下の最適化問題として定式化されます。

$$\min_{L, S} \|L\|_* + \lambda \|S\|_1 \quad \text{subject to} \quad L + S = M$$

---

## 1. 拡張ラグランジュ関数の構成

制約 $L + S = M$ を含めた拡張ラグランジュ関数 $L(L, S, Y, \rho)$ を定義します。

$$L(L, S, Y, \rho) = \|L\|_* + \lambda \|S\|_1 + \langle Y, M - L - S \rangle + \frac{\rho}{2} \|M - L - S\|_F^2$$

ここで：
- $Y$：ラグランジュ乗数行列
- $\rho$：ペナルティパラメータ（$\rho > 0$）
- $\langle \cdot, \cdot \rangle$：行列の内積（$\text{tr}(A^T B)$）
- $\| \cdot \|_F$：フロベニウスノルム

計算を簡略化するため、第3項と第4項を平方完成の要領でまとめます。

$$L(L, S, Y, \rho) = \|L\|_* + \lambda \|S\|_1 + \frac{\rho}{2} \left\| M - L - S + \frac{1}{\rho} Y \right\|_F^2 - \frac{1}{2\rho} \|Y\|_F^2$$

---

## 2. ADMMによる変数更新の導出

ADMMでは、各変数を順番に最適化（交互最小化）します。

### (1) $L$ の更新式
$S$ と $Y$ を固定し、$L$ に関する最小化を行います。

$$L_{k+1} = \arg \min_L \|L\|_* + \frac{\rho}{2} \left\| L - (M - S_k + \frac{1}{\rho} Y_k) \right\|_F^2$$

この形式は **核ノルムの近接写像（Proximal Operator）** であり、解は **特異値しきい値処理（SVT: Singular Value Thresholding）** によって与えられます。

1. $X = M - S_k + \frac{1}{\rho} Y_k$ と置く。
2. $X$ を特異値分解（SVD）する：$X = U \Sigma V^T$
3. 特異値行列 $\Sigma$ の各対角成分 $\sigma_i$ に対してソフトしきい値処理を行う：$\mathcal{S}_{\frac{1}{\rho}}(\sigma_i) = \text{sgn}(\sigma_i) \max(|\sigma_i| - \frac{1}{\rho}, 0)$
4. **$L_{k+1} = U \mathcal{S}_{\frac{1}{\rho}}(\Sigma) V^T$**

ご質問の式変形は、**一次項（ラグランジュ乗数項）を二次項に「まとめる」ための平方完成**です。  
具体的に導出します。

---

## 1. 元の目的関数

RPCAのADMMにおける \(L\)-更新は

\[
L^{k+1} = \arg\min_L \; \|L\|_* + \langle \Lambda^k, M - L - S^k \rangle + \frac{\rho}{2} \|M - L - S^k\|_F^2
\]

です。ここで
- \(\Lambda^k\)：双対変数（ラグランジュ乗数）
- \(S^k\)：現在のスパース成分
- \(\rho > 0\)：ペナルティパラメータ



## 2. 一次項と二次項のまとめ

二次項 \(\frac{\rho}{2} \|M - L - S^k\|_F^2\) を展開します。

まず、誤差ベクトル（行列）を

\[
R = M - L - S^k
\]

とおくと、

\[
\frac{\rho}{2} \|R\|_F^2 = \frac{\rho}{2} \langle R, R \rangle
\]

です。一方、一次項は

\[
\langle \Lambda^k, R \rangle
\]

です。これらをまとめると、

\[
\langle \Lambda^k, R \rangle + \frac{\rho}{2} \|R\|_F^2
= \frac{\rho}{2} \left( \|R\|_F^2 + \frac{2}{\rho} \langle \Lambda^k, R \rangle \right)
\]

ここで、**平方完成**を行います：

\[
\|R\|_F^2 + \frac{2}{\rho} \langle \Lambda^k, R \rangle
= \left\| R + \frac{\Lambda^k}{\rho} \right\|_F^2 - \left\| \frac{\Lambda^k}{\rho} \right\|_F^2
\]

したがって、

\[
\langle \Lambda^k, R \rangle + \frac{\rho}{2} \|R\|_F^2
= \frac{\rho}{2} \left\| R + \frac{\Lambda^k}{\rho} \right\|_F^2 - \frac{1}{2\rho} \|\Lambda^k\|_F^2
\]

となります。右辺第2項 \(-\frac{1}{2\rho} \|\Lambda^k\|_F^2\) は \(L\) に依存しない定数なので、最小化問題では無視できます。


## 3. \(R\) を元の変数に戻す

\(R = M - L - S^k\) だったので、

\[
R + \frac{\Lambda^k}{\rho} = M - L - S^k + \frac{\Lambda^k}{\rho}
\]

です。これを \(L\) について整理すると、

\[
R + \frac{\Lambda^k}{\rho} = - \left( L - \left( M - S^k + \frac{\Lambda^k}{\rho} \right) \right)
\]

ノルムは符号に依存しないので、

\[
\left\| R + \frac{\Lambda^k}{\rho} \right\|_F^2
= \left\| L - \left( M - S^k + \frac{\Lambda^k}{\rho} \right) \right\|_F^2
\]

となります。



## 4. 最終的な形

以上より、\(L\) に依存する部分だけを抜き出すと、

\[
L^{k+1} = \arg\min_L \; \|L\|_* + \frac{\rho}{2} \left\| L - \left( M - S^k + \frac{\Lambda^k}{\rho} \right) \right\|_F^2
\]

です。ここで

\[
W_L^k = M - S^k + \frac{\Lambda^k}{\rho}
\]

とおけば、

\[
L^{k+1} = \arg\min_L \; \|L\|_* + \frac{\rho}{2} \|L - W_L^k\|_F^2
\]

となります。これがご質問の式です。


## 5. まとめ

- 一次項 \(\langle \Lambda^k, M - L - S^k \rangle\) と二次項 \(\frac{\rho}{2} \|M - L - S^k\|_F^2\) を**平方完成**することで、
- ラグランジュ乗数項を二次項に「吸収」し、
- 結果として
  \[
  L^{k+1} = \arg\min_L \; \|L\|_* + \frac{\rho}{2} \|L - W_L^k\|_F^2,\quad W_L^k = M - S^k + \frac{\Lambda^k}{\rho}
  \]
  という**核ノルム正則化付き二次最小化問題**に帰着されます。

この形にすることで、**特異値閾値作用素（SVT）**をそのまま適用できるようになります。

---


### (2) $S$ の更新式
$L$ と $Y$ を固定し、$S$ に関する最小化を行います。

$$S_{k+1} = \arg \min_S \lambda \|S\|_1 + \frac{\rho}{2} \left\| S - (M - L_{k+1} + \frac{1}{\rho} Y_k) \right\|_F^2$$

これは **$L_1$ ノルムの近接写像** であり、解は **ソフトしきい値処理（Soft-Thresholding）** によって要素ごとに計算されます。

1. $X' = M - L_{k+1} + \frac{1}{\rho} Y_k$ と置く。
2. 各要素 $x'_{ij}$ に対して以下を適用する：
**$S_{k+1} = \text{sgn}(X') \odot \max\left(|X'| - \frac{\lambda}{\rho}, 0\right)$**
（ここで $\odot$ は要素ごとの積）

### (3) $Y$ （乗数）の更新式
双対変数を更新します。これは勾配上昇法に相当します。

**$Y_{k+1} = Y_k + \rho (M - L_{k+1} - S_{k+1})$**

---

## 3. 更新式のまとめ

ADMMを用いたRPCAのアルゴリズムは、収束するまで以下のステップを繰り返します。

- **Step 1 ($L$ の更新):**
  $L_{k+1} = \text{SVT}_{\frac{1}{\rho}} \left( M - S_k + \frac{1}{\rho} Y_k \right)$
- **Step 2 ($S$ の更新):**
  $S_{k+1} = \mathcal{S}_{\frac{\lambda}{\rho}} \left( M - L_{k+1} + \frac{1}{\rho} Y_k \right)$
- **Step 3 ($Y$ の更新):**
  $Y_{k+1} = Y_k + \rho (M - L_{k+1} - S_{k+1})$

> [!TIP]
> 実際の計算では、収束を早めるために $\rho$ を固定せず、各ステップで $\rho_{k+1} = \min(\rho_{max}, \mu \rho_k)$ （$\mu > 1$）のように大きくしていく手法が一般的です。これは **Inexact ALM** とも呼ばれますが、本質的な更新式の導出ロジックはADMMと同じです。

この手法により、大規模な行列データから背景（低ランク成分）と移動物体やノイズ（スパースな異常成分）を効率的に分離することが可能になります。

# 制約違反の意味

拡張ラグランジュ関数

\[
L_\rho(L, S, \Lambda) = \|L\|_* + \lambda \|S\|_1 + \langle \Lambda, M - L - S \rangle + \frac{\rho}{2} \|M - L - S\|_F^2
\]

における

- 一次項：\(\langle \Lambda, M - L - S \rangle\)
- 二次項：\(\frac{\rho}{2} \|M - L - S\|_F^2\)

は、**役割が異なる**ため、両方が必要です。それぞれの役割と違いを説明します。

---

## 1. 一次項 \(\langle \Lambda, M - L - S \rangle\) の役割

これは**ラグランジュ乗数項（双対変数項）**です。

### 1.1 元々のラグランジュ関数

制約付き問題

\[
\min_{L,S} f(L,S) \quad \text{s.t.} \quad M = L + S
\]

のラグランジュ関数は

\[
\mathcal{L}(L, S, \Lambda) = f(L,S) + \langle \Lambda, M - L - S \rangle
\]

です。ここで \(\Lambda\) は**ラグランジュ乗数（双対変数）**です。

### 1.2 一次項の役割

- **最適性条件の調整**：  
  制約付き最適化のKKT条件において、\(\nabla f\) と制約の勾配をバランスさせる役割を持ちます。
- **双対問題との関係**：  
  双対問題を構成する際に中心的な役割を果たします。
- **制約違反に対する「線形のペナルティ」**：  
  制約違反 \(M - L - S\) が大きいほど、この内積の絶対値が大きくなり、目的関数に影響を与えます。

しかし、一次項だけでは**制約を「強く」強制する力が弱い**ことがあります。

---

## 2. 二次項 \(\frac{\rho}{2} \|M - L - S\|_F^2\) の役割

これは**拡張ラグランジュ法における二次ペナルティ項**です。

### 2.1 二次項の役割

- **制約違反に対する「二次のペナルティ」**：  
  制約違反 \(M - L - S\) が大きいほど、**二乗に比例して**コストが増えます。
- **制約への「強制力」**：  
  一次項より強い「凹み」を作り、解が制約に近づくように強く誘導します。
- **収束性の改善**：  
  多くの場合、二次項を加えることでアルゴリズムの収束が安定・高速になります。

### 2.2 なぜ二次か？

- 一次項だけだと、制約違反に対して**線形のコスト**しかかかりません。  
  これは「遠くに行っても傾きが一定」というイメージで、**制約への引き戻し力が弱い**ことがあります。
- 二次項を加えると、制約から離れるほど**勾配が大きくなる**ため、**制約方向への「強い引力」**が働きます。

---

## 3. なぜ両方が必要か？— 拡張ラグランジュ法の考え方

### 3.1 通常のペナルティ法の問題

もし二次項だけを使う「ペナルティ法」を考えると：

\[
\min_{L,S} f(L,S) + \frac{\rho}{2} \|M - L - S\|_F^2
\]

- \(\rho\) を大きくすれば制約はほぼ満たされますが、**数値的に不安定**になりやすい。
- \(\rho\) が有限のとき、**厳密な制約満足は得られない**（近似解になる）。

### 3.2 ラグランジュ法だけの問題

一方、一次項だけのラグランジュ法：

\[
\min_{L,S} f(L,S) + \langle \Lambda, M - L - S \rangle
\]

- 双対変数 \(\Lambda\) を適切に更新すれば理論的には収束しますが、**実用上は収束が遅い・不安定**なことが多い。
- 制約違反に対する「凹み」が弱く、**制約への引き戻し力が不十分**な場合があります。

### 3.3 拡張ラグランジュ法のアイデア

そこで、**両方を組み合わせた拡張ラグランジュ関数**を使います：

\[
L_\rho(L, S, \Lambda) = f(L,S) + \langle \Lambda, M - L - S \rangle + \frac{\rho}{2} \|M - L - S\|_F^2
\]

- 一次項：**双対変数を通じた最適性条件の調整**
- 二次項：**制約への強い誘導・収束性の改善**

という**役割分担**があります。

さらに、双対変数 \(\Lambda\) を反復ごとに更新することで、

- 初期段階では二次項が主に制約を近づける役割を担い、
- 収束に近づくにつれて一次項（双対変数）が**精密に制約を満たす方向に調整**する

という**二段階の制御**が可能になります。

---

## 4. まとめ

- **一次項 \(\langle \Lambda, M - L - S \rangle\)**  
  - ラグランジュ乗数項（双対変数項）  
  - 最適性条件の調整、双対問題との関係  
  - 制約違反に対する「線形のペナルティ」

- **二次項 \(\frac{\rho}{2} \|M - L - S\|_F^2\)**  
  - 拡張ラグランジュ法の二次ペナルティ項  
  - 制約違反に対する「二次のペナルティ」  
  - 制約への強い誘導、収束性の改善

- **なぜ両方が必要か**  
  - 一次項だけでは制約への「引き戻し力」が弱く、収束が遅い・不安定になりがち。  
  - 二次項だけでは \(\rho\) を無限大にしない限り厳密な制約満足が得られず、数値的に不安定。  
  - 両方を組み合わせることで、**双対変数による精密な調整**と**二次ペナルティによる強い制約誘導**を両立させ、安定かつ効率的な収束を実現します。

したがって、一次項と二次項は**役割が異なる別々の正則項**として存在しており、拡張ラグランジュ法の重要な構成要素です。

