ご質問の式

$$
{ z^{k+1} = \arg\min_z \; \lambda \|z\|_1 + \frac{\rho}{2} \|z - w\|_2^2,\quad w = x^{k+1} + \frac{u^k}{\rho} }
$$

が、成分ごとに独立な問題に分解され、ソフト閾値作用素

$$
{ z_i^{k+1} = \operatorname{sign}(w_i) \max\bigl(|w_i| - \frac{\lambda}{\rho},\; 0\bigr) }
$$

で解ける理由を、**1次元の凸最適化問題**として詳しく導出します。

---

## 1. 問題の分解

目的関数は

$$
{ J(z) = \lambda \|z\|_1 + \frac{\rho}{2} \|z - w\|_2^2 }
$$

です。L1ノルムとL2ノルムはともに**成分ごとに分離可能**なので、各成分 

$$
{ z_i }
$$

 について独立に最小化できます：

$$
{ z_i^{k+1} = \arg\min_{z_i} \; \lambda |z_i| + \frac{\rho}{2} (z_i - w_i)^2 }
$$

ここで 

$$
{ w_i }
$$

 は定数です。以降は添字 
$$
{ i }
$$

 を省略し、1次元の問題

$$
{ \min_{z \in \mathbb{R}} \; \lambda |z| + \frac{\rho}{2} (z - w)^2 }
$$

を考えます（

$$
{ \lambda > 0, \rho > 0 }
$$

）。

---

## 2. 目的関数の凸性と最適性条件

目的関数

$$
{ f(z) = \lambda |z| + \frac{\rho}{2} (z - w)^2 }
$$

は

- $$
  { |z| }$$：凸だが $${ z=0 }$$ で微分不可能
  $$
- $$
  { (z - w)^2 }$$：滑らかな凸二次関数
  $$

なので、**合計も凸関数**です。したがって、**劣微分のゼロ点**が最小点を与えます。

劣微分の定義より、

$$
{ \partial f(z) = \lambda \partial |z| + \rho (z - w) }
$$

ここで 

$$
{ |z| }
$$

 の劣微分は

$$
{ \partial |z| = \begin{cases} \{1\} & z > 0 \\ [-1, 1] & z = 0 \\ \{-1\} & z < 0 \end{cases} }
$$

です。

最適性条件は

$$
{ 0 \in \partial f(z) = \lambda \partial |z| + \rho (z - w) }
$$

すなわち

$$
{ \rho (w - z) \in \lambda \partial |z| }
$$

です。

---

## 3. 場合分けによる解の導出

### 場合1：
$$
{ z > 0 }
$$

 のとき

このとき 

$$
{ \partial |z| = \{1\} }
$$

 なので、最適性条件は

$$
{ \rho (w - z) = \lambda \cdot 1 \quad \Rightarrow \quad z = w - \frac{\lambda}{\rho} }
$$

です。ただし、この解が 

$$
{ z > 0 }
$$

 を満たすには

$$
{ w - \frac{\lambda}{\rho} > 0 \quad \Rightarrow \quad w > \frac{\lambda}{\rho} }
$$

が必要です。

したがって、

 **のとき**は

$$
{ z = w - \frac{\lambda}{\rho} }
$$

が解です。

---

### 場合2：
$$
{ z < 0 }
$$

 のとき

このとき 

$$
{ \partial |z| = \{-1\} }
$$

 なので、最適性条件は

$$
{ \rho (w - z) = \lambda \cdot (-1) \quad \Rightarrow \quad z = w + \frac{\lambda}{\rho} }
$$

です。この解が 

$$
{ z < 0 }
$$

 を満たすには

$$
{ w + \frac{\lambda}{\rho} < 0 \quad \Rightarrow \quad w < -\frac{\lambda}{\rho} }
$$

が必要です。

したがって、

 **のとき**は

$$
{ z = w + \frac{\lambda}{\rho} }
$$

が解です。

---

### 場合3：
$$
{ z = 0 }
$$

 のとき

このとき 

$$
{ \partial |z| = [-1, 1] }
$$

 なので、最適性条件は

$$
{ \rho (w - 0) \in \lambda [-1, 1] \quad \Rightarrow \quad \rho w \in [-\lambda, \lambda] }
$$

すなわち

$$
{ -\lambda \le \rho w \le \lambda \quad \Rightarrow \quad |w| \le \frac{\lambda}{\rho} }
$$

です。

したがって、

 **のとき**は 
$$
{ z = 0 }
$$

 が解です。

---

## 4. ソフト閾値作用素としてのまとめ

以上をまとめると、最適解は

$$
{ z^* = \begin{cases} w - \frac{\lambda}{\rho} & w > \frac{\lambda}{\rho} \\ 0 & |w| \le \frac{\lambda}{\rho} \\ w + \frac{\lambda}{\rho} & w < -\frac{\lambda}{\rho} \end{cases} }
$$

です。これはまさに**ソフト閾値作用素**の定義です。

コンパクトに書くと、

$$
{ z^* = \operatorname{sign}(w) \max\bigl(|w| - \frac{\lambda}{\rho},\; 0\bigr) }
$$

となります。

---

## 5. 幾何学的な解釈

- 目的関数 
  $$
  { f(z) = \lambda |z| + \frac{\rho}{2} (z - w)^2 }
  $$

   は、- 二次関数 

    $$
    { \frac{\rho}{2} (z - w)^2 }
    $$

    （頂点 
    $$
    { w }
    $$

    ）に
  - L1ペナルティ 

    $$
    { \lambda |z| }
    $$

     が加わった形です。
- L1ペナルティは原点付近で「平らに」する効果があり、その結果、
  - $$
    { |w| }$$ が小さいときは最適解が原点に「縮退」し（$${ z^* = 0 }$$）、
    $$
  - $$
    { |w| }$$ が大きいときは原点から $${ \frac{\lambda}{\rho} }$$ だけ内側に引き戻された値（$${ z^* = w - \frac{\lambda}{\rho} \operatorname{sign}(w) }$$）になります。
    $$

これが「ソフト閾値」と呼ばれる所以です。

---

## 6. 元の問題への戻り

元のベクトル問題に戻ると、各成分 

$$
{ i }
$$

 について

$$
{ z_i^{k+1} = \arg\min_{z_i} \; \lambda |z_i| + \frac{\rho}{2} (z_i - w_i)^2,\quad w_i = x_i^{k+1} + \frac{u_i^k}{\rho} }
$$

であり、上記の1次元の結果をそのまま適用して

$$
{ z_i^{k+1} = \operatorname{sign}(w_i) \max\bigl(|w_i| - \frac{\lambda}{\rho},\; 0\bigr) }
$$

が得られます。

---

### まとめ

- L1ノルム付き二次関数の最小化は、**成分ごとに独立な1次元凸最適化問題**に分解できます。
- 各成分について、**劣微分を用いた最適性条件**を解くと、場合分けによりソフト閾値作用素が導出されます。
- 幾何学的には、L1ペナルティが原点付近を「平らに」し、大きい値は原点方向に引き戻す効果を持ちます。
- これがADMMの 
  $$
  { z }
  $$

  -更新ステップで使われる「ソフト閾値作用素」の正体です。

## RPCA

RPCA（Robust Principal Component Analysis）の標準的な定式化と、ADMM（Alternating Direction Method of Multipliers）による更新式の導出を説明します。

---

## 1. RPCA の問題設定

観測行列 

$$
{ M \in \mathbb{R}^{m \times n} }
$$

 を

$$
{ M = L + S }
$$

と分解し、以下の最適化問題を解きます：

$$
{ \min_{L, S} \; \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.} \quad M = L + S }
$$

ここで

- $$
  { \|L\|_* }$$：核ノルム（特異値の和）
  $$
- $$
  { \|S\|_1 }$$：要素ごとの L1 ノルム（絶対値和）
  $$
- $$
  { \lambda > 0 }$$：スパース成分の正則化パラメータ
  $$

核ノルムは低ランク行列への誘導、L1ノルムはスパースな外れ値への誘導を行います。

---

## 2. ADMM の適用（補助変数の導入）

ADMM を適用するために、制約 

$$
{ M = L + S }
$$

 に対する拡張ラグランジュ関数を構成します：

$$
{ L_\rho(L, S, \Lambda) = \|L\|_* + \lambda \|S\|_1 + \langle \Lambda, M - L - S \rangle + \frac{\rho}{2} \|M - L - S\|_F^2 }
$$

ここで

- $$
  { \Lambda \in \mathbb{R}^{m \times n} }$$：ラグランジュ乗数（双対変数）
  $$
- $$
  { \rho > 0 }$$：ペナルティパラメータ
  $$
- $$
  { \langle A, B \rangle = \operatorname{tr}(A^\top B) }$$：フロベニウス内積
  $$
- $$
  { \|A\|_F }$$：フロベニウスノルム
  $$

ADMM は、この 

$$
{ L_\rho }
$$

 に対して以下の**交互最小化**を行います：

1. $$
   { L }$$-更新：$${ S, \Lambda }$$ を固定して $${ L_\rho }$$ を $${ L }$$ について最小化
   $$
2. $$
   { S }$$-更新：$${ L, \Lambda }$$ を固定して $${ L_\rho }$$ を $${ S }$$ について最小化
   $$
3. $$
   { \Lambda }$$-更新：双対変数 $${ \Lambda }$$ を更新
   $$

---

## 3. 
$$
{ L }
$$

-更新式の導出（核ノルム最小化）

$$
{ S = S^k }$$, $${ \Lambda = \Lambda^k }$$ を固定し、$${ L }$$ について最小化します：

$${ L^{k+1} = \arg\min_L \; \|L\|_* + \langle \Lambda^k, M - L - S^k \rangle + \frac{\rho}{2} \|M - L - S^k\|_F^2 }
$$

$$
{ \Lambda^k }$$ に関する項をまとめると、

$${ L^{k+1} = \arg\min_L \; \|L\|_* + \frac{\rho}{2} \|L - (M - S^k + \frac{\Lambda^k}{\rho})\|_F^2 }
$$

ここで

$$
{ W_L^k = M - S^k + \frac{\Lambda^k}{\rho} }
$$

とおくと、

$$
{ L^{k+1} = \arg\min_L \; \|L\|_* + \frac{\rho}{2} \|L - W_L^k\|_F^2 }
$$

これは**核ノルム正則化付き二次関数**の最小化であり、**特異値閾値作用素（Singular Value Thresholding, SVT）**で解けます。

具体的には、

$$
{ W_L^k }
$$

 の特異値分解を

$$
{ W_L^k = U \Sigma V^\top,\quad \Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_r) }
$$

とすると、解は

$$
{ L^{k+1} = U \operatorname{soft}_{1/\rho}(\Sigma) V^\top }
$$

です。ここで 

$$
{ \operatorname{soft}_{\tau}(\sigma) }
$$

 はソフト閾値作用素：

$$
{ \operatorname{soft}_{\tau}(\sigma) = \operatorname{sign}(\sigma) \max(|\sigma| - \tau, 0) }
$$

したがって、

$$
{ L^{k+1} = U \operatorname{diag}\bigl( \operatorname{soft}_{1/\rho}(\sigma_1), \dots, \operatorname{soft}_{1/\rho}(\sigma_r) \bigr) V^\top }
$$

が 

$$
{ L }
$$

-更新式です。

---

## 4. 
$$
{ S }
$$

-更新式の導出（L1ノルム最小化）

$$
{ L = L^{k+1} }$$, $${ \Lambda = \Lambda^k }$$ を固定し、$${ S }$$ について最小化します：

$${ S^{k+1} = \arg\min_S \; \lambda \|S\|_1 + \langle \Lambda^k, M - L^{k+1} - S \rangle + \frac{\rho}{2} \|M - L^{k+1} - S\|_F^2 }
$$

$$
{ \Lambda^k }$$ に関する項をまとめると、

$${ S^{k+1} = \arg\min_S \; \lambda \|S\|_1 + \frac{\rho}{2} \|S - (M - L^{k+1} + \frac{\Lambda^k}{\rho})\|_F^2 }
$$

ここで

$$
{ W_S^k = M - L^{k+1} + \frac{\Lambda^k}{\rho} }
$$

とおくと、

$$
{ S^{k+1} = \arg\min_S \; \lambda \|S\|_1 + \frac{\rho}{2} \|S - W_S^k\|_F^2 }
$$

これは**L1ノルム付き二次関数**の最小化であり、**要素ごとのソフト閾値作用素**で解けます：

$$
{ S^{k+1} = \operatorname{soft}_{\lambda/\rho}(W_S^k) }
$$

成分ごとに

$$
{ S_{ij}^{k+1} = \operatorname{sign}(W_{S,ij}^k) \max\bigl(|W_{S,ij}^k| - \frac{\lambda}{\rho},\; 0\bigr) }
$$

---

## 5. 
$$
{ \Lambda }
$$

-更新式の導出（双対変数の更新）

双対変数 

$$
{ \Lambda }
$$

 は、**制約の違反量**に比例して更新されます：

$$
{ \Lambda^{k+1} = \Lambda^k + \rho (M - L^{k+1} - S^{k+1}) }
$$

これは拡張ラグランジュ法の標準的な更新則です。

---

## 6. まとめ：RPCA に対する ADMM 更新式

以上をまとめると、RPCA 問題

$$
{ \min_{L, S} \; \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.} \quad M = L + S }
$$

に対する ADMM の更新式は

$$
{ \begin{aligned} W_L^k &= M - S^k + \frac{\Lambda^k}{\rho} \\ L^{k+1} &= \operatorname{SVT}_{1/\rho}(W_L^k) \quad \text{（特異値閾値作用素）} \\ W_S^k &= M - L^{k+1} + \frac{\Lambda^k}{\rho} \\ S^{k+1} &= \operatorname{soft}_{\lambda/\rho}(W_S^k) \quad \text{（要素ごとのソフト閾値）} \\ \Lambda^{k+1} &= \Lambda^k + \rho (M - L^{k+1} - S^{k+1}) \end{aligned} }
$$

となります。

---

### 補足

- **特異値閾値作用素（SVT）**は、行列の特異値分解を行い、各特異値に対してソフト閾値を適用する操作です。
- **ソフト閾値作用素**は、L1ノルム付き二次関数の最小化を閉形式で与えます。
- ADMM の各ステップで「完全に最適化」する（勾配ゼロの解を選ぶ）ことが、理論的な収束保証の前提になっています。
- 実装上は、特異値分解の計算コストが高いため、大規模問題では近似手法（ランダム化SVDなど）を使うことがあります。

このように、RPCA は核ノルムとL1ノルムを組み合わせた凸最適化問題として定式化され、ADMM により効率的に解くことができます。
