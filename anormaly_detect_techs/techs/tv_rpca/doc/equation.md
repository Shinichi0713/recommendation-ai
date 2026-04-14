## Stanford

指定されたページは「Total Variation Denoising via ADMM」の解説ですが、ここで示されているADMMの流れをベースに、**RPCA＋TV正則化項を解く場合のアルゴリズムの流れ**を説明します。

---

## 1. 元のTV Denoising via ADMMの流れ（Stanfordページの内容）

### 目的関数

Total Variation Denoisingでは、観測画像 $b$ から真の画像 $x$ を推定する問題を

$$
\min_x \ \frac{1}{2} \|x - b\|_2^2 + \lambda \|Dx\|_1
$$

と書きます。ここで：

- $D$：差分行列（勾配オペレータ）
- $\|Dx\|_1$：TV正則化項（勾配のL1ノルム）

### 変数分割（ADMMの準備）

ADMMを使うために、補助変数 $z$ を導入して

$$
\min_{x, z} \ \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 \quad \text{s.t.} \quad z = Dx
$$

と書き直します。

### 拡張ラグランジュ関数

ペナルティパラメータ $\rho > 0$ を用いて、拡張ラグランジュ関数は

$$
L_\rho(x, z, u) = \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 + u^\top (Dx - z) + \frac{\rho}{2} \|Dx - z\|_2^2
$$

となります（$u$ は双対変数）。

### ADMMの更新ステップ

ページでは、以下の3ステップを繰り返します[Stanford ADMM TV Denoising](https://web.stanford.edu/~boyd/papers/admm/total_variation/total_variation.html)：

1. **x-update（画像更新）**

   $$
   x^{k+1} = (I + \rho D^\top D)^{-1} \big( b + \rho D^\top (z^k - u^k) \big)
   $$

   これは線形方程式を解く問題です。
2. **z-update（勾配変数更新）**

   $$
   z^{k+1} = \mathrm{shrink}(D x^{k+1} + u^k, \lambda / \rho)
   $$

   ここで $\mathrm{shrink}(a, \kappa) = \max(0, a - \kappa) - \max(0, -a - \kappa)$ はソフトしきい値関数です。
3. **u-update（双対変数更新）**

   $$
   u^{k+1} = u^k + D x^{k+1} - z^{k+1}
   $$

### 収束判定

- 主残差：$r = Dx - z$
- 双対残差：$s = -\rho D^\top (z - z_{\text{old}})$

これらが所定の許容誤差（ABSTOL, RELTOL）以下になれば収束と判定します[Stanford ADMM TV Denoising](https://web.stanford.edu/~boyd/papers/admm/total_variation/total_variation.html)。

---

## 2. RPCA＋TV正則化項を解く場合のアルゴリズムの流れ

RPCA＋TVでは、観測行列（画像）$X$ を

$$
X = L + S
$$

と分解し、目的関数を

$$
\min_{L, S} \ \|L\|_* + \lambda \|S\|_1 + \mu \mathrm{TV}(L)
$$

とします（TVは例えば $\mathrm{TV}(L) = \|D L\|_1$）。

ここで、TV DenoisingのADMMを拡張する形で、以下のようにアルゴリズムを組み立てます。

### 目的関数と変数分割

RPCA＋TVの目的関数を

$$
\min_{L, S, Z} \ \|L\|_* + \lambda \|S\|_1 + \mu \|Z\|_1 \quad \text{s.t.} \quad X = L + S,\quad Z = D L
$$

と書き、拡張ラグランジュ関数を

$$
\begin{aligned}
L_\rho(L, S, Z, U_1, U_2) 
&= \|L\|_* + \lambda \|S\|_1 + \mu \|Z\|_1 \\
&\quad + U_1^\top (X - L - S) + \frac{\rho_1}{2} \|X - L - S\|_F^2 \\
&\quad + U_2^\top (D L - Z) + \frac{\rho_2}{2} \|D L - Z\|_F^2
\end{aligned}
$$

とします（$U_1, U_2$ は双対変数）。

### ADMMの更新ステップ（拡張版）

1. **L-update（低ランク成分の更新）**

   $$
   L^{k+1} = \arg\min_L \ \|L\|_* + \frac{\rho_1}{2} \|X - L - S^k + U_1^k\|_F^2 + \frac{\rho_2}{2} \|D L - Z^k + U_2^k\|_F^2
   $$

   これは「核ノルム＋二次項」の最小化問題で、**特異値しきい値処理（SVT）**を含む形になります。実装上は、勾配降下＋近接オペレータ、あるいは線形方程式を解く形に変形してADMM的に解きます。
2. **S-update（スパース成分の更新）**

   $$
   S^{k+1} = \arg\min_S \ \lambda \|S\|_1 + \frac{\rho_1}{2} \|X - L^{k+1} - S + U_1^k\|_F^2
   $$

   これは要素ごとの**ソフトしきい値処理**で閉形式に解けます：

   $$
   S^{k+1} = \mathrm{shrink}(X - L^{k+1} + U_1^k, \lambda / \rho_1)
   $$
3. **Z-update（TV成分の更新）**

   $$
   Z^{k+1} = \arg\min_Z \ \mu \|Z\|_1 + \frac{\rho_2}{2} \|D L^{k+1} - Z + U_2^k\|_F^2
   $$

   これも要素ごとのソフトしきい値処理で閉形式に解けます：

   $$
   Z^{k+1} = \mathrm{shrink}(D L^{k+1} + U_2^k, \mu / \rho_2)
   $$

   このステップは、TV Denoisingのz-updateと本質的に同じです[Stanford ADMM TV Denoising](https://web.stanford.edu/~boyd/papers/admm/total_variation/total_variation.html)。
4. **双対変数の更新**

   $$
   \begin{aligned}
   U_1^{k+1} &= U_1^k + (X - L^{k+1} - S^{k+1}) \\
   U_2^{k+1} &= U_2^k + (D L^{k+1} - Z^{k+1})
   \end{aligned}
   $$

### 収束判定

- 主残差1：$r_1 = X - L - S$
- 主残差2：$r_2 = D L - Z$
- 双対残差：$s_1, s_2$（各制約に対応）

これらが所定の許容誤差以下になれば収束と判定します（TV Denoisingの収束判定を拡張した形）[Stanford ADMM TV Denoising](https://web.stanford.edu/~boyd/papers/admm/total_variation/total_variation.html)。

---

## 3. まとめ

- StanfordのTV Denoising via ADMMページでは、**「二次損失＋TV正則化」をADMMで解く標準的な流れ**が示されています[Stanford ADMM TV Denoising](https://web.stanford.edu/~boyd/papers/admm/total_variation/total_variation.html)。
- RPCA＋TV正則化項を解く場合は、これを拡張して：
  - 低ランク正則化（核ノルム）の更新（L-update）
  - スパース正則化（L1ノルム）の更新（S-update）
  - TV正則化（勾配のL1ノルム）の更新（Z-update）
    をADMMの枠組みで交互に解く形になります。
- TV Denoisingのz-update（ソフトしきい値）と双対更新は、そのままRPCA＋TVのZ-updateに流用できます。

このように、TV DenoisingのADMM実装をベースに、低ランク成分とスパース成分の更新ステップを追加することで、RPCA＋TV正則化項を解くアルゴリズムを構築できます。

## TV+RPCA

この式は、**ADMMの「x-update」ステップで、画像 $x$ を最小二乗的に更新している**ことを表しています。
具体的には、「データフィット項（$x$ が観測 $b$ に近いこと）」と「TV制約（$Dx$ が補助変数 $z$ に近いこと）」のバランスを取る形で $x$ を決めています。

---

## 1. 元の問題と拡張ラグランジュ関数

Total Variation Denoisingの問題は

$$
\min_{x, z} \ \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 \quad \text{s.t.} \quad z = Dx
$$

です。ADMMでは、これを拡張ラグランジュ関数

$$
L_\rho(x, z, u) = \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 + u^\top (Dx - z) + \frac{\rho}{2} \|Dx - z\|_2^2
$$

で表し、$x, z, u$ を交互に更新します。

---

## 2. x-update の導出

x-update は、$z, u$ を固定した状態で $x$ について $L_\rho$ を最小化するステップです：

$$
x^{k+1} = \arg\min_x L_\rho(x, z^k, u^k)
$$

ここで、$z$ と $u$ は定数とみなせるので、$x$ に依存する項だけを抜き出すと：

$$
\begin{aligned}
L_\rho(x, z^k, u^k)
&\propto \frac{1}{2} \|x - b\|_2^2 + u^{k\top} Dx + \frac{\rho}{2} \|Dx - z^k\|_2^2 \\
&= \frac{1}{2} \|x - b\|_2^2 + \frac{\rho}{2} \|Dx - z^k\|_2^2 + (u^k)^\top Dx
\end{aligned}
$$

（$\lambda \|z\|_1$ は $x$ に依存しないので無視）

これを $x$ で微分して $0$ とおきます。

---

## 3. 微分と正規方程式

ベクトル微分の公式を使うと：

- $\frac{\partial}{\partial x} \frac{1}{2} \|x - b\|_2^2 = x - b$
- $\frac{\partial}{\partial x} \frac{\rho}{2} \|Dx - z^k\|_2^2 = \rho D^\top (Dx - z^k)$
- $\frac{\partial}{\partial x} (u^k)^\top Dx = D^\top u^k$

したがって、

$$
\frac{\partial L_\rho}{\partial x} = (x - b) + \rho D^\top (Dx - z^k) + D^\top u^k = 0
$$

整理すると：

$$
x + \rho D^\top D x = b + \rho D^\top z^k - D^\top u^k
$$

左辺をまとめると：

$$
(I + \rho D^\top D) x = b + \rho D^\top z^k - D^\top u^k
$$

ここで、$u^k$ の符号を合わせるために、ADMMの標準的な形に合わせて $-D^\top u^k$ を $+ \rho D^\top (-u^k / \rho)$ と見なすと、

$$
(I + \rho D^\top D) x = b + \rho D^\top \left( z^k - \frac{u^k}{\rho} \right)
$$

となります。ページでは $u^k$ を「スケーリングされた双対変数」として扱っているため、$u^k$ をそのまま使って

$$
(I + \rho D^\top D) x = b + \rho D^\top (z^k - u^k)
$$

と書いています[Stanford ADMM TV Denoising](https://web.stanford.edu/~boyd/papers/admm/total_variation/total_variation.html)。

したがって、

$$
x^{k+1} = (I + \rho D^\top D)^{-1} \big( b + \rho D^\top (z^k - u^k) \big)
$$

という更新式になります。

---

## 4. この式が意味すること

この式は、以下の2つの情報をバランスよく取り入れて $x$ を決めていることを表しています：

1. **データフィット**：$x$ が観測 $b$ に近いこと（$\|x - b\|_2^2$）
2. **TV制約**：$Dx$ が補助変数 $z^k$ に近いこと（$\|Dx - z^k\|_2^2$）

- $I$ の項：データフィットの重み
- $\rho D^\top D$ の項：TV制約（勾配の一致）の重み

右辺の $b + \rho D^\top (z^k - u^k)$ は、

- $b$：観測データからの寄与
- $\rho D^\top (z^k - u^k)$：TV制約（$Dx \approx z$）からの寄与

を合わせた「目標値」のようなものです。

つまり、**「観測 $b$ に近く、かつ勾配 $Dx$ が $z^k$ に近くなるような $x$」を最小二乗的に求める**ステップが、この x-update です。

---

## 5. まとめ

- x-update の式

  $$
  x^{k+1} = (I + \rho D^\top D)^{-1} \big( b + \rho D^\top (z^k - u^k) \big)
  $$

  は、**拡張ラグランジュ関数を $x$ について最小化した結果**として得られます。
- これは「データフィット」と「TV制約（$Dx \approx z$）」のバランスを取る最小二乗問題の解であり、ADMMの1ステップとして $x$ を更新しています。

与えられたラグランジュ関数

$$
L_\rho(x, z, y) = \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 + y^\top (D x - z) + \frac{\rho}{2} \|D x - z\|_2^2
$$

に対して、変数分割

$$
x - b = z_1,\quad z = z_2,\quad D x - z = z_3,\quad D x - z = z_4
$$

とおいてADMM的に解く、という意図だと解釈します。ただし、$D x - z$ が2回出てきているのは重複しているので、ここでは

- $x - b = z_1$
- $z = z_2$
- $D x - z = z_3$

という3つの制約に分ける形で導出します（$z_4$ は不要）。

---

## 1. 問題の再定式化

元の問題は

$$
\min_{x, z} \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 \quad \text{s.t.} \quad D x - z = 0
$$

です。これを変数分割して

$$
\min_{x, z, z_1, z_2, z_3} \frac{1}{2} \|z_1\|_2^2 + \lambda \|z_2\|_1
$$

subject to

$$
\begin{aligned}
x - b &= z_1 \\
z &= z_2 \\
D x - z &= z_3
\end{aligned}
$$

と書きます。ここで $z_3 = 0$ です。

---

## 2. 拡張ラグランジュ関数

双対変数を $y_1, y_2, y_3$ とすると、拡張ラグランジュ関数は

$$
\begin{aligned}
L_\rho(x, z, z_1, z_2, z_3, y_1, y_2, y_3)
&= \frac{1}{2} \|z_1\|_2^2 + \lambda \|z_2\|_1 \\
&\quad + y_1^\top (x - b - z_1) + \frac{\rho_1}{2} \|x - b - z_1\|_2^2 \\
&\quad + y_2^\top (z - z_2) + \frac{\rho_2}{2} \|z - z_2\|_2^2 \\
&\quad + y_3^\top (D x - z - z_3) + \frac{\rho_3}{2} \|D x - z - z_3\|_2^2
\end{aligned}
$$

となります。

---

## 3. ADMMの更新式

ADMMでは、各変数について交互に最小化します。

### (1) x-update

$z, z_1, z_2, z_3$ を固定して $x$ について最小化：

$$
\begin{aligned}
x^{k+1} &= \arg\min_x \left\{ 
y_1^\top (x - b - z_1^k) + \frac{\rho_1}{2} \|x - b - z_1^k\|_2^2 
+ y_3^\top (D x - z^k - z_3^k) + \frac{\rho_3}{2} \|D x - z^k - z_3^k\|_2^2
\right\}
\end{aligned}
$$

勾配を0とおくと、

$$
\rho_1 (x - b - z_1^k + y_1^k/\rho_1) + \rho_3 D^\top (D x - z^k - z_3^k + y_3^k/\rho_3) = 0
$$

より

$$
(\rho_1 I + \rho_3 D^\top D) x = \rho_1 (b + z_1^k - y_1^k/\rho_1) + \rho_3 D^\top (z^k + z_3^k - y_3^k/\rho_3)
$$

という線形方程式を解けばよいです。
フーリエ空間で解く場合は、$D$ が循環行列と仮定して

$$
x^{k+1} = \mathcal{F}^{-1}\left( \frac{\mathcal{F}(\rho_1 (b + z_1^k - y_1^k/\rho_1) + \rho_3 D^\top (z^k + z_3^k - y_3^k/\rho_3))}{\rho_1 + \rho_3 |\mathcal{F}(D)|^2} \right)
$$

のように書けます。

---

### (2) z-update

$x, z_1, z_2, z_3$ を固定して $z$ について最小化：

$$
\begin{aligned}
z^{k+1} &= \arg\min_z \left\{ 
y_2^\top (z - z_2^k) + \frac{\rho_2}{2} \|z - z_2^k\|_2^2 
+ y_3^\top (D x^{k+1} - z - z_3^k) + \frac{\rho_3}{2} \|D x^{k+1} - z - z_3^k\|_2^2
\right\}
\end{aligned}
$$

勾配を0とおくと、

$$
\rho_2 (z - z_2^k + y_2^k/\rho_2) - \rho_3 (D x^{k+1} - z - z_3^k + y_3^k/\rho_3) = 0
$$

より

$$
(\rho_2 + \rho_3) z = \rho_2 (z_2^k - y_2^k/\rho_2) + \rho_3 (D x^{k+1} - z_3^k + y_3^k/\rho_3)
$$

したがって

$$
z^{k+1} = \frac{\rho_2 (z_2^k - y_2^k/\rho_2) + \rho_3 (D x^{k+1} - z_3^k + y_3^k/\rho_3)}{\rho_2 + \rho_3}
$$

---

### (3) z₁-update

$x, z, z_2, z_3$ を固定して $z_1$ について最小化：

$$
z_1^{k+1} = \arg\min_{z_1} \left\{ \frac{1}{2} \|z_1\|_2^2 + y_1^\top (x^{k+1} - b - z_1) + \frac{\rho_1}{2} \|x^{k+1} - b - z_1\|_2^2 \right\}
$$

勾配を0とおくと、

$$
z_1 + \rho_1 (z_1 - (x^{k+1} - b + y_1^k/\rho_1)) = 0
$$

より

$$
(1 + \rho_1) z_1 = \rho_1 (x^{k+1} - b + y_1^k/\rho_1)
$$

したがって

$$
z_1^{k+1} = \frac{\rho_1}{1 + \rho_1} (x^{k+1} - b + y_1^k/\rho_1)
$$

---

### (4) z₂-update

$x, z, z_1, z_3$ を固定して $z_2$ について最小化：

$$
z_2^{k+1} = \arg\min_{z_2} \left\{ \lambda \|z_2\|_1 + y_2^\top (z^{k+1} - z_2) + \frac{\rho_2}{2} \|z^{k+1} - z_2\|_2^2 \right\}
$$

これはL1正則化付き最小二乗なので、**ソフトしきい値処理**で閉形式に解けます：

$$
z_2^{k+1} = \mathrm{shrink}(z^{k+1} + y_2^k/\rho_2, \lambda / \rho_2)
$$

---

### (5) z₃-update

$x, z, z_1, z_2$ を固定して $z_3$ について最小化：

$$
z_3^{k+1} = \arg\min_{z_3} \left\{ y_3^\top (D x^{k+1} - z^{k+1} - z_3) + \frac{\rho_3}{2} \|D x^{k+1} - z^{k+1} - z_3\|_2^2 \right\}
$$

これは二次関数のみなので、勾配を0とおくと

$$
z_3^{k+1} = D x^{k+1} - z^{k+1} + y_3^k/\rho_3
$$

となります。ただし、元の問題では $z_3 = 0$ なので、実装上は $z_3^{k+1} = 0$ と固定しても構いません（その場合は双対更新で吸収されます）。

---

### (6) 双対変数の更新

各制約に対応する双対変数を更新します：

$$
\begin{aligned}
y_1^{k+1} &= y_1^k + \rho_1 (x^{k+1} - b - z_1^{k+1}) \\
y_2^{k+1} &= y_2^k + \rho_2 (z^{k+1} - z_2^{k+1}) \\
y_3^{k+1} &= y_3^k + \rho_3 (D x^{k+1} - z^{k+1} - z_3^{k+1})
\end{aligned}
$$

---

## 4. まとめ

- 与えられたラグランジュ関数に対して、変数分割 $x - b = z_1,\ z = z_2,\ D x - z = z_3$ を行い、ADMMの更新式を導出しました。
- 更新式は：
  - **x-update**：線形方程式（フーリエ空間で解く）
  - **z-update**：線形方程式（閉形式）
  - **z₁-update**：閉形式（二次関数の最小化）
  - **z₂-update**：ソフトしきい値処理（L1正則化）
  - **z₃-update**：閉形式（二次関数の最小化）
  - **双対更新**：各制約の残差に基づく更新

となります。
ただし、このように細かく変数分割すると更新ステップが増え、実装が複雑になるため、通常は $z_1, z_3$ を省略した形（元のTV DenoisingのADMM）で解くことが多いです。


## 補助更新式の導入

RPCA＋TV（核ノルム＋L1＋TV）に対して、**4つの補助変数 $Z_1, Z_2, Z_3, Z_4$** を導入したADMMの更新式を導出します。

---

## 1. 問題の再定式化（補助変数導入）

RPCA＋TVの目的関数は

$$
\min_{L, S, Z} \ \|L\|_* + \lambda \|S\|_1 + \mu \|Z\|_1 \quad \text{s.t.} \quad X = L + S,\ Z = D L
$$

です。ここで、以下のように**4つの補助変数**を導入します：

- $Z_1 = L$（低ランク成分）
- $Z_2 = S$（スパース成分）
- $Z_3 = Z$（TV成分）
- $Z_4 = D L$（勾配）

制約は

$$
\begin{aligned}
X &= Z_1 + Z_2 \\
Z_3 &= Z_4 \\
Z_1 &= L \\
Z_4 &= D L
\end{aligned}
$$

となります。ただし $Z_3 = Z_4$ は $Z = D L$ を意味します。

目的関数を補助変数で書き直すと：

$$
\min_{L, S, Z, Z_1, Z_2, Z_3, Z_4} \ \|Z_1\|_* + \lambda \|Z_2\|_1 + \mu \|Z_3\|_1
$$

subject to

$$
\begin{aligned}
X &= Z_1 + Z_2 \\
Z_3 &= Z_4 \\
Z_1 &= L \\
Z_4 &= D L
\end{aligned}
$$

---

## 2. 拡張ラグランジュ関数

双対変数を

- $U_1$：$X = Z_1 + Z_2$ に対応
- $U_2$：$Z_3 = Z_4$ に対応
- $V_1$：$Z_1 = L$ に対応
- $V_2$：$Z_4 = D L$ に対応

とすると、拡張ラグランジュ関数は

$$
\begin{aligned}
L_\rho &= \|Z_1\|_* + \lambda \|Z_2\|_1 + \mu \|Z_3\|_1 \\
&\quad + U_1^\top (X - Z_1 - Z_2) + \frac{\rho_1}{2} \|X - Z_1 - Z_2\|_F^2 \\
&\quad + U_2^\top (Z_3 - Z_4) + \frac{\rho_2}{2} \|Z_3 - Z_4\|_F^2 \\
&\quad + V_1^\top (Z_1 - L) + \frac{\rho_3}{2} \|Z_1 - L\|_F^2 \\
&\quad + V_2^\top (Z_4 - D L) + \frac{\rho_4}{2} \|Z_4 - D L\|_F^2
\end{aligned}
$$

となります。

---

## 3. ADMMの更新式

ADMMでは、各変数について交互に最小化します。

### (1) L-update

$Z_1, Z_2, Z_3, Z_4$ を固定して $L$ について最小化：

$$
\begin{aligned}
L^{k+1} &= \arg\min_L \left\{ 
- V_1^\top L + \frac{\rho_3}{2} \|Z_1^k - L\|_F^2 
- V_2^\top D L + \frac{\rho_4}{2} \|Z_4^k - D L\|_F^2
\right\}
\end{aligned}
$$

勾配を0とおくと、

$$
\rho_3 (L - Z_1^k + V_1^k/\rho_3) + \rho_4 D^\top (D L - Z_4^k + V_2^k/\rho_4) = 0
$$

より

$$
(\rho_3 I + \rho_4 D^\top D) L = \rho_3 (Z_1^k - V_1^k/\rho_3) + \rho_4 D^\top (Z_4^k - V_2^k/\rho_4)
$$

という線形方程式を解けばよいです。  
フーリエ空間で解く場合は、

$$
L^{k+1} = \mathcal{F}^{-1}\left( \frac{\mathcal{F}(\rho_3 (Z_1^k - V_1^k/\rho_3) + \rho_4 D^\top (Z_4^k - V_2^k/\rho_4))}{\rho_3 + \rho_4 |\mathcal{F}(D)|^2} \right)
$$

---

### (2) Z₁-update（低ランク成分）

$L, Z_2, Z_3, Z_4$ を固定して $Z_1$ について最小化：

$$
\begin{aligned}
Z_1^{k+1} &= \arg\min_{Z_1} \left\{ 
\|Z_1\|_* 
- U_1^\top Z_1 + \frac{\rho_1}{2} \|X - Z_1 - Z_2^k\|_F^2 
+ V_1^\top Z_1 + \frac{\rho_3}{2} \|Z_1 - L^{k+1}\|_F^2
\right\}
\end{aligned}
$$

これは「核ノルム＋二次項」の最小化なので、**特異値しきい値処理（SVT）**で解けます：

$$
Z_1^{k+1} = \mathrm{prox}_{\|\cdot\|_*}\left( \frac{\rho_1 (X - Z_2^k + U_1^k/\rho_1) + \rho_3 (L^{k+1} - V_1^k/\rho_3)}{\rho_1 + \rho_3} \right)
$$

---

### (3) Z₂-update（スパース成分）

$L, Z_1, Z_3, Z_4$ を固定して $Z_2$ について最小化：

$$
\begin{aligned}
Z_2^{k+1} &= \arg\min_{Z_2} \left\{ 
\lambda \|Z_2\|_1 
- U_1^\top Z_2 + \frac{\rho_1}{2} \|X - Z_1^{k+1} - Z_2\|_F^2
\right\}
\end{aligned}
$$

これはL1正則化付き最小二乗なので、**ソフトしきい値処理**で閉形式に解けます：

$$
Z_2^{k+1} = \mathrm{shrink}(X - Z_1^{k+1} + U_1^k/\rho_1, \lambda / \rho_1)
$$

---

### (4) Z₃-update（TV成分）

$L, Z_1, Z_2, Z_4$ を固定して $Z_3$ について最小化：

$$
\begin{aligned}
Z_3^{k+1} &= \arg\min_{Z_3} \left\{ 
\mu \|Z_3\|_1 
+ U_2^\top Z_3 + \frac{\rho_2}{2} \|Z_3 - Z_4^k\|_F^2
\right\}
\end{aligned}
$$

これもL1正則化付き最小二乗なので、ソフトしきい値処理：

$$
Z_3^{k+1} = \mathrm{shrink}(Z_4^k - U_2^k/\rho_2, \mu / \rho_2)
$$

---

### (5) Z₄-update（勾配成分）

$L, Z_1, Z_2, Z_3$ を固定して $Z_4$ について最小化：

$$
\begin{aligned}
Z_4^{k+1} &= \arg\min_{Z_4} \left\{ 
- U_2^\top Z_4 + \frac{\rho_2}{2} \|Z_3^{k+1} - Z_4\|_F^2 
+ V_2^\top Z_4 + \frac{\rho_4}{2} \|Z_4 - D L^{k+1}\|_F^2
\right\}
\end{aligned}
$$

これは二次関数のみなので、勾配を0とおくと：

$$
(\rho_2 + \rho_4) Z_4 = \rho_2 (Z_3^{k+1} + U_2^k/\rho_2) + \rho_4 (D L^{k+1} - V_2^k/\rho_4)
$$

より

$$
Z_4^{k+1} = \frac{\rho_2 (Z_3^{k+1} + U_2^k/\rho_2) + \rho_4 (D L^{k+1} - V_2^k/\rho_4)}{\rho_2 + \rho_4}
$$

---

### (6) 双対変数の更新

各制約に対応する双対変数を更新します：

$$
\begin{aligned}
U_1^{k+1} &= U_1^k + \rho_1 (X - Z_1^{k+1} - Z_2^{k+1}) \\
U_2^{k+1} &= U_2^k + \rho_2 (Z_3^{k+1} - Z_4^{k+1}) \\
V_1^{k+1} &= V_1^k + \rho_3 (Z_1^{k+1} - L^{k+1}) \\
V_2^{k+1} &= V_2^k + \rho_4 (Z_4^{k+1} - D L^{k+1})
\end{aligned}
$$

---

## 4. まとめ

- RPCA＋TVに対して、**4つの補助変数 $Z_1, Z_2, Z_3, Z_4$** を導入し、ADMMの更新式を導出しました。
- 更新式は：
  - **L-update**：線形方程式（フーリエ空間で解く）
  - **Z₁-update**：核ノルムの近接オペレータ（SVT）
  - **Z₂-update**：L1ソフトしきい値処理
  - **Z₃-update**：TVソフトしきい値処理
  - **Z₄-update**：二次関数の最小化（閉形式）
  - **双対更新**：各制約の残差に基づく更新

となります。  
ただし、補助変数を増やすと更新ステップが多くなり収束が遅くなる可能性があるため、実用上は**必要最小限の変数分割**（$L, S, Z$ のみ）で解くことが多いです。

## 論文の記事

結論から言うと、**「TV付きRPCAに対して4つの補助変数 $Z_1, Z_2, Z_3, Z_4$ を導入したADMM更新式」を明示的に導出している論文・記事は、公開情報の範囲では見当たりません**。  
ただし、**ADMMの一般理論**と**TV正則化付き最適化**の論文を組み合わせれば、同様の更新式を自力で導出することは可能です。

以下、関連する文献と導出の考え方を整理します。

---

## 1. ADMMの一般理論（変数分割・補助変数の扱い）

### (1) Distributed Optimization and Statistical Learning via ADMM（Boyd et al.）
- **PDF**: https://web.stanford.edu/~boyd/papers/admm_diststat.html  
- **内容**: ADMMの一般形（$f(x) + g(z)$ s.t. $Ax + Bz = c$）と、変数分割・補助変数の導入方法を詳しく解説。  
- **補助変数の扱い**:  
  - 目的関数を分離可能な形に書き換えるために、補助変数 $z$ を導入する方法が説明されています。  
  - 複数の制約がある場合の拡張（複数の双対変数・ペナルティ項）も理論的にカバーされています。

### (2) ADMM: Algorithms using Alternating Direction Method of Multipliers（Rパッケージ資料）
- **PDF**: https://cran.r-project.org/web/packages/ADMM/ADMM.pdf  
- **内容**: ADMMの実装例（Lasso, TV Denoisingなど）と、補助変数を導入した変数分割の具体例。  
- **補助変数の導入**:  
  - Lassoの例で $\beta = \alpha$ という補助変数を導入し、ADMM更新式を導出しています。  
  - 同様の考え方をRPCA＋TVに拡張すれば、$Z_1, Z_2, Z_3, Z_4$ のような補助変数を導入した更新式を導出できます。

---

## 2. TV正則化付き最適化（TV Denoising）のADMM

### (1) Total Variation Denoising via ADMM（Stanford）
- **URL**: https://web.stanford.edu/~boyd/papers/admm/total_variation/total_variation.html  
- **内容**: TV DenoisingのADMM実装（$u$-update, $z$-update, dual-update）を詳細に解説。  
- **補助変数の扱い**:  
  - $z = D u$ という補助変数を導入し、ADMMの更新式（$u$-update: 線形方程式、$z$-update: ソフトしきい値処理）を導出しています。  
  - RPCA＋TVの $Z_3, Z_4$ 部分は、このTV DenoisingのADMMをそのまま拡張した形になります。

---

## 3. RPCA＋TV（TV正則化付きRPCA）の代表的な論文

### (1) TVRPCA+: Low-rank and sparse decomposition based on spectral norm, structured sparse norm and total variation regularization
- **Journal**: Signal Processing, 2023  
- **URL**: https://www.sciencedirect.com/science/article/pii/S0165168423000630  
- **内容**:  
  - RPCA＋TVの目的関数とADMM更新式（L, S, Z, 双対）を導出。  
  - 補助変数（$Z_1, Z_2, Z_3$ に相当）を導入した変数分割の具体例が記載されています。  
  - ただし、4つの補助変数（$Z_1, Z_2, Z_3, Z_4$）を明示的に使う形ではありませんが、**複数の制約を扱うADMMの拡張**として理解できます。

### (2) An improved total variation regularized RPCA for moving object detection with dynamic background
- **Journal**: Journal of Industrial and Management Optimization, 2019  
- **URL**: https://www.aimsciences.org/article/doi/10.3934/jimo.2019106  
- **内容**:  
  - 動的背景を持つ動画の前景検出にTV-RPCAを適用。  
  - ADMMベースの更新式（L, S, Z, 双対）が導出されており、補助変数の導入方法も参考になります。

### (3) Low-Rank Decomposition and Total Variation Regularization of Hyperspectral Video Sequences
- **Report**: UCLA CAM Report 18-13  
- **URL**: https://www.math.ucla.edu/applied/cam/index.php?page=cam-reports  
- **内容**:  
  - ハイパースペクトル動画に対する低ランク＋TV分解。  
  - ADMM/PDSの更新式が導出されており、複数の制約を扱う変数分割の例が含まれます。

---

## 4. 導出の考え方（4つの補助変数 $Z_1, Z_2, Z_3, Z_4$）

RPCA＋TVの目的関数

$$
\min_{L, S, Z} \ \|L\|_* + \lambda \|S\|_1 + \mu \|Z\|_1 \quad \text{s.t.} \quad X = L + S,\ Z = D L
$$

に対して、補助変数を

- $Z_1 = L$（低ランク成分）
- $Z_2 = S$（スパース成分）
- $Z_3 = Z$（TV成分）
- $Z_4 = D L$（勾配）

と導入し、制約を

$$
\begin{aligned}
X &= Z_1 + Z_2 \\
Z_3 &= Z_4 \\
Z_1 &= L \\
Z_4 &= D L
\end{aligned}
$$

と書き換えます。  
拡張ラグランジュ関数を

$$
\begin{aligned}
L_\rho &= \|Z_1\|_* + \lambda \|Z_2\|_1 + \mu \|Z_3\|_1 \\
&\quad + U_1^\top (X - Z_1 - Z_2) + \frac{\rho_1}{2} \|X - Z_1 - Z_2\|_F^2 \\
&\quad + U_2^\top (Z_3 - Z_4) + \frac{\rho_2}{2} \|Z_3 - Z_4\|_F^2 \\
&\quad + V_1^\top (Z_1 - L) + \frac{\rho_3}{2} \|Z_1 - L\|_F^2 \\
&\quad + V_2^\top (Z_4 - D L) + \frac{\rho_4}{2} \|Z_4 - D L\|_F^2
\end{aligned}
$$

と定義し、ADMMの更新式（L-update, Z₁-update, Z₂-update, Z₃-update, Z₄-update, dual-update）を導出する、という流れになります。  
この導出は、上記のADMM一般論とTV DenoisingのADMMを組み合わせることで可能です。

---

## 5. まとめ

- **4つの補助変数 $Z_1, Z_2, Z_3, Z_4$ を明示的に使ったTV付きRPCAのADMM更新式**を導出している論文・記事は、公開情報の範囲では見当たりません。  
- しかし、**ADMMの一般理論（BoydのPDF）**と**TV DenoisingのADMM（Stanfordページ）**、さらに**TVRPCA+などのRPCA＋TV論文**を組み合わせれば、同様の更新式を自力で導出することは十分可能です。  
- 実用上は、補助変数を増やしすぎると収束が遅くなるため、**必要最小限の変数分割（$L, S, Z$ のみ）**で解くことが多いです。

したがって、**「4つの補助変数を使ったTV付きRPCAのADMM更新式」そのものを解説した文献はない**ものの、**ADMMとTV正則化の理論を組み合わせれば導出可能**というのが現状の結論です。


