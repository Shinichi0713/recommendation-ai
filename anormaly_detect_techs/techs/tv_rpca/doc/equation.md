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
   これは「核ノルム＋二次項」の最小化問題で、**特異値しきい値処理（SVT）**を含む形になります。  
   実装上は、勾配降下＋近接オペレータ、あるいは線形方程式を解く形に変形してADMM的に解きます。

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

とおいてADMM的に解く、という意図だと解釈します。  
ただし、$D x - z$ が2回出てきているのは重複しているので、ここでは

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
