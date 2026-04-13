ADMMの更新式に現れる $u$ は、**双対変数（dual variable）**、あるいは**スケーリングされたラグランジュ乗数**です。
制約付き最適化問題における「制約違反に対するペナルティ」を調整する役割を持っています。

---

## 1. 元の制約とラグランジュ乗数

Total Variation Denoisingの問題は

$$
\min_{x, z} \ \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 \quad \text{s.t.} \quad z = Dx
$$

です。ここで制約 $z = Dx$ を満たすために、ラグランジュ乗数（ベクトル）$y$ を導入すると、ラグランジュ関数は

$$
L(x, z, y) = \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 + y^\top (Dx - z)
$$

となります。

---

## 2. スケーリングされた双対変数 $u$

ADMMでは、ペナルティパラメータ $\rho$ を使った**拡張ラグランジュ関数**を扱います：

$$
L_\rho(x, z, y) = \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 + y^\top (Dx - z) + \frac{\rho}{2} \|Dx - z\|_2^2
$$

ここで、**スケーリングされた双対変数** $u = y / \rho$ を導入すると、式が少しすっきりします：

$$
L_\rho(x, z, u) = \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 + \frac{\rho}{2} \|Dx - z + u\|_2^2 - \frac{\rho}{2} \|u\|_2^2
$$

（最後の定数項は最小化には影響しないので無視できます）

このときの $u$ は：

- 元のラグランジュ乗数 $y$ を $\rho$ で割ったもの：$u = y / \rho$
- 制約 $z = Dx$ の「ずれ」を記憶し、次の反復で補正するための変数

という意味を持ちます。

---

## 3. $u$ の役割（直感的な説明）

ADMMの更新式では：

- x-update：$x$ を「データフィット＋TV制約」に基づいて更新
- z-update：$z$ を「TV正則化（L1ノルム）」に基づいて更新
- **u-update**：$u$ を「制約違反の累積」に基づいて更新

具体的には、u-update は

$$
u^{k+1} = u^k + Dx^{k+1} - z^{k+1}
$$

です。これは：

- $Dx^{k+1} - z^{k+1}$：現在の制約違反（勾配と補助変数の差）
- $u^k$：これまでの制約違反の累積

を足し合わせて、次の反復で「より強く制約を満たすように」補正する役割です。

---

## 4. まとめ

- $u$ は**双対変数（スケーリングされたラグランジュ乗数）**です。
- 制約 $z = Dx$ の違反を記憶し、ADMMの反復ごとに「制約を満たす方向」に $x$ と $z$ を補正する役割を持ちます。
- u-update の式 $u^{k+1} = u^k + Dx^{k+1} - z^{k+1}$ は、「現在の制約違反を累積する」ステップであり、これによってADMMが制約付き最適解に収束していきます。

つまり、$u$ は「制約をどれだけ破っているか」を表す補正項であり、ADMMが正しく収束するための重要な変数です。


# total ADMM

指定されたページは「Total Variation Denoising via ADMM」の解説ですが、ここで示されているADMMの流れをベースに、**RPCA＋TV正則化項を解く場合のアルゴリズムの流れ**を説明します。


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


## 3. まとめ

- StanfordのTV Denoising via ADMMページでは、**「二次損失＋TV正則化」をADMMで解く標準的な流れ**が示されています[Stanford ADMM TV Denoising](https://web.stanford.edu/~boyd/papers/admm/total_variation/total_variation.html)。
- RPCA＋TV正則化項を解く場合は、これを拡張して：
  - 低ランク正則化（核ノルム）の更新（L-update）
  - スパース正則化（L1ノルム）の更新（S-update）
  - TV正則化（勾配のL1ノルム）の更新（Z-update）
  をADMMの枠組みで交互に解く形になります。
- TV Denoisingのz-update（ソフトしきい値）と双対更新は、そのままRPCA＋TVのZ-updateに流用できます。

このように、TV DenoisingのADMM実装をベースに、低ランク成分とスパース成分の更新ステップを追加することで、RPCA＋TV正則化項を解くアルゴリズムを構築できます。
