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

