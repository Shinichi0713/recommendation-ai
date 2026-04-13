z-update は、ADMMの「zについての最小化ステップ」です。
TV Denoising via ADMM では、拡張ラグランジュ関数（スケーリング版）を $z$ について最小化することで、**ソフトしきい値処理（shrinkage）**の形で閉形式解が得られます。

---

## 1. 拡張ラグランジュ関数（スケーリング版）

前回の導出から、スケーリングされた双対変数 $u = y/\rho$ を用いた拡張ラグランジュ関数は

$$
L_\rho(x, z, u) = \frac{1}{2} \|x - b\|_2^2 + \lambda \|z\|_1 + \frac{\rho}{2} \|Dx - z + u\|_2^2 - \frac{\rho}{2} \|u\|_2^2
$$

です。z-update では $x, u$ を固定して $z$ について最小化するので、$z$ に依存しない項は無視できます。したがって、実質的に最小化する関数は

$$
J(z) = \lambda \|z\|_1 + \frac{\rho}{2} \|Dx - z + u\|_2^2
$$

です（ここで $x, u$ は定数とみなします）。

---

## 2. z-update の最小化問題

z-update は

$$
z^{k+1} = \arg\min_z \left\{ \lambda \|z\|_1 + \frac{\rho}{2} \|Dx^{k+1} - z + u^k\|_2^2 \right\}
$$

という問題です。ここで $a = Dx^{k+1} + u^k$ とおくと、

$$
J(z) = \lambda \|z\|_1 + \frac{\rho}{2} \|a - z\|_2^2
$$

となります。これは**L1正則化付き最小二乗問題**であり、近接オペレータの形で閉形式に解けます。

---

## 3. 近接オペレータとしての解（ソフトしきい値）

一般に、関数

$$
\min_z \ \mu \|z\|_1 + \frac{1}{2} \|a - z\|_2^2
$$

の解は、要素ごとの**ソフトしきい値処理**で与えられます：

$$
z_i = \mathrm{shrink}(a_i, \mu) = 
\begin{cases}
a_i - \mu & (a_i > \mu) \\
0 & (|a_i| \le \mu) \\
a_i + \mu & (a_i < -\mu)
\end{cases}
$$

あるいは、絶対値を使って

$$
\mathrm{shrink}(a, \mu) = \max(0, a - \mu) - \max(0, -a - \mu)
$$

と書けます。

---

## 4. TV Denoising の z-update への適用

TV Denoising の場合、$J(z)$ は

$$
J(z) = \lambda \|z\|_1 + \frac{\rho}{2} \|a - z\|_2^2
$$

なので、$\mu = \lambda / \rho$ とおけば、

$$
\min_z J(z) = \min_z \left\{ \rho \left( \frac{\lambda}{\rho} \|z\|_1 + \frac{1}{2} \|a - z\|_2^2 \right) \right\}
$$

となり、スケール因子 $\rho$ は最小化に影響しません。したがって、解は

$$
z^{k+1} = \mathrm{shrink}(a, \lambda / \rho)
$$

です。ここで $a = Dx^{k+1} + u^k$ なので、

$$
z^{k+1} = \mathrm{shrink}(Dx^{k+1} + u^k, \lambda / \rho)
$$

となります。

---

## 5. まとめ

- z-update は、拡張ラグランジュ関数を $z$ について最小化するステップです。
- 最小化問題は
  $$
  \min_z \ \lambda \|z\|_1 + \frac{\rho}{2} \|Dx - z + u\|_2^2
  $$

  というL1正則化付き最小二乗問題になり、解は要素ごとのソフトしきい値処理で与えられます。
- したがって、
  $$
  z^{k+1} = \mathrm{shrink}(Dx^{k+1} + u^k, \lambda / \rho)
  $$

  という更新式が得られます。

この式は、TV正則化項（L1ノルム）と二次ペナルティ項のバランスを取る形で、$z$ を「しきい値処理された勾配」として更新していることを表しています。
