Z-updateの式

$$
Z^{k+1} = \mathrm{shrink}(D L^{k+1} + U_2^k, \mu / \rho_2)
$$

は、**L1正則化付き最小二乗問題の近接オペレータ**として導出されます。  
以下、ステップごとに説明します。

---

## 1. Z-updateの最小化問題

Z-updateでは、$L, U_2$ を固定して $Z$ について

$$
Z^{k+1} = \arg\min_Z \left\{ \mu \|Z\|_1 + \frac{\rho_2}{2} \|D L^{k+1} - Z + U_2^k\|_F^2 \right\}
$$

を解きます。ここで $A = D L^{k+1} + U_2^k$ とおくと、

$$
J(Z) = \mu \|Z\|_1 + \frac{\rho_2}{2} \|A - Z\|_F^2
$$

となります。

---

## 2. 要素ごとの分解

フロベニウスノルム $\|A - Z\|_F^2$ は要素の二乗和なので、

$$
\|A - Z\|_F^2 = \sum_{i,j} (A_{ij} - Z_{ij})^2
$$

です。またL1ノルム $\|Z\|_1$ も要素の絶対値の和なので、

$$
\|Z\|_1 = \sum_{i,j} |Z_{ij}|
$$

です。したがって、目的関数は**各要素 $(i,j)$ について独立**に最小化できます：

$$
J(Z) = \sum_{i,j} \left( \mu |Z_{ij}| + \frac{\rho_2}{2} (A_{ij} - Z_{ij})^2 \right)
$$

よって、各 $(i,j)$ について

$$
\min_{Z_{ij}} \ \mu |Z_{ij}| + \frac{\rho_2}{2} (A_{ij} - Z_{ij})^2
$$

を解けばよいことになります。

---

## 3. スカラーL1正則化付き最小二乗問題の解

一般に、スカラー変数 $z$ に対する問題

$$
\min_z \ \mu |z| + \frac{\rho}{2} (a - z)^2
$$

の解は、**ソフトしきい値処理（soft thresholding）**で与えられます：

$$
z = \mathrm{shrink}(a, \mu / \rho) = 
\begin{cases}
a - \mu/\rho & (a > \mu/\rho) \\
0 & (|a| \le \mu/\rho) \\
a + \mu/\rho & (a < -\mu/\rho)
\end{cases}
$$

あるいは絶対値を使って

$$
\mathrm{shrink}(a, \kappa) = \max(0, a - \kappa) - \max(0, -a - \kappa)
$$

と書けます。

---

## 4. Z-updateへの適用

Z-updateの場合、各要素について

$$
\min_{Z_{ij}} \ \mu |Z_{ij}| + \frac{\rho_2}{2} (A_{ij} - Z_{ij})^2
$$

を解くので、$\kappa = \mu / \rho_2$ とおけば、解は

$$
Z_{ij}^{k+1} = \mathrm{shrink}(A_{ij}, \mu / \rho_2)
$$

です。ここで $A = D L^{k+1} + U_2^k$ なので、行列として書くと

$$
Z^{k+1} = \mathrm{shrink}(D L^{k+1} + U_2^k, \mu / \rho_2)
$$

となります。

---

## 5. まとめ

- Z-updateの最小化問題は、要素ごとに独立なL1正則化付き最小二乗問題に分解できます。
- 各要素の問題は、ソフトしきい値処理で閉形式に解けます。
- したがって、
  $$
  Z^{k+1} = \mathrm{shrink}(D L^{k+1} + U_2^k, \mu / \rho_2)
  $$
  という更新式が得られます。

この式は、TV正則化項（$\mu \|Z\|_1$）と「$Z$ が $D L$ に近いこと」を表す二次ペナルティ項のバランスを取る形で、$Z$ を「しきい値処理された勾配」として更新していることを表しています。