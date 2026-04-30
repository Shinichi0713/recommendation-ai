L1 と L2 の正則化項がそれぞれ複数ある場合の ADMM（Alternating Direction Method of Multipliers）による最適化を導出します。

---

## 1. 問題設定

目的関数を一般化して

\[
\min_{x} \; f(x) + \sum_{i=1}^{m_1} \lambda_i \|A_i x\|_1 + \sum_{j=1}^{m_2} \mu_j \|B_j x\|_2^2
\]

とします。ここで

- \(x \in \mathbb{R}^n\)：最適化変数
- \(f(x)\)：データ適合項（例：\(\frac12\|Cx - d\|_2^2\)）
- \(A_i \in \mathbb{R}^{p_i \times n}\)：L1 正則化の線形変換
- \(B_j \in \mathbb{R}^{q_j \times n}\)：L2 正則化の線形変換
- \(\lambda_i, \mu_j > 0\)：正則化パラメータ

---

## 2. ADMM のための変数分割

ADMM では、分離可能な項ごとに補助変数を導入します。

- L1 正則化項に対応する補助変数：
  \[
  z_i = A_i x \quad (i=1,\dots,m_1)
  \]
- L2 正則化項に対応する補助変数：
  \[
  w_j = B_j x \quad (j=1,\dots,m_2)
  \]

これにより、元の問題は

\[
\begin{aligned}
\min_{x, z, w} &\quad f(x) + \sum_{i=1}^{m_1} \lambda_i \|z_i\|_1 + \sum_{j=1}^{m_2} \mu_j \|w_j\|_2^2 \\
\text{s.t.} &\quad z_i = A_i x \quad (i=1,\dots,m_1), \\
&\quad w_j = B_j x \quad (j=1,\dots,m_2)
\end{aligned}
\]

という制約付き最適化問題になります。

---

## 3. 拡張ラグランジュ関数

制約 \(z_i = A_i x\)、\(w_j = B_j x\) に対して、それぞれ双対変数 \(u_i \in \mathbb{R}^{p_i}\)、\(v_j \in \mathbb{R}^{q_j}\) とペナルティパラメータ \(\rho_i, \sigma_j > 0\) を導入します。

拡張ラグランジュ関数は

\[
\begin{aligned}
L_\rho(x, z, w, u, v)
&= f(x)
+ \sum_{i=1}^{m_1} \left\{ \lambda_i \|z_i\|_1 + u_i^\top (A_i x - z_i) + \frac{\rho_i}{2} \|A_i x - z_i\|_2^2 \right\} \\
&\quad + \sum_{j=1}^{m_2} \left\{ \mu_j \|w_j\|_2^2 + v_j^\top (B_j x - w_j) + \frac{\sigma_j}{2} \|B_j x - w_j\|_2^2 \right\}.
\end{aligned}
\]

---

## 4. ADMM の更新式

ADMM では、以下の 3 ステップを交互に更新します。

### (1) \(x\)-更新（\(z, w, u, v\) を固定）

\[
x^{k+1} = \arg\min_x L_\rho(x, z^k, w^k, u^k, v^k).
\]

目的関数を \(x\) について整理すると、

\[
\begin{aligned}
x^{k+1}
&= \arg\min_x \; f(x) \\
&\quad + \sum_{i=1}^{m_1} \left\{ u_i^{k\top} A_i x + \frac{\rho_i}{2} \|A_i x - z_i^k\|_2^2 \right\} \\
&\quad + \sum_{j=1}^{m_2} \left\{ v_j^{k\top} B_j x + \frac{\sigma_j}{2} \|B_j x - w_j^k\|_2^2 \right\}.
\end{aligned}
\]

これは \(x\) について**滑らかな凸最小化問題**です。  
特に \(f(x) = \frac12\|Cx - d\|_2^2\) のような二次形式なら、正規方程式

\[
\left( C^\top C + \sum_{i=1}^{m_1} \rho_i A_i^\top A_i + \sum_{j=1}^{m_2} \sigma_j B_j^\top B_j \right) x
= C^\top d + \sum_{i=1}^{m_1} A_i^\top (\rho_i z_i^k - u_i^k) + \sum_{j=1}^{m_2} B_j^\top (\sigma_j w_j^k - v_j^k)
\]

を解くことで \(x^{k+1}\) が得られます（一般の滑らかな \(f\) の場合は勾配法・ニュートン法などで解きます）。

---

### (2) \(z\)-更新（\(x, w, u, v\) を固定）

各 \(i\) について \(z_i\) は独立に更新できます：

\[
z_i^{k+1} = \arg\min_{z_i} \; \lambda_i \|z_i\|_1 - u_i^{k\top} z_i + \frac{\rho_i}{2} \|A_i x^{k+1} - z_i\|_2^2.
\]

線形項をまとめると、

\[
z_i^{k+1} = \arg\min_{z_i} \; \lambda_i \|z_i\|_1 + \frac{\rho_i}{2} \|z_i - (A_i x^{k+1} + \frac{1}{\rho_i} u_i^k)\|_2^2.
\]

これは **L1 正則化付き最小二乗（Lasso 型）**の proximal 問題であり、解は**ソフト閾値作用素（soft-thresholding）**で与えられます：

\[
z_i^{k+1} = S_{\lambda_i/\rho_i}\!\left( A_i x^{k+1} + \frac{1}{\rho_i} u_i^k \right),
\]

ここで

\[
S_\kappa(a) = \mathrm{sign}(a) \odot \max(|a| - \kappa, 0)
\]

（成分ごとの soft-thresholding）です。

---

### (3) \(w\)-更新（\(x, z, u, v\) を固定）

各 \(j\) について \(w_j\) は独立に更新できます：

\[
w_j^{k+1} = \arg\min_{w_j} \; \mu_j \|w_j\|_2^2 - v_j^{k\top} w_j + \frac{\sigma_j}{2} \|B_j x^{k+1} - w_j\|_2^2.
\]

これは \(w_j\) についての**二次関数**であり、微分して 0 とおくと閉形式解が得られます：

\[
w_j^{k+1} = \left( 2\mu_j I + \sigma_j I \right)^{-1} \left( \sigma_j B_j x^{k+1} + v_j^k \right)
= \frac{\sigma_j B_j x^{k+1} + v_j^k}{2\mu_j + \sigma_j}.
\]

---

### (4) 双対変数 \(u, v\) の更新

双対変数は、制約違反に比例して更新します：

\[
\begin{aligned}
u_i^{k+1} &= u_i^k + \rho_i (A_i x^{k+1} - z_i^{k+1}), \\
v_j^{k+1} &= v_j^k + \sigma_j (B_j x^{k+1} - w_j^{k+1}).
\end{aligned}
\]

---

## 5. アルゴリズムのまとめ

ADMM による反復は、初期値 \(x^0, z^0, w^0, u^0, v^0\) から始めて、各ステップで以下を繰り返します：

1. **\(x\)-更新**：滑らかな凸最小化（二次なら正規方程式、一般なら勾配法など）
2. **\(z\)-更新**：各 \(i\) について soft-thresholding
3. **\(w\)-更新**：各 \(j\) について閉形式の二次最小化
4. **双対更新**：\(u_i, v_j\) を制約違反に比例して更新

これにより、L1 正則化項と L2 正則化項がそれぞれ複数ある場合でも、**変数分割と ADMM によって、滑らかな部分（\(f(x)\) と L2 正則化）と非滑らかな部分（L1 正則化）を分離して効率的に最適化**できます。

---

以上が、L1 と L2 の正則化項がそれぞれ複数ある場合の ADMM による最適化の導出です。


## 更新式の導出

「L1+L2 のラグランジュ関数」という表現だけでは問題が一意に定まらないので、**典型的な L1 正則化付き最小二乗問題**を例に、ADMM の更新式を導出します。

---

## 1. 問題設定

L1 正則化付き最小二乗（Lasso 型）問題を考えます：

\[
\min_x \; \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_1
\]

ここで
- \(A \in \mathbb{R}^{m \times n}\), \(b \in \mathbb{R}^m\), \(x \in \mathbb{R}^n\),
- \(\lambda > 0\) は正則化パラメータです。

ADMM を適用するために、補助変数 \(z\) を導入し、以下の等価問題に書き換えます：

\[
\begin{aligned}
\min_{x,z} &\quad \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|z\|_1 \\
\text{s.t.} &\quad x - z = 0
\end{aligned}
\]

このとき、目的関数は
- \(f(x) = \frac{1}{2} \|Ax - b\|_2^2\)（L2 部分）
- \(g(z) = \lambda \|z\|_1\)（L1 部分）

に分離されています。

---

## 2. 拡張ラグランジュ関数

等式制約 \(x - z = 0\) に対する拡張ラグランジュ関数は

\[
L_\rho(x, z, u) = \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|z\|_1 + u^\top (x - z) + \frac{\rho}{2} \|x - z\|_2^2
\]

です。ここで
- \(u \in \mathbb{R}^n\) はラグランジュ乗数（双対変数）、
- \(\rho > 0\) はペナルティパラメータです。

ADMM は、この \(L_\rho\) に対して以下の**交互最小化**を行います：

1. \(x\)-更新：\(z, u\) を固定して \(L_\rho\) を \(x\) について最小化
2. \(z\)-更新：\(x, u\) を固定して \(L_\rho\) を \(z\) について最小化
3. \(u\)-更新：双対変数 \(u\) を更新

---

## 3. \(x\)-更新式の導出

\(z = z^k\), \(u = u^k\) を固定し、\(x\) について最小化します：

\[
x^{k+1} = \arg\min_x \; \frac{1}{2} \|Ax - b\|_2^2 + (u^k)^\top (x - z^k) + \frac{\rho}{2} \|x - z^k\|_2^2
\]

定数項を無視して整理すると、

\[
x^{k+1} = \arg\min_x \; \frac{1}{2} \|Ax - b\|_2^2 + \frac{\rho}{2} \|x - z^k + \frac{u^k}{\rho}\|_2^2
\]

これは**二次関数の最小化問題**です。勾配をゼロとおいて解きます。

目的関数を
\[
J(x) = \frac{1}{2} \|Ax - b\|_2^2 + \frac{\rho}{2} \|x - v^k\|_2^2,\quad v^k = z^k - \frac{u^k}{\rho}
\]
とおくと、勾配は
\[
\nabla_x J(x) = A^\top(Ax - b) + \rho (x - v^k)
\]
です。これをゼロとおいて
\[
A^\top A x + \rho x = A^\top b + \rho v^k
\]
\[
(A^\top A + \rho I) x = A^\top b + \rho v^k
\]
したがって
\[
x^{k+1} = (A^\top A + \rho I)^{-1} \bigl( A^\top b + \rho v^k \bigr),\quad v^k = z^k - \frac{u^k}{\rho}
\]
が \(x\)-更新式です。

---

## 4. \(z\)-更新式の導出

\(x = x^{k+1}\), \(u = u^k\) を固定し、\(z\) について最小化します：

\[
z^{k+1} = \arg\min_z \; \lambda \|z\|_1 + (u^k)^\top (x^{k+1} - z) + \frac{\rho}{2} \|x^{k+1} - z\|_2^2
\]

\(u^k\) に関する項をまとめると、

\[
z^{k+1} = \arg\min_z \; \lambda \|z\|_1 + \frac{\rho}{2} \|z - (x^{k+1} + \frac{u^k}{\rho})\|_2^2
\]

これは**L1 ノルム付き二次関数の最小化**であり、**ソフト閾値作用素（soft-thresholding）**で解けます。

具体的には、各成分ごとに独立な問題になります。\(w = x^{k+1} + \frac{u^k}{\rho}\) とおくと、

\[
z^{k+1}_i = \operatorname{soft}_{\lambda/\rho}(w_i)
= 
\begin{cases}
w_i - \frac{\lambda}{\rho} & w_i > \frac{\lambda}{\rho} \\
0 & |w_i| \le \frac{\lambda}{\rho} \\
w_i + \frac{\lambda}{\rho} & w_i < -\frac{\lambda}{\rho}
\end{cases}
\]

あるいはコンパクトに

\[
z^{k+1}_i = \operatorname{sign}(w_i) \max\bigl(|w_i| - \frac{\lambda}{\rho},\; 0\bigr)
\]

と書けます。

---

## 5. \(u\)-更新式の導出

双対変数 \(u\) は、**制約の違反量**に比例して更新されます：

\[
u^{k+1} = u^k + \rho (x^{k+1} - z^{k+1})
\]

これは拡張ラグランジュ法の標準的な更新則です。

---

## 6. まとめ：ADMM の更新式（L1+L2 の場合）

以上をまとめると、L1 正則化付き最小二乗問題

\[
\min_x \; \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_1
\]

に対する ADMM の更新式は、補助変数 \(z\) と双対変数 \(u\) を導入して

\[
\begin{aligned}
x^{k+1} &= (A^\top A + \rho I)^{-1} \bigl( A^\top b + \rho (z^k - \frac{u^k}{\rho}) \bigr) \\
z^{k+1} &= \operatorname{soft}_{\lambda/\rho}\bigl( x^{k+1} + \frac{u^k}{\rho} \bigr) \\
u^{k+1} &= u^k + \rho (x^{k+1} - z^{k+1})
\end{aligned}
\]

となります。ここで \(\operatorname{soft}_{\tau}(w)\) はソフト閾値作用素です。

---

### 補足

- 上記は「L2 部分が最小二乗、L1 部分が L1 正則化」という典型的なケースです。
- もし「L1+L2 のラグランジュ関数」が別の形（例えば Elastic Net など）を指している場合は、目的関数の形に応じて \(f(x), g(z)\) を適切に定義し、同様の手順で ADMM の更新式を導出できます。
- 一般に、ADMM では
  - \(f(x)\) 側：滑らかな最適化（多くの場合は二次最小化）
  - \(g(z)\) 側：近接作用素（proximal operator）による更新
  という分離ができるように補助変数を導入します。

## 補助変数導入理由

上記の L1 正則化付き最小二乗問題

\[
\min_x \; \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_1
\]

で補助変数 \(z\) を導入した理由は、主に以下の3点です。

---

### 1. L1 と L2 を「変数レベルで分離」するため

元の問題では、目的関数が

- L2 部分：\(\frac{1}{2} \|Ax - b\|_2^2\)（滑らかで微分可能）
- L1 部分：\(\lambda \|x\|_1\)（非滑らかで微分不可能）

という**性質の異なる2つの項**が、**同じ変数 \(x\) に同時にかかっている**形になっています。

このままでは、

- 勾配法を使おうとすると L1 部分でつまずく（劣勾配法などが必要）、
- 近接勾配法なども使えますが、ADMM の利点を活かしにくい

という問題があります。

そこで補助変数 \(z\) を導入し、

\[
\min_{x,z} \; \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|z\|_1 \quad \text{s.t.} \quad x = z
\]

と書き換えることで、

- \(x\) には L2 部分だけ（滑らかな最小二乗）
- \(z\) には L1 部分だけ（非滑らかな L1 正則化）

というように、**変数ごとに役割を分離**します。

---

### 2. ADMM の「交互に簡単なサブ問題を解く」構造を作るため

ADMM は、拡張ラグランジュ関数

\[
L_\rho(x, z, u) = \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|z\|_1 + u^\top (x - z) + \frac{\rho}{2} \|x - z\|_2^2
\]

に対して、

1. \(x\)-更新：\(z, u\) 固定 → L2 の二次最小化（解析的に解ける）
2. \(z\)-更新：\(x, u\) 固定 → L1 付き二次最小化（ソフト閾値で解ける）
3. \(u\)-更新：双対変数の更新

という**交互最小化**を行います。

補助変数 \(z\) を導入していない元の問題では、このような「変数ごとに分離された簡単なサブ問題」を作ることができません。  
補助変数を入れることで、

- \(x\)-更新：滑らかな凸二次計画（行列の逆を一度計算すればよい）
- \(z\)-更新：L1 ノルム付き二次関数 → ソフト閾値作用素で成分ごとに閉形式

という、**それぞれが効率的に解ける形**になります。

---

### 3. L1 ノルムの「近接作用素」を自然に使えるようにするため

L1 ノルム \(\lambda \|z\|_1\) に対する近接作用素は、ソフト閾値作用素としてよく知られています：

\[
\operatorname{prox}_{\lambda \|\cdot\|_1}(w) = \operatorname{soft}_{\lambda}(w)
\]

ADMM の \(z\)-更新ステップは、まさにこの近接作用素を呼び出す形になります：

\[
z^{k+1} = \arg\min_z \; \lambda \|z\|_1 + \frac{\rho}{2} \|z - w\|_2^2 = \operatorname{soft}_{\lambda/\rho}(w)
\]

補助変数 \(z\) を導入することで、**L1 ノルムを「z だけの関数」に分離**し、この近接作用素をそのまま適用できる形にしています。

---

### まとめ

上記の L1+L2 問題で補助変数 \(z\) を導入した理由は、

1. **L2 部分（滑らか）と L1 部分（非滑らか）を変数レベルで分離するため**
2. **ADMM が「x と z を交互に簡単に更新できる」形にするため**
3. **L1 ノルムに対する近接作用素（ソフト閾値）を自然に使えるようにするため**

です。

これにより、元の問題を「滑らかな最小二乗」と「L1 正則化の近接作用素」という2つの簡単なサブ問題に分解し、効率的に解くことができます。