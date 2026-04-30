TV正則化項を持つRPCA（Total Variation Regularized RPCA）は、観測行列 \(Y\) を

- **低ランク成分** \(L\)（背景など）
- **スパース成分** \(S\)（前景・外れ値など）
- **TV正則化をかけた成分**（画像の滑らかさを保つ成分）

に分解するモデルです。ここでは、画像系列（ビデオ）を想定した**行列版RPCA＋TV**の一般的な形と、ADMMによる更新式を説明します。

---

## 1. TV正則化付きRPCAの定義（目的関数）

観測行列 \(Y \in \mathbb{R}^{m \times n}\)（例：各列が1フレームの画像ベクトル）に対し、

\[
Y = L + S + E
\]

と分解します。

- \(L\)：低ランク成分（背景）
- \(S\)：スパース成分（前景・ノイズ）
- \(E\)：ガウス雑音などの小さい外乱（必要に応じて）

TV正則化付きRPCAの目的関数は、例えば次のように書けます：

\[
\begin{aligned}
\min_{L,S} &\quad \frac{1}{2} \|Y - L - S\|_F^2 \\
&\quad + \lambda_L \|L\|_* \quad \text{（低ランク正則化：核ノルム）} \\
&\quad + \lambda_S \|S\|_1 \quad \text{（スパース正則化：L1ノルム）} \\
&\quad + \lambda_{\mathrm{TV}} \|\nabla S\|_1 \quad \text{（TV正則化：Total Variation）}
\end{aligned}
\]

ここで

- \(\|L\|_*\)：特異値の和（核ノルム）
- \(\|S\|_1\)：要素ごとのL1ノルム
- \(\|\nabla S\|_1\)：画像として解釈した \(S\) に対する**Total Variationノルム**（各フレームの勾配のL1ノルム）

TV項は通常、**空間方向の勾配**（あるいは時空間3D-TV）をとり、前景 \(S\) の**ピースワイズ定数性（エッジ保存の平滑化）**を促します。

---

## 2. ADMMによる定式化と更新式（概要）

上記の目的関数は、ADMM（Alternating Direction Method of Multipliers）で効率的に解くことができます。ここでは、変数を分離しやすい形に書き直します。

### 2.1 補助変数の導入

TV項を扱いやすくするため、補助変数 \(Z = \nabla S\) を導入します。すると目的関数は

\[
\begin{aligned}
\min_{L,S,Z} &\quad \frac{1}{2} \|Y - L - S\|_F^2 + \lambda_L \|L\|_* + \lambda_S \|S\|_1 + \lambda_{\mathrm{TV}} \|Z\|_1 \\
\text{s.t.} &\quad Z = \nabla S
\end{aligned}
\]

となります。

### 2.2 拡張ラグランジュ関数

制約 \(Z = \nabla S\) に対するラグランジュ乗数を \(\Lambda\) とし、ペナルティ係数を \(\rho > 0\) とすると、拡張ラグランジュ関数は

\[
\begin{aligned}
\mathcal{L}(L,S,Z,\Lambda) = &\ \frac{1}{2} \|Y - L - S\|_F^2 + \lambda_L \|L\|_* + \lambda_S \|S\|_1 \\
&+ \lambda_{\mathrm{TV}} \|Z\|_1 + \langle \Lambda, Z - \nabla S \rangle + \frac{\rho}{2} \|Z - \nabla S\|_F^2
\end{aligned}
\]

です。

---

## 3. ADMMの更新式（各変数ごとの最小化）

ADMMでは、以下の変数を交互に最小化します。

### (1) \(L\) の更新（低ランク成分）

\(S,Z\) を固定したときの \(L\) に関する部分問題は

\[
\min_L \frac{1}{2} \|Y - L - S\|_F^2 + \lambda_L \|L\|_*
\]

です。これは**核ノルム正則化付き最小二乗**であり、解は

\[
L \leftarrow \mathrm{prox}_{\lambda_L \|\cdot\|_*}(Y - S)
\]

で与えられます。ここで \(\mathrm{prox}\) は**特異値しきい値処理（singular value thresholding）**です：

- \(Y - S = U \Sigma V^\top\) と特異値分解
- \(\tilde{\Sigma}_{ii} = \max(\Sigma_{ii} - \lambda_L, 0)\)
- \(L = U \tilde{\Sigma} V^\top\)

### (2) \(S\) の更新（スパース成分＋TV制約）

\(L,Z\) を固定したときの \(S\) に関する部分問題は

\[
\min_S \frac{1}{2} \|Y - L - S\|_F^2 + \lambda_S \|S\|_1 + \frac{\rho}{2} \|Z - \nabla S + \Lambda/\rho\|_F^2
\]

です。これは

- データ整合項（L2）
- L1正則化項
- TV制約からの乖離（L2）

の和であり、**線形作用素 \(\nabla\) を含む二次関数＋L1**の最小化問題になります。
この更新は、例えば**共役勾配法（CG）**や**FFTを用いた高速解法**（周期境界仮定など）で解きます。

### (3) \(Z\) の更新（TV補助変数）

\(L,S\) を固定したときの \(Z\) に関する部分問題は

\[
\min_Z \lambda_{\mathrm{TV}} \|Z\|_1 + \frac{\rho}{2} \|Z - (\nabla S - \Lambda/\rho)\|_F^2
\]

です。これは**L1正則化付き最小二乗**であり、解は**ソフトしきい値演算（soft thresholding）**で与えられます：

\[
Z \leftarrow \mathrm{soft}\left(\nabla S - \frac{\Lambda}{\rho},\ \frac{\lambda_{\mathrm{TV}}}{\rho}\right)
\]

ここで

\[
\mathrm{soft}(x,\tau) = \mathrm{sign}(x) \cdot \max(|x| - \tau, 0)
\]

（要素ごと）です。

### (4) 双対変数 \(\Lambda\) の更新

最後に、双対変数（ラグランジュ乗数）を

\[
\Lambda \leftarrow \Lambda + \rho (Z - \nabla S)
\]

で更新します。

---

## 4. まとめ

- **TV正則化付きRPCA**は、観測 \(Y\) を低ランク \(L\)、スパース \(S\)、TV正則化項を持つ成分に分解するモデルです。
- 目的関数は、データ整合項（L2）＋低ランク正則化（核ノルム）＋スパース正則化（L1）＋TV正則化（勾配のL1）からなります。
- **ADMM**を用いると、各変数（\(L, S, Z, \Lambda\)）ごとに**近接演算（proximal operator）**や**線形システムの求解**に帰着でき、効率的に最適化できます。

実際の実装では、画像の離散勾配演算子 \(\nabla\) の具体的な形（前進差分・中心差分など）や境界条件に応じて、\(S\) の更新式が少し変わりますが、上記が基本的な枠組みです。[Total Variation Regularized Tensor RPCA for Background Subtraction](https://arxiv.org/abs/1503.01868)
