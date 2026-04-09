RPCA（Robust Principal Component Analysis）にTV（Total Variation）正則化項を適用したモデルは、一般に **TVRPCA** と呼ばれます。ここでは、動画の移動物体検出を想定した代表的な定式化を紹介します。

---

## 1. 観測モデルと分解

観測された動画フレームをベクトル化して並べた行列を \( O \) とします。これを

\[
O = B + M
\]

と分解し、さらに \( M \) を

\[
M = F + E
\]

と分解します。

- \( B \)：低ランクな **静的背景**（low-rank static background）
- \( F \)：**スパースかつ滑らかな前景**（sparse and smooth foreground）
- \( E \)：**よりスパースな動的背景（ノイズ）**（sparser dynamic background）

---

## 2. TVRPCA の目的関数（凸代理問題）

TVRPCA の凸代理問題としての目的関数は、以下のように書かれます[Total Variation Regularized RPCA for Irregularly Moving Object Detection](https://yangliang.github.io/pdf/07089247.pdf)：

\[
\min_{B, M, F, E} \|B\|_* + \lambda_1 \|M\|_1 + \lambda_2 \|E\|_1 + \lambda_3 \|D f\|_q
\]
\[
\text{s.t.} \quad O = B + M,\quad M = F + E
\]

ここで：

- \( f = \mathrm{vec}(F) \)：前景行列 \( F \) をベクトル化したもの
- \( D \)：差分演算子（horizontal, vertical, temporal の差分を縦に並べた行列）

---

## 3. 各項の意味

### (1) 低ランク項（静的背景）

\[
\|B\|_*
\]

- \( \|B\|_* \) は行列 \( B \) の **核ノルム（nuclear norm）** です。
- これは \( B \) の特異値の和であり、ランク関数の凸緩和として用いられます。
- 背景が時間的に変化が少なく、低ランク構造を持つことを仮定しています。

### (2) スパース項（動的背景・ノイズ）

\[
\lambda_1 \|M\|_1 + \lambda_2 \|E\|_1
\]

- \( \|M\|_1, \|E\|_1 \) は行列要素の **L1 ノルム**（要素ごとの絶対値の和）です。
- \( M \) は前景＋動的背景の残差、\( E \) は動的背景（ノイズ）を表し、それぞれが **スパース** であることを促します。
- \( \lambda_1, \lambda_2 \) はスパース性の強さを制御する正則化パラメータです。

### (3) TV 正則化項（前景の滑らかさ）

\[
\lambda_3 \|D f\|_q
\]

ここで：

- \( f = \mathrm{vec}(F) \)：前景行列 \( F \) をベクトル化したベクトル
- \( D \) は差分演算子の連結行列：
  \[
  D = \begin{bmatrix} D_h^\top & D_v^\top & D_t^\top \end{bmatrix}^\top
  \]
  - \( D_h \)：水平方向差分（空間的な横方向の勾配）
  - \( D_v \)：垂直方向差分（空間的な縦方向の勾配）
  - \( D_t \)：時間方向差分（フレーム間の変化）
- \( \|D f\|_q \) は、通常 \( q = 1 \) または \( q = 2 \) の **TV ノルム** です。
  - \( q = 1 \) のとき：各方向の差分の L1 ノルム（勾配のスパース性を重視）
  - \( q = 2 \) のとき：L2 ノルム（より滑らかな変化を重視）

この項は、前景 \( F \) が **空間的・時間的に連続な（滑らかな）動き** を持つことを仮定し、ノイズによる孤立した点状の変化を抑制します。

---

## 4. まとめ

RPCA に TV 正則化項を適用した場合の理論式は、観測 \( O \) を

- 低ランク背景 \( B \)（核ノルム）
- スパースな残差 \( M, E \)（L1 ノルム）
- TV 正則化された前景 \( F \)（勾配の L1/L2 ノルム）

に分解する次の最適化問題として与えられます：

\[
\min_{B, M, F, E} \|B\|_* + \lambda_1 \|M\|_1 + \lambda_2 \|E\|_1 + \lambda_3 \|D f\|_q
\]
\[
\text{s.t.} \quad O = B + M,\quad M = F + E
\]

このモデルは、特に動画中の移動物体検出や、空間的・時間的に連続な構造を持つ信号の分離に用いられます[Total Variation Regularized RPCA for Irregularly Moving Object Detection](https://yangliang.github.io/pdf/07089247.pdf)。


