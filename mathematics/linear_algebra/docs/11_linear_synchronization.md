## 回転運動の合成

線形写像の「合成」と行列の「積」がなぜあのような複雑な計算（行×列）になるのか、その理由は **「写像を順番に適用した結果を、一つの指示書（行列）にまとめたい」** という数学的な要請にあります。

### 1. 概念のイメージ：リレー形式の変換

2つの線形写像 **$f$** と **$g$** があるとします。

* **写像 **$g$**:** ベクトル **$\mathbf{x}$** を **$\mathbf{y}$** に移す（第1の変換）
* **写像 **$f$**:** ベクトル **$\mathbf{y}$** を **$\mathbf{z}$** に移す（第2の変換）

この2つを連続して行う操作 **$f(g(\mathbf{x}))$** を**合成写像**と呼びます。

### 2. 数学的な導出：なぜ「積」になるのか

それぞれの写像を、対応する行列 **$A$** と **$B$** で表現してみましょう。

1. まず **$g$** を適用する： **$\mathbf{y} = B\mathbf{x}$**
2. 次に **$f$** を適用する： **$\mathbf{z} = A\mathbf{y}$**

この2つをガッチャンコさせると、以下のようになります。

$$
\mathbf{z} = A(B\mathbf{x})
$$

行列の結合法則により、これは次のように書き換えられます。

$$
\mathbf{z} = (AB)\mathbf{x}
$$

つまり、 **「写像 **$g$** をやってから **$f$** をやる」という一連の流れは、新しい一つの行列 **$C = AB$** を掛けることと等しい** のです。

### 3. なぜ行列の積は「行×列」なのか？

行列の積 **$AB$** の成分を計算する際、なぜ「**$A$** の行」と「**$B$** の列」を掛け合わせるのでしょうか。

それは、**「**$B$** によって引っ越した後の基底ベクトルたちに、さらに **$A$** という変換を施した結果」を計算しているから**です。

* **$B$** の各列は、標準基底が **$g$** によってどこへ飛んだかを表します。
* その「飛んでいった先のベクトル」に対して **$A$** を掛ける（＝ **$A$** の各行と内積をとる）ことで、最終的な目的地が決まります。

### 4. 注意点：順番が命！

線形写像の合成において、**順番は極めて重要**です。

通常、**$AB \neq BA$** です。

* **$AB$:** **$B$** を先にやってから **$A$** をやる。
* **$BA$:** **$A$** を先にやってから **$B$** をやる。

例えば、「**$90^\circ$** 回転させてから、横に引き延ばす」のと、「横に引き延ばしてから、**$90^\circ$** 回転させる」のでは、最終的な形や向きが変わってしまいますよね。行列の積が交換不能なのは、この **「操作の順番」** という物理的な直感と一致しています。

### 5. Pythonによる「合成」の確認

「回転」の後に「引き延ばし」を行う合成行列を作ってみましょう。

**Python**

```python
import numpy as np

# A: y軸方向に2倍に引き延ばす行列
A = np.array([[1, 0],
              [0, 2]])

# B: 90度回転させる行列
B = np.array([[0, -1],
              [1, 0]])

# 合成行列 C = AB (Bを適用してからAを適用)
C = A @ B

# テストベクトル [1, 0] (右向き)
v = np.array([1, 0])

# 順番に適用: 回転して [0, 1] になり、引き延ばされて [0, 2] になる
print(f"C @ v = {C @ v}")  # 結果: [0, 2]
```

### 6.回転行列の合成
次に合成の例として回転行列を扱います。

__1. 2次元回転行列の定義__

「角度 $\alpha$ の回転 $R(\alpha)$」の後に「角度 $\beta$ の回転 $R(\beta)$」を合成した行列 $R(\beta)R(\alpha)$ を計算してみます。

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

__2. 回転の合成（行列の積）__

回転行列の合成は、行列の積を理解する上で最も美しい例の一つです。なぜなら、「角度 α の回転」の後に「角度 β の回転」を行えば、結果は「角度 α+β の回転」になるはずだという直感が、行列計算によって完璧に証明されるからです。

$$R(\beta)R(\alpha) = \begin{pmatrix} \cos\beta & -\sin\beta \\ \sin\beta & \cos\beta \end{pmatrix} \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}$$

この行列の積（行 $\times$ 列）を計算すると、各成分は以下のようになります。

1. **左上:** $\cos\beta\cos\alpha - \sin\beta\sin\alpha$
2. **右上:** $-(\cos\beta\sin\alpha + \sin\beta\cos\alpha)$
3. **左下:** $\sin\beta\cos\alpha + \cos\beta\sin\alpha$
4. **右下:** $-\sin\beta\sin\alpha + \cos\beta\cos\alpha$

ここで、 **「三角関数の加法定理」** を思い出してください。

- $\cos(\alpha + \beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$
- $\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$

これらを当てはめると、なんと計算結果は以下のようになります。

$$R(\beta)R(\alpha) = \begin{pmatrix} \cos(\alpha+\beta) & -\sin(\alpha+\beta) \\ \sin(\alpha+\beta) & \cos(\alpha+\beta) \end{pmatrix} = R(\alpha + \beta)$$

**「行列の積」が「角度の足し算」に対応している** ことが数学的に証明されました。

__3. Pythonによる可視化__

「$30^\circ$ 回転」と「$60^\circ$ 回転」を合成して、合計「$90^\circ$ 回転」になる様子を確認しましょう。

```python
import numpy as np
import matplotlib.pyplot as plt

def get_rot_matrix(deg):
    rad = np.radians(deg)
    return np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad),  np.cos(rad)]
    ])

# 1. 各回転行列を作成
R1 = get_rot_matrix(30)
R2 = get_rot_matrix(60)

# 2. 行列を合成 (R2 * R1 は「30度やってから60度」)
R_total = R2 @ R1

# 3. テストベクトル (x軸方向の [1, 0])
v = np.array([1, 0])

# 各段階のベクトル
v1 = R1 @ v          # 30度回転
v2 = R_total @ v     # 30+60 = 90度回転

# --- プロット ---
plt.figure(figsize=(6,6))
plt.quiver(0, 0, v[0], v[1], color='black', angles='xy', scale_units='xy', scale=1, label='Original (0°)')
plt.quiver(0, 0, v1[0], v1[1], color='blue', angles='xy', scale_units='xy', scale=1, label='After R1 (30°)')
plt.quiver(0, 0, v2[0], v2[1], color='red', angles='xy', scale_units='xy', scale=1, label='After R2*R1 (90°)')

plt.xlim(-0.2, 1.2); plt.ylim(-0.2, 1.2)
plt.axhline(0, color='gray', lw=0.5); plt.axvline(0, color='gray', lw=0.5)
plt.legend(); plt.grid(True)
plt.title("Composition of Rotation Matrices")
plt.show()

```

__4. なぜこれがすごいのか？__

回転行列のような **ユニタリ行列（直交行列）** の合成は、以下の素晴らしい性質を持ちます。

- **情報の保存:** $R(\alpha)$ も $R(\beta)$ も行列式は $1$ です。合成した $R(\alpha+\beta)$ も行列式は $1$（面積が変わらない）。
- **可換性（特殊例）:** 一般に行列の積 $AB \neq BA$ ですが、2次元の回転行列同士に限っては、$R(\alpha)R(\beta) = R(\beta)R(\alpha)$ が成り立ちます（$30^\circ + 60^\circ$ も $60^\circ + 30^\circ$ も同じだからです）。
- 注：3次元以上の回転では、回す軸の順番によって結果が変わるため、この可換性は失われます。






