
## 行列式

**行列式（Determinant）** とは、一言で言えば **「その行列が、空間を何倍に膨らませたか（あるいは押しつぶしたか）を表す拡大率」** のことです。

行列が「空間を動かす指示書」なら、行列式はその結果として **「どれくらい面積や体積が変化したか」** というスコアのようなものです。通常、行列  に対して  や  と表記されます。

### 1. 幾何学的な意味：面積と体積の拡大率

2次元空間で考えると、行列式の意味が非常にクリアになります。

* **2次元の場合：** 標準基底が作る「面積 1 の正方形」が、行列 $A$ によって変形された後の「平行四辺形の面積」が行列式です。
* **3次元の場合：** 「体積 1 の立方体」が変形された後の「平行六面体の体積」が行列式です。

### 2. 行列式の「値」が教えてくれること

行列式の値を見るだけで、その線形写像が空間に何をしたのかがわかります。

| 行列式の値 | 空間に起きたこと | 数学的な意味 |
| --- | --- | --- |
| $\det(A) = 2$ | 面積が 2 倍に膨らんだ。 | 通常の変換。 |
| $\det(A) = 1$ | 面積は変わらない（回転など）。 | **ユニタリ行列**などがこれに当たる。 |
| $\det(A) = -1$ | 面積は同じだが、**「裏返し」**になった。 | 鏡映（リフレクション）など。 |
| $\det(A) = 0$ | 空間が**ペシャンコに潰れた**。 | **逆行列が存在しない**（正則でない）。 |


### 3. なぜ行列式が重要なのか？（実用的な視点）

__① 逆行列の存在チェック__

行列式が $0$ ということは、2次元の平面を1次元の「線」に、あるいは0次元の「点」に押しつぶしてしまったことを意味します。潰れたものを元の広さに戻す（逆再生する）ことは不可能なため、 **「$\det(A) = 0 \iff$ 逆行列が存在しない」** という非常に重要な判定基準になります。

__② 連立方程式の解__

連立方程式（Systems of linear equations）を解く際、行列式が $0$ でなければ、解が一意に定まります。
これを利用したのが後ほど扱う「クラメルの公式」です。

__③ 多変数関数の積分（ヤコビアン）__

積分で変数を変換する際（例えば直交座標から極座標へ）、微小な領域がどれくらい歪んだかを補正する必要があります。この補正係数として「ヤコビ行列式（ヤコビアン）」が登場します。

### 4. 計算のイメージ（2次の場合）

$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$

このとき、$\det(A) = ad - bc$ と計算されます。これは、ベクトル $\begin{pmatrix} a \\ c \end{pmatrix}$ と $\begin{pmatrix} b \\ d \end{pmatrix}$ が作る平行四辺形の面積を求めていることに他なりません。


__例題:__ ベクトルと逆行列

行列式が $0$ に近づくにつれて、 **「2次元の面積を持っていた世界が、1次元の『線』へとペシャンコに押しつぶされていく」** 様子を可視化します。

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_collapsing_space(steps=5):
    # 1. 元の基底ベクトル
    e1 = np.array([1, 0])
    e2_start = np.array([0, 1])  # 最初は直交（行列式=1）
    e2_end = np.array([1, 0])    # 最後はe1と同じ（行列式=0、空間が潰れる）

    # グリッド描画用のデータ
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack([X.flatten(), Y.flatten()])

    fig, axes = plt.subplots(1, steps, figsize=(20, 4))
    
    for i, t in enumerate(np.linspace(0, 1, steps)):
        # e2 を徐々に e1 に近づける
        e2_current = (1 - t) * e2_start + t * e2_end
        
        # 写像行列 A = [e1, e2_current]
        A = np.column_stack([e1, e2_current])
        det = np.linalg.det(A)
        
        # 空間の変形
        t_pts = A @ pts
        TX = t_pts[0, :].reshape(X.shape)
        TY = t_pts[1, :].reshape(Y.shape)
        
        # 描画
        ax = axes[i]
        # 変形後のグリッド
        for j in range(len(x)):
            ax.plot(TX[j, :], TY[j, :], color='gray', alpha=0.3)
            ax.plot(TX[:, j], TY[:, j], color='gray', alpha=0.3)
        
        # ベクトルの描画（複数のベクトルが潰れていく様子）
        ax.quiver(0, 0, A[0,0], A[1,0], color='red', angles='xy', scale_units='xy', scale=1, label='v1')
        ax.quiver(0, 0, A[0,1], A[1,1], color='blue', angles='xy', scale_units='xy', scale=1, label='v2')
        
        # 設定
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f"Det = {det:.2f}")
        if i == 0: ax.legend()

    plt.suptitle("Space collapsing from 2D to 1D as Determinant approaches 0", fontsize=16)
    plt.tight_layout()
    plt.show()

# シミュレーション実行
simulate_collapsing_space(steps=5)
```


