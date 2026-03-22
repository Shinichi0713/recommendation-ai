# 内積

ベクトル空間には和積演算があります。
ここまでは未だ代数的な演算をしているにすぎませんでした。
ですが、ベクトルの長さやベクトルの関係を表す内積や角度が導入されると幾何学的にとらえられるようになります。
本章ではベクトル空間に幾何学的なアプローチをとる手法について説明していきます。

## 計量

数学や線形代数の文脈における　**「計量（Metric / Metric structure）」**　とは、簡単に言うと　**「図形的な『長さ』や『角度』を測るためのルール」** のことです。

通常のベクトル空間は、単なる「矢印の集まり」であり、そのままでは「どのくらい長いか」や「どちらを向いているか」を厳密に定義できていません。そこに**内積**という仕組みを導入することで、初めて「測る」ことができるようになります。


### 1. なぜ「計量」が必要なのか？
例えば、2つのベクトル $\mathbf{x}$ と $\mathbf{y}$ があるとき、これらが「直交しているか」や「どちらが長いか」を判断するには、基準となる「ものさし」が必要です。

- **計量がない世界**: 伸び縮みするゴムの上のような世界。形はわかるが、正確な長さは測れない。
- **計量がある世界**: 硬い定規がある世界。長さ、角度、距離が確定する。



### 2. 線形代数における「計量」の正体
線形代数では、**計量行列（Metric Tensor / Gram Matrix）** というものを使って計量を表現します。

2つのベクトル $\mathbf{x}, \mathbf{y}$ の内積を次のように定義します。
$$\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T G \mathbf{y}$$
ここで、中心にある行列 $G$ が**計量行列**です。

- **標準的な計量**: $G$ が単位行列 $E$ のとき。私たちが普段使っている $x_1 y_1 + x_2 y_2 + \dots$ という普通の内積になります（ユークリッド計量）。
- **特殊な計量**: $G$ の値を変えると、「特定の方向だけ長さを 2 倍としてカウントする」といった、歪んだ空間の計量を定義できます。

### 3. 具体的な役割と便利さ

__① 長さ（ノルム）の定義__

計量（内積）が決まれば、ベクトルの長さ $\|\mathbf{x}\|$ は $\sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$ として定義されます。

__② 角度と直交の定義__

2つのベクトルのなす角 $\theta$ は、計量を用いて次のように計算できます。
$$\cos \theta = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|}$$
内積が $0$ になることを「直交」と呼びますが、これは選んだ「計量」に基づいた直交です。

__③ 物理学（相対性理論）での応用__

アインシュタインの相対性理論では、時間と空間を合わせた4次元空間の「計量」を考えます。場所によってこの計量が変化することで、「重力によって空間が歪む（長さや時間の進みが変わる）」ことを数式で表現しています。



### 4. 計量の役割

内積を考える上で、「計量（特に**計量行列**）」は、 **「空間の歪み」や「軸の測り方」を決定する司令塔**の役割を果たします。

私たちが普段使っている普通の内積（ドット積）は、実は数ある「計量」の中の特殊なケース（標準計量）に過ぎません。

__1. 内積を「一般化」する役割__
通常の内積は $\mathbf{x} \cdot \mathbf{y} = x_1 y_1 + x_2 y_2 + \dots$ ですが、これをより広いルールで定義するのが計量行列 $G$ です。

$$\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T G \mathbf{y}$$

この $G$ がどのような値を持つかによって、同じベクトル同士であっても **「内積の値」が変わります。** つまり、計量は **「計算のルールブック」** そのものです。

- **$G$ が単位行列 $E$ のとき**: 
  私たちが知っている「直交座標系」の世界です。
- **$G$ が対角行列（成分が $2, 1, 1$ など）のとき**: 
  特定の方向（この場合は $x$ 軸方向）だけ「重み」が違う世界です。
- **$G$ に非対角成分があるとき**: 
  座標軸が斜めに交わっている「斜交座標系」の世界です。

__2. 「直交」の意味を決める役割__

「直交」とは「内積が $0$」であることと定義されます。しかし、**何をもって $0$（直交）とするかは計量 $G$ 次第**です。

ある計量では直交して見えるベクトルも、別の計量（別のものさし）で見れば直交していないことがあります。
- **役割:** 計量は、その空間における**「垂直」という概念の基準**を提供します。



__3. 「長さ（距離）」を定義する役割__

ベクトルの長さ（ノルム）は、自分自身との内積の平方根 $\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$ で決まります。

もし計量行列 $G$ の成分が大きければ、同じ成分を持つベクトルでも「より長い」と判定されます。
- **役割:** 計量は、空間の各点や各方向における**「単位長さ」のスケール**を決定します。

__4. 具体的なエンジニアリング・物理での役割__

__① 統計学（マハラノビス距離）__

データの「ばらつき」を計量として取り入れます。
データの分散が大きい方向には「甘いものさし」を、分散が小さい方向には「厳しいものさし」を適用することで、**データの分布に即した正しい「近さ」** を測れるようになります。



__② 一般相対性理論__

宇宙空間の場所ごとに計量 $G$（計量テンソル）が変化すると考えます。
「計量が変化する = 長さの測り方が変わる」ことで、光が曲がったり時間が遅れたりする **「空間の歪み」** を数学的に記述します。


## 空間の内積と外積

空間における「内積」と「外積」は、どちらも2つのベクトルから新しい値を導き出す計算ですが、その**性質と役割は全く対照的**です。

一言でいうと、内積は **「重なり（スカラー）」** を求め、外積は **「回転と面積（ベクトル）」** を求めます。

### 1. 内積（Inner Product / Dot Product）

内積は、2つのベクトルを掛け合わせて**「一つの数値（スカラー）」**を取り出す計算です。

__数学的な定義__

2つのベクトル $\mathbf{a}, \mathbf{b}$ のなす角を $\theta$ とすると：
$$\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}| |\mathbf{b}| \cos \theta$$
成分表示（$n$ 次元）では、対応する成分同士を掛けて足します。
$$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n$$

__役割とイメージ__

- **影の長さ（射影）**: ベクトル $\mathbf{a}$ を $\mathbf{b}$ の方向に投影したときの「重なり具合」を表します。
- **直交判定**: 内積が $0$ ならば、2つのベクトルは垂直（$90^\circ$）です。
- **エネルギーや仕事**: 物理では「力 $\times$ 移動距離 $\times \cos\theta$」で仕事量を求める際に使われます。

__定理:__

2つのベクトル $\mathbf{a}$ を $\mathbf{b}$ の成分表示を $\mathbf{a} = (a_1, a_2, a_3)$ を $\mathbf{b} = (b_1, b_2, b_3)$ とするとき、

$$
(\mathbf{a}, \mathbf{b}) = a_1 b_1 + a_2 b_2 + a_3 b_3
$$

が成立する。

---

成分表示を用いた内積の公式の証明には、**「正規直交基底」** という概念と、**「内積の基本性質（線形性）」** を用います。

この証明を理解すると、なぜ「各成分を掛けて足すだけ」というシンプルな計算で内積（$|\mathbf{a}||\mathbf{b}|\cos\theta$）が求まるのか、その仕組みがスッキリと分かります。

__1. 準備：基本単位ベクトル（正規直交基底）__

3次元空間の座標軸方向を向いた、長さ $1$ のベクトルを $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ とします。
$$\mathbf{e}_1 = (1, 0, 0), \quad \mathbf{e}_2 = (0, 1, 0), \quad \mathbf{e}_3 = (0, 0, 1)$$

これらは互いに直交し、長さが $1$ であるため、内積には以下の性質があります。
- **自分自身との内積**: $(\mathbf{e}_i, \mathbf{e}_i) = |\mathbf{e}_i|^2 = 1$
- **異なるもの同士の内積**: $(\mathbf{e}_i, \mathbf{e}_j) = 0 \quad (i \neq j)$



__2. ベクトルの成分表示__

ベクトル $\mathbf{a}$ と $\mathbf{b}$ は、これらの基本単位ベクトルを使って次のように書き表せます。
$$\mathbf{a} = a_1 \mathbf{e}_1 + a_2 \mathbf{e}_2 + a_3 \mathbf{e}_3 = \sum_{i=1}^3 a_i \mathbf{e}_i$$
$$\mathbf{b} = b_1 \mathbf{e}_1 + b_2 \mathbf{e}_2 + b_3 \mathbf{e}_3 = \sum_{j=1}^3 b_j \mathbf{e}_j$$

<img src="image/18_inner_product/1774052676304.png" width="550" style="display: block; margin: 0 auto;">

__3. 証明：内積の展開__

内積の性質（線形性：分配法則ができること）を用いて、$ (\mathbf{a}, \mathbf{b}) $ を展開します。

$$
\begin{aligned}
(\mathbf{a}, \mathbf{b}) &= (a_1 \mathbf{e}_1 + a_2 \mathbf{e}_2 + a_3 \mathbf{e}_3, \ b_1 \mathbf{e}_1 + b_2 \mathbf{e}_2 + b_3 \mathbf{e}_3) \\
&= \sum_{i=1}^3 \sum_{j=1}^3 a_i b_j (\mathbf{e}_i, \mathbf{e}_j)
\end{aligned}
$$

この和を書き出すと 9 つの項が出てきますが、前述の「正規直交基底の性質」により、ほとんどが $0$ になります。

- **$i \neq j$ の項**: $(\mathbf{e}_i, \mathbf{e}_j) = 0$ なので、すべて消えます。
  - 例: $a_1 b_2 (\mathbf{e}_1, \mathbf{e}_2) = 0$
- **$i = j$ の項**: $(\mathbf{e}_i, \mathbf{e}_i) = 1$ なので、係数だけが残ります。
  - $a_1 b_1 (\mathbf{e}_1, \mathbf{e}_1) = a_1 b_1$
  - $a_2 b_2 (\mathbf{e}_2, \mathbf{e}_2) = a_2 b_2$
  - $a_3 b_3 (\mathbf{e}_3, \mathbf{e}_3) = a_3 b_3$

これらをすべて足し合わせると、次のようになります。

$$(\mathbf{a}, \mathbf{b}) = a_1 b_1 + a_2 b_2 + a_3 b_3$$

**（証明終）**


---


__例題:__ ベクトルの射影（影）の可視化目的

ベクトル $\mathbf{a}$ を $\mathbf{b}$ 方向に投影した「影」を描画する。
なす角 $\theta$ が鋭角・直角・鈍角のときに、内積の正負がどう変わるかを確認する。

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_dot_product(a, b):
    # 内積の計算
    dot_val = np.dot(a, b)
    
    # ベクトル b 方向への射影ベクトル (shadow) を計算
    # formula: (a・b / |b|^2) * b
    b_norm_sq = np.dot(b, b)
    proj_a_on_b = (dot_val / b_norm_sq) * b

    # 可視化の設定
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    
    # 元のベクトル a, b の描画
    plt.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, color='blue', label=f'Vector a {a}')
    plt.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='red', label=f'Vector b {b}')
    
    # 影（射影ベクトル）の描画
    plt.quiver(0, 0, proj_a_on_b[0], proj_a_on_b[1], angles='xy', scale_units='xy', scale=1, 
               color='gray', alpha=0.5, label='Projection (Shadow)')
    
    # aの先端からbのラインへ下ろす垂線
    plt.plot([a[0], proj_a_on_b[0]], [a[1], proj_a_on_b[1]], 'k--', lw=1)

    # グラフの調整
    limit = max(np.linalg.norm(a), np.linalg.norm(b)) + 1
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.title(f"Dot Product: {dot_val:.2f}\nShadow length: {np.linalg.norm(proj_a_on_b):.2f}")
    plt.gca().set_aspect('equal')
    plt.show()

# --- テストケース ---
# 1. 鋭角 (内積 > 0)
visualize_dot_product(np.array([3, 4]), np.array([5, 0]))

# 2. 直角 (内積 = 0)
# visualize_dot_product(np.array([0, 4]), np.array([5, 0]))

# 3. 鈍角 (内積 < 0)
# visualize_dot_product(np.array([-2, 3]), np.array([5, 0]))
```

__結果__

- 影の長さと内積:内積 $a \cdot b$ は、「$b$ の長さ $\times$ $a$ が $b$ の上に落とす影の長さ」に対応しています。もし $b$ が単位ベクトル（長さ 1）なら、内積そのものが影の長さになります。
- 正・負・ゼロの意味:
  - 正の値: 2つのベクトルが「同じ方向」を向いている（鋭角）。
  - ゼロ: 2つのベクトルが「完全に独立（直交）」している。影が一点（原点）に潰れてしまいます。
  - 負の値: 2つのベクトルが「逆方向」を向いている（鈍角）。影が $b$ と反対方向に伸びます。

<img src="image/18_inner_product/1774014843359.png" width="550" style="display: block; margin: 0 auto;">


### 2. 外積（Vector Product / Cross Product）

外積は、2つのベクトルから **「新しいベクトル」** を作り出す計算です。
※主に **3次元空間** で定義されます。

__数学的な定義__

結果として得られるベクトル $\mathbf{a} \times \mathbf{b}$ は、以下の性質を持ちます。
1.  **方向**: $\mathbf{a}$ と $\mathbf{b}$ の**両方に垂直**な方向（右ねじの法則）。
2.  **大きさ**: $\mathbf{a}$ と $\mathbf{b}$ が作る**平行四辺形の面積**に等しい。
$$|\mathbf{a} \times \mathbf{b}| = |\mathbf{a}| |\mathbf{b}| \sin \theta$$

__役割とイメージ__

- **回転の軸**: トルク（力回し）や角運動量など、回転運動を記述する際に、回転の「軸」として外積が使われます。
- **法線ベクトル**: 平面の「向き」を定義するために、平面上の2ベクトルから垂直なベクトルを作ります。
- **面積計算**: 3Dグラフィックスなどでポリゴンの面積を求める際に必須です。


3次元ベクトル空間における外積（ベクトル積）を、標準基底 $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ を用いて表現すると、その構造が非常にはっきりと見えてきます。

内積が「同じ基底同士」で生き残ったのに対し、外積は **「異なる基底同士」の組み合わせ** によって新しいベクトルを生み出します。

__1. 標準基底の外積ルール__

標準基底 $\mathbf{e}_1 = (1, 0, 0), \mathbf{e}_2 = (0, 1, 0), \mathbf{e}_3 = (0, 0, 1)$ の間には、以下の「ジャンケンのような」循環ルールがあります。

* **自分自身との外積**: $\mathbf{e}_i \times \mathbf{e}_i = \mathbf{0}$ （平行なもの同士の外積は $0$）
* **異なる基底との外積（正の順）**:
    * $\mathbf{e}_1 \times \mathbf{e}_2 = \mathbf{e}_3$
    * $\mathbf{e}_2 \times \mathbf{e}_3 = \mathbf{e}_1$
    * $\mathbf{e}_3 \times \mathbf{e}_1 = \mathbf{e}_2$
* **逆順の外積**: 順序を入れ替えると符号が反転します（反交換性）。
    * $\mathbf{e}_2 \times \mathbf{e}_1 = -\mathbf{e}_3$



__2. 成分表示による展開__

2つのベクトル $\mathbf{a} = a_1 \mathbf{e}_1 + a_2 \mathbf{e}_2 + a_3 \mathbf{e}_3$ と $\mathbf{b} = b_1 \mathbf{e}_1 + b_2 \mathbf{e}_2 + b_3 \mathbf{e}_3$ の外積 $\mathbf{a} \times \mathbf{b}$ を展開します。

$$
\begin{aligned}
\mathbf{a} \times \mathbf{b} &= (a_1 \mathbf{e}_1 + a_2 \mathbf{e}_2 + a_3 \mathbf{e}_3) \times (b_1 \mathbf{e}_1 + b_2 \mathbf{e}_2 + b_3 \mathbf{e}_3) \\
&= a_1 b_2 (\mathbf{e}_1 \times \mathbf{e}_2) + a_1 b_3 (\mathbf{e}_1 \times \mathbf{e}_3) + a_2 b_1 (\mathbf{e}_2 \times \mathbf{e}_1) + \dots
\end{aligned}
$$

自分自身との外積（$a_1 b_1 (\mathbf{e}_1 \times \mathbf{e}_1)$ など）はすべて消えるため、生き残る項を整理すると以下のようになります。

$$
\mathbf{a} \times \mathbf{b} = (a_2 b_3 - a_3 b_2) \mathbf{e}_1 + (a_3 b_1 - a_1 b_3) \mathbf{e}_2 + (a_1 b_2 - a_2 b_1) \mathbf{e}_3
$$

これが、私たちがよく知る**「外積の成分表示」**の正体です。

__3. 行列式（デタミナント）による表現__

標準基底を使うと、この複雑な式を**行列式の形式**で非常に美しく書き直すことができます。

$$
\mathbf{a} \times \mathbf{b} = \det \begin{pmatrix} \mathbf{e}_1 & \mathbf{e}_2 & \mathbf{e}_3 \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{pmatrix}
$$

この形式で 1 行目について展開（余因子展開）すると、各基底にかかる係数が自動的に求まります。

<img src="image/18_inner_product/1774059763768.png" width="600" style="display: block; margin: 0 auto;">


__定理:__

外積について成り立つ法則は以下の通りである。

(1) 反交換法則 

$$\mathbf{a} \times \mathbf{b} = - (\mathbf{b} \times \mathbf{a})$$

(2) 双線形性

- 分配法則: $\mathbf{a} \times (\mathbf{b} + \mathbf{c}) = \mathbf{a} \times \mathbf{b} + \mathbf{a} \times \mathbf{c}$

- 定数倍: $(k\mathbf{a}) \times \mathbf{b} = \mathbf{a} \times (k\mathbf{b}) = k(\mathbf{a} \times \mathbf{b})$

(3) 自己外積と平行なベクトルの外積

$$\mathbf{a} \times \mathbf{a} = \mathbf{0}$$

$$\mathbf{a} // \mathbf{b} \implies \mathbf{a} \times \mathbf{b} = \mathbf{0}$$

(4) ヤコビの恒等式

$$\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) + \mathbf{b} \times (\mathbf{c} \times \mathbf{a}) + \mathbf{c} \times (\mathbf{a} \times \mathbf{b}) = \mathbf{0}$$

---

外積の諸法則の証明には、**成分表示**を用いる方法と、**エディントンのレヴィ=チヴィタ記号（$\epsilon_{ijk}$）**を用いる方法がありますが、ここでは直感的に理解しやすい成分表示と外積の定義（幾何学的意味）を組み合わせて証明します。

__(1) 反交換法則の証明__

$$\mathbf{a} \times \mathbf{b} = - (\mathbf{b} \times \mathbf{a})$$

**証明：**
外積の成分表示の定義 $\mathbf{a} \times \mathbf{b} = (a_2 b_3 - a_3 b_2, a_3 b_1 - a_1 b_3, a_1 b_2 - a_2 b_1)$ を用います。
$\mathbf{b} \times \mathbf{a}$ を計算すると：
$$\mathbf{b} \times \mathbf{a} = (b_2 a_3 - b_3 a_2, b_3 a_1 - b_1 a_3, b_1 a_2 - b_2 a_1)$$
各成分を $\mathbf{a} \times \mathbf{b}$ と比較すると、すべての項で符号が反転していることがわかります。
例：第1成分 $b_2 a_3 - b_3 a_2 = -(a_2 b_3 - a_3 b_2)$
したがって、 $\mathbf{a} \times \mathbf{b} = - (\mathbf{b} \times \mathbf{a})$ が成り立ちます。



__(2) 双線形性の証明__

分配法則：$\mathbf{a} \times (\mathbf{b} + \mathbf{c}) = \mathbf{a} \times \mathbf{b} + \mathbf{a} \times \mathbf{c}$

**証明：**
第1成分に注目します。$(\mathbf{b} + \mathbf{c})$ の第 $i$ 成分は $(b_i + c_i)$ なので、
左辺の第1成分 $= a_2(b_3 + c_3) - a_3(b_2 + c_2)$
$= (a_2 b_3 - a_3 b_2) + (a_2 c_3 - a_3 c_2)$
これは $(\mathbf{a} \times \mathbf{b})$ の第1成分と $(\mathbf{a} \times \mathbf{c})$ の第1成分の和に等しいです。他の成分も同様です。

#### 定数倍：$(k\mathbf{a}) \times \mathbf{b} = k(\mathbf{a} \times \mathbf{b})$

**証明：**
第1成分は $(k a_2)b_3 - (k a_3)b_2 = k(a_2 b_3 - a_3 b_2)$ となり、明らかに成立します。

__(3) 自己外積と平行なベクトルの外積の証明__

$$\mathbf{a} \times \mathbf{a} = \mathbf{0}$$

**証明：**
外積の大きさの定義 $|\mathbf{a} \times \mathbf{b}| = |\mathbf{a}||\mathbf{b}|\sin\theta$ を用います。
自分自身とのなす角 $\theta$ は $0^\circ$ です。 $\sin 0^\circ = 0$ なので、大きさは $0$、すなわち零ベクトル $\mathbf{0}$ となります。
（成分表示でも $a_i a_j - a_j a_i = 0$ となり、簡単に示せます）

平行な場合も $\theta = 0^\circ$ または $180^\circ$ であり、 $\sin\theta = 0$ となるため同様に $\mathbf{0}$ です。



__(4) ヤコビの恒等式の証明__

$$\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) + \mathbf{b} \times (\mathbf{c} \times \mathbf{a}) + \mathbf{c} \times (\mathbf{a} \times \mathbf{b}) = \mathbf{0}$$

**証明：**
この証明には、非常に便利な**ベクトル三重積の公式（bac-cab公式）**を用います。
$$\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) = \mathbf{b}(\mathbf{a} \cdot \mathbf{c}) - \mathbf{c}(\mathbf{a} \cdot \mathbf{b})$$

これを使って、左辺の各項を展開します。
1. $\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) = \mathbf{b}(\mathbf{a} \cdot \mathbf{c}) - \mathbf{c}(\mathbf{a} \cdot \mathbf{b})$
2. $\mathbf{b} \times (\mathbf{c} \times \mathbf{a}) = \mathbf{c}(\mathbf{b} \cdot \mathbf{a}) - \mathbf{a}(\mathbf{b} \cdot \mathbf{c})$
3. $\mathbf{c} \times (\mathbf{a} \times \mathbf{b}) = \mathbf{a}(\mathbf{c} \cdot \mathbf{b}) - \mathbf{b}(\mathbf{c} \cdot \mathbf{a})$

これらをすべて足すと：
$(\mathbf{b} \cdot \mathbf{a})$ と $(\mathbf{a} \cdot \mathbf{b})$ は内積の交換法則により等しいので、 $\mathbf{c}(\mathbf{b} \cdot \mathbf{a}) - \mathbf{c}(\mathbf{a} \cdot \mathbf{b}) = 0$ となります。
同様に $\mathbf{a}$ の項、 $\mathbf{b}$ の項もすべて打ち消し合い、結果は $\mathbf{0}$ となります。


---

## 内積空間

**内積空間（Inner Product Space）** とは、一言で言えば **「幾何学的な『長さ』や『角度』を厳密に計算できるルールが備わったベクトル空間」** のことです。

単なる「ベクトル空間」は、足し算と定数倍ができるだけの「平坦な集合」に過ぎませんが、そこに「内積」という道具をインストールすることで、空間に豊かな図形的構造が生まれます。


### 1. 内積空間の「3種の神器」
内積が定義されることで、私たちは以下の3つを数学的に扱えるようになります。

- **長さ（ノルム）**: ベクトル自体の大きさを測る。
- **角度**: 2つのベクトルがどのくらい離れているかを測る。
- **直交**: 2つのベクトルが「垂直（角度 $90^\circ$）」であることを定義する。




### 2. 内積と認められるための「4つのルール」
どんな計算でも「内積」と呼べるわけではありません。関数 $(\mathbf{x}, \mathbf{y})$ が内積であるためには、以下の4つの性質（公理）を完璧に満たす必要があります。

1. **正定値性**: 自分の内積 $(\mathbf{x}, \mathbf{x})$ は必ず $0$ 以上で、自分が $0$ のときだけ $0$ になる。
2. **対称性（共役対称性）**: 順番を入れ替えても値が変わらない（複素数の場合は共役を取る）。
3. **加法性**: $(\mathbf{x} + \mathbf{y}, \mathbf{z}) = (\mathbf{x}, \mathbf{z}) + (\mathbf{y}, \mathbf{z})$
4. **斉次性（スカラー倍）**: $(k\mathbf{x}, \mathbf{y}) = k(\mathbf{x}, \mathbf{y})$


### 3. なぜ「空間」と呼ぶのか？
「内積」というルールが空間全体に行き渡っているからです。これにより、空間内のあらゆる場所で「最短距離」を求めたり、複雑な波やデータを「直交する成分」に分解したりすることが可能になります。

__具体的な例__

- **ユークリッド空間**: 私たちが日常で使う、成分同士を掛けて足す普通の内積空間。
- **関数空間($L^2$ 空間)** : 「関数」をベクトルとみなし、積の積分 $\int f(x)g(x)dx$ を内積とする空間。フーリエ解析などの基礎になります。


### 4. 応用例

内積空間は、単なる数学の理論にとどまらず、現代のIT・AI技術、物理学、信号処理など、幅広い分野で「基盤」として用いられています。

内積空間の最大の特徴は、**「距離（似ている度合い）」** と **「直交（無相関・独立）」** を計算できることです。この性質がどのように応用されているか、具体的な例を挙げます。

__1. AI・機械学習：レコメンドシステムと検索__

現代のAIにおいて、内積空間は「データの意味的な近さ」を測るステージです。

- **コサイン類似度**: 文章や画像を数千次元のベクトルに変換し、その「内積」を計算することで、ユーザーの好みに近い商品やニュースを探し出します。
- **重みの学習**: ニューラルネットワークの各層で行われる計算（$\mathbf{w}^T \mathbf{x}$）は、入力データと学習した重みベクトルの内積そのものです。



__2. 信号処理・通信：フーリエ変換とデータ圧縮__

「関数」をベクトルとみなす内積空間（$L^2$空間）の考え方が、デジタル機器を支えています。

- **周波数分解**: 複雑な音や画像の波を、単純なサイン波・コサイン波に分解します。これは、元の信号ベクトルを「直交する基底（各周波数）」に投影し、内積によって各成分の強さを取り出す作業です。
- **ノイズ除去**: ノイズ成分と信号成分が「直交」していれば、内積を使ってノイズだけをきれいに取り除くことができます。



__3. 量子コンピュータと量子力学：ヒルベルト空間__

量子力学の世界では、物理状態は「ヒルベルト空間」と呼ばれる、複素数を扱う特別な内積空間上のベクトルとして記述されます。

- **観測と確率**: 量子ビットの状態を観測して、ある結果が得られる「確率」を計算する際に、状態ベクトル同士の内積が使われます。
- **量子ゲート**: 量子計算の操作（ゲート操作）は、内積空間の「長さ（確率の合計=1）」を保ったままベクトルを回転させる操作（ユニタリ変換）に対応します。



__4. 統計学・データ分析：主成分分析 (PCA)__

大量のデータから重要な情報だけを抜き出す際にも内積空間が活躍します。

- **次元削減**: データのばらつき（分散）が最大になるような「新しい軸」を探します。この軸は、データの共分散行列の「固有ベクトル」として求まり、元のデータをその軸へ「内積（投影）」することで、情報を保持したままデータを圧縮します。

__5. 制御工学・ロボティクス：最短経路と最適制御__

ロボットアームの動きや自動運転の進路決定において、「エネルギーを最小にする」といった最適化問題を解く際に用いられます。

- **最小二乗法**: 観測データに最も適合する数式を探す手法です。これは幾何学的には「データが成す部分空間へ、垂線を下ろす（直交射影する）」操作であり、内積によって計算されます。



