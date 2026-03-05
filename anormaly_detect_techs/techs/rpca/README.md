RPCA（Robust Principal Component Analysis）のように、反復計算（アルゴリズムのループ）を用いて背景成分  と異常（スパース）成分  を分離する手法では、 **「最適化が正しく進んでいるか」「いつ計算を止めるべきか」** を判断するために複数の指標をモニタリングします。

主に以下の4つの観点から指標をチェックするのが一般的です。


### 1. 収束の度合い（Convergence Error）

最も基本的で重要な指標です。前ステップの解と現在のステップの解がどれくらい変化したかを測定します。

* **再構成誤差 (Relative Error)**:
元の画像 $M$ と、現在の推定値 $L + S$ の差をモニタリングします。

$$\text{Error} = \frac{\|M - (L + S)\|_F}{\|M\|_F}$$

（$\| \cdot \|_F$ はフロベニウスノルム）
この値が設定した閾値（例：$10^{-7}$）を下回った時に「収束」とみなします。
この値が設定した閾値（例：）を下回った時に「収束」とみなします。
* **変数の変化量 (Variable Change)**:

$L$ や $S$ 自体が前ステップからどれだけ動いたかを見ます。これが急激に減少しなくなるポイントが、最適化の「安定期」です。

---

### 2. ランクとスパース性のトレードオフ

RPCAは「$L$ のランクを低く保ちつつ、$S$ をいかにスカスカ（スパース）にするか」というバランスを探る作業です。

* **$L$ の有効ランク (Effective Rank of $L$)**:

反復ごとに $L$ の特異値（Singular Values）を計算します。特異値が急激に減衰し、少数の主要な値だけに集中していく様子をモニタリングします。
* **$S$ の密度 (Density / Cardinality of $S$)**:

$S$ 成分のうち、非ゼロ（または微小値以上の値）を持つ画素の割合を記録します。
> **分析ポイント**: 最適化が進むにつれ、この密度が徐々に下がり、特定の異常箇所だけに値が集中していくのが理想的な挙動です。



---

### 3. 双対ギャップ (Duality Gap)

数学的に厳密な収束を保証したい場合に使用されます。

* **内容**: 主問題（$L$ と $S$ の最小化）と、その双対問題の解の差を計算します。
* **意味**: 理論上、最適解ではこの差が 0 になります。この値が小さくなるほど、アルゴリズムが「真の最適解」に近い場所にいることを示します。

---

### 4. 拡張ラグランジュ関数（ALM）のペナルティ項

RPCAの実装によく使われる **IALM (Inexact Augmented Lagrange Multiplier)** 拘束条件 $M - L - S = 0$ を守らせるための「罰金（ペナルティ）」を課します。

* **$\mu$ (Penalty Parameter) の推移**:

反復ごとに増加させていく重み係数 $\mu$ を記録します。この値が大きくなるにつれ、$M = L + S$ という制約がより厳格に守られるようになります。
* **残差の推移**:

$M - (L + S)$ のノルムが $\mu$ の増加に伴って滑らかに減少しているかを確認します。

---

### 実務でのモニタリング画面（イメージ）

もしモニタリングツールを作るなら、以下の4画面を並べるのがベストです。

| 指標 | グラフの動き | 異常のサイン |
| --- | --- | --- |
| **Rel. Error** | 右肩下がりに減衰 | 途中で横ばいになる（局所解へのトラップ） |
| **Rank of ** | 一定値で安定 | ランクがいつまでも高い（背景が複雑すぎ） |
| **Density of ** | 急降下して安定 | 密度が高いまま（ノイズを拾いすぎ） |
| **Objective Value** | 徐々に減少 | 激しく上下する（学習率/ステップ幅が大きすぎ） |

### 次のステップへのアドバイス

RPCAの最適化がうまくいかない（異常が見えない）場合、**「収束する前に止まっている」**のか、それとも**「パラメータ（）が不適切で、全てが  または  に吸い込まれている」**のかを、これらの指標で見極めることができます。

現在、反復回数は何回程度で設定されていますか？もし収束が遅い場合は、加速手法（ADM等）の導入を検討するのも手ですよ。


## 実装法

マルチスペクトルやハイパースペクトル画像のような多チャネルデータに対して、**RPCA（Robust Principal Component Analysis: ロバスト主成分分析）**は非常に強力な異常検知手法です。

### RPCAによる異常検知の理論

RPCAは、観測データ行列 $D$ を以下の2つの成分に分解します：

1. **低ランク成分 ($L$)**: 画像全体の統計的な「背景（共通する構造）」を表します。
2. **スパース成分 ($S$)**: 背景から外れた「異常（ノイズやターゲット）」を表します。

数学的には、主成分追跡（Principal Component Pursuit）と呼ばれる最適化問題を解きます：


$$\min_{L, S} \|L\|_* + \lambda \|S\|_1 \quad \text{subject to} \quad L + S = D$$


ここで $\|L\|_*$ は核ノルム（低ランク化）、$\|S\|_1$ は $L_1$ ノルム（スパース化）を促進します。

---

### PythonによるRPCA（ADMMアルゴリズム）の実装

行列分解には、収束が速く安定している**ADMM（交互方向乗数法）**を用います。

```python
import numpy as np
import matplotlib.pyplot as plt

class RPCADetector:
    def __init__(self, hsi_cube):
        """
        Args:
            hsi_cube (np.ndarray): (H, W, Bands)
        """
        self.h, self.w, self.bands = hsi_cube.shape
        # 3次元キューブを2次元行列 (Bands, Pixels) に変換
        # 行：波長、列：各画素 とすることで、全画素に共通するスペクトル構造を低ランク成分とする
        self.D = hsi_cube.reshape(-1, self.bands).T 
        
    def _soft_threshold(self, x, tau):
        """ソフトしきい値演算 (L1ノルムの近接作用素)"""
        return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

    def _svd_threshold(self, x, tau):
        """特異値しきい値演算 (核ノルムの近接作用素)"""
        U, S, Vh = np.linalg.svd(x, full_matrices=False)
        S_thresh = self._soft_threshold(S, tau)
        return (U * S_thresh) @ Vh

    def decompose(self, lamda=None, max_iter=100, tol=1e-7):
        """
        ADMMアルゴリズムによるロバスト主成分分析
        """
        n1, n2 = self.D.shape
        if lamda is None:
            lamda = 1 / np.sqrt(max(n1, n2))
            
        # 初期化
        L = np.zeros_like(self.D)
        S = np.zeros_like(self.D)
        Y = np.zeros_like(self.D) # ラグランジュ乗数
        mu = (n1 * n2) / (4.0 * np.linalg.norm(self.D, ord=1))
        
        print(f"Running RPCA Decomposition (lambda={lamda:.4f})...")
        
        for i in range(max_iter):
            # 1. L (低ランク成分) の更新
            L = self._svd_threshold(self.D - S + (1/mu) * Y, 1/mu)
            
            # 2. S (スパース成分) の更新
            S = self._soft_threshold(self.D - L + (1/mu) * Y, lamda/mu)
            
            # 3. Y (乗数) の更新と収束判定
            Z = self.D - L - S
            Y = Y + mu * Z
            
            err = np.linalg.norm(Z, 'fro') / np.linalg.norm(self.D, 'fro')
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}: error = {err:.2e}")
            
            if err < tol:
                break
                
        # 結果を元の3次元形状に復元
        # スコアマップは各画素のスパース成分のエネルギー（L2ノルム）として算出
        sparse_cube = S.T.reshape(self.h, self.w, self.bands)
        low_rank_cube = L.T.reshape(self.h, self.w, self.bands)
        
        # 異常スコア：全バンドにおけるスパース成分の強度の合計
        anomaly_score = np.linalg.norm(sparse_cube, axis=2)
        
        return low_rank_cube, sparse_cube, anomaly_score

# --- 使用例 ---
# detector = RPCADetector(hsi_cube)
# L_cube, S_cube, scores = detector.decompose(max_iter=50)

# # 可視化
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].imshow(detector.get_pseudo_rgb(hsi_cube)) # 以前のメソッド等を利用
# axes[0].set_title("Original RGB")
# im = axes[1].imshow(scores, cmap='hot')
# axes[1].set_title("RPCA Anomaly Score")
# plt.colorbar(im, ax=axes[1])
# plt.show()

```

---

### RPCAが多チャネルデータに有効な理由

* **スペクトル相関の利用**:
マルチスペクトルデータでは、各バンドは互いに強い相関を持っています。背景（土、草、コンクリート）はこの「相関の塊」として**低ランク成分**に吸い込まれます。
* **空間情報の制約なし**:
RXやCRDと異なり、RPCAは「窓サイズ」を定義する必要がありません。画像全体の統計から外れたピクセルを自動的に**スパース成分**として抽出するため、窓サイズの設定ミスによる見落としがありません。

---

### 実用上のアドバイス

1. **前処理としての空間フィルタ**:
RPCAをかける前に、これまで紹介した **Guided Filter** や **Median Filter** で空間的な微細ノイズを消しておくと、スパース成分に「本当の異常（ターゲット）」だけが残りやすくなります。
2. **パラメータ $\lambda$ の調整**:
* $\lambda$ を大きくする：より「稀な」ものしか異常とみなされなくなります。
* $\lambda$ を小さくする：少しの変化でもスパース成分（異常）として抽出されます。



これで、**空間フィルタリング、RX、CRD、RPCA** という異常検知の主要な武器がすべて揃いました。

データセットに対して、どの手法が最も「ターゲットを鮮明に、背景を静かに」映し出すか、比較実験を行ってみますか？例えば、**同じデータに対してこれら4つのスコアマップを並べて表示する比較スクリプト**を作成することも可能です。
