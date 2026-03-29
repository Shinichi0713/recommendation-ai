import numpy as np
import matplotlib.pyplot as plt

# 複素2次元ベクトル空間でのユニタリ行列の例
# 例: 対角成分が e^{iθ}, e^{-iθ} のユニタリ行列
theta = np.pi / 3
U = np.array([
    [np.exp(1j * theta), 0],
    [0, np.exp(-1j * theta)]
], dtype=complex)

print("U:\n", U)
print("U^† U:\n", U.conj().T @ U)  # ユニタリ性の確認

# 複素ベクトル（例: 第1成分が 1+0i, 第2成分が 0+0i）
z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

# ユニタリ変換後のベクトル
z_transformed = U @ z

print("元のベクトル z:", z)
print("変換後 Uz:", z_transformed)
print("ノルム |z|:", np.linalg.norm(z))
print("ノルム |Uz|:", np.linalg.norm(z_transformed))

# 可視化: 第1成分の実部・虚部を2次元平面にプロット
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# 元のベクトルの第1成分 (Re z1, Im z1)
z1_orig = z[0]
ax.quiver(0, 0, z1_orig.real, z1_orig.imag,
          angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.01, label='z[0] (original)')

# 変換後のベクトルの第1成分 (Re (Uz)1, Im (Uz)1)
z1_trans = z_transformed[0]
ax.quiver(0, 0, z1_trans.real, z1_trans.imag,
          angles='xy', scale_units='xy', scale=1,
          color='red', width=0.01, label='(Uz)[0] (transformed)')

ax.legend()
ax.set_title(f"ユニタリ行列による複素ベクトルの変換\n(第1成分の実部・虚部を表示, θ = {theta:.2f} rad)")
plt.show()