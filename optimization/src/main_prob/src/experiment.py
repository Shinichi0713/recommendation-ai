import numpy as np
from scipy.optimize import linprog

# --- 主問題 (Primal) を解く ---
# 目的関数の係数 (500x + 400y) -> 最小化にするためマイナス
c_primal = [-500, -400]
# 制約の左辺係数
A_primal = [[100, 200], [200, 100]]
# 制約の右辺（在庫）
b_primal = [2000, 2000]

res_primal = linprog(c_primal, A_ub=A_primal, b_ub=b_primal, bounds=(0, None))

# --- 双対問題 (Dual) を解く ---
# 目的関数の係数 (2000u + 2000v)
c_dual = [2000, 2000]
# 制約の左辺係数 (主問題の A を転置し、不等号の向きを調整)
# 100u + 200v >= 500  => -100u - 200v <= -500
# 200u + 100v >= 400  => -200u - 100v <= -400
A_dual = [[-100, -200], [-200, -100]]
b_dual = [-500, -400]

res_dual = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, bounds=(0, None))

# --- 結果の表示 ---
print("=== 主問題 (Primal) の結果 ===")
print(f"最適な生産量: 特製ミックス {res_primal.x[0]:.2f}杯, 濃厚ブレンド {res_primal.x[1]:.2f}杯")
print(f"最大売上: {-res_primal.fun:.0f}円")

print("\n=== 双対問題 (Dual) の結果 ===")
print(f"リンゴの価値(u): {res_dual.x[0]:.2f}円/g")
print(f"オレンジの価値(v): {res_dual.x[1]:.2f}円/g")
print(f"最小資源価値: {res_dual.x[0]*2000 + res_dual.x[1]*2000:.0f}円")