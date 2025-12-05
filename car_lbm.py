"""
# 自動車周りの流れの簡易シミュレーション
# （格子ボルツマン法 LBM, D2Q9 モデル）
# ※あくまで「デモ用」の簡易コードです。
# 実務レベルの精度やロバスト性はありません。
# 実行方法
python3 car_lbm.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ----- 計算格子のサイズ -----
nx = 300  # x方向格子数
ny = 80   # y方向格子数

# ----- LBM のパラメータ -----
omega = 1.5 # 緩和パラメータ（衝突項の強さ） 1 < omega < 2 程度
u_in = 0.1  # 入口の流速（x方向）

# D2Q9 の重み
w = np.array([4/9] + [1/9]*4 + [1/36]*4)

# D2Q9 の速度ベクトル (cx, cy)
c = np.array([
    [ 0,  0],  # 0: 静止
    [ 1,  0],  # 1: 東
    [ 0,  1],  # 2: 北
    [-1,  0],  # 3: 西
    [ 0, -1],  # 4: 南
    [ 1,  1],  # 5: 北東
    [-1,  1],  # 6: 北西
    [-1, -1],  # 7: 南西
    [ 1, -1],  # 8: 南東
], dtype=int)

# 速度成分を別配列に分解
cx = c[:, 0]
cy = c[:, 1]

# ----- 自動車（障害物）の形状を作る -----
obstacle = np.zeros((ny, nx), dtype=bool)

# 簡易的に「車の側面」を長方形＋少し丸いフロントで表現
# 位置はドメイン中央付近
car_length = 60
car_height = 30
car_x_start = 80
car_y_center = ny // 2
car_y_start = car_y_center - car_height // 2

# ボディ部分（長方形）
obstacle[car_y_start:car_y_start+car_height,
         car_x_start:car_x_start+car_length] = True

# フロント部分を丸める（半円っぽく削る）
for y in range(car_y_start, car_y_start+car_height):
    # 車の前面のx座標
    front_x = car_x_start
    # 上下方向の相対位置
    dy = (y - car_y_center) / (car_height/2)
    if abs(dy) < 1.0:
        # 半円の式 x^2 + y^2 = 1 を使って少し削る
        dx = int((1 - dy**2)**0.5 * 10)  # 10は丸みの大きさ
        obstacle[y, front_x-dx:front_x] = False

# 上下壁（ノースリップ条件）を障害物として扱う
obstacle[0, :] = True
obstacle[-1, :] = True

# ----- 分布関数 f の初期化 -----
# f[方向, y, x]
f = np.zeros((9, ny, nx))

# 初期密度と速度
rho0 = 1.0
ux = np.full((ny, nx), u_in)  # 一様なx方向速度
uy = np.zeros((ny, nx))

# 障害物内部は速度ゼロ
ux[obstacle] = 0.0
uy[obstacle] = 0.0

def equilibrium(rho, ux, uy):
    """
    平衡分布関数 feq を計算する関数
    rho: 密度
    ux, uy: 速度ベクトル
    """
    # 速度の二乗
    u2 = ux**2 + uy**2
    feq = np.zeros((9, ny, nx))
    for i in range(9):
        cu = cx[i]*ux + cy[i]*uy  # 内積 c_i・u
        feq[i] = w[i] * rho * (
            1 + 3*cu + 4.5*cu**2 - 1.5*u2
        )
    return feq

# 初期分布は平衡分布に設定
rho = np.full((ny, nx), rho0)
f = equilibrium(rho, ux, uy)

# 図の準備
fig, ax = plt.subplots(figsize=(10, 3))
# 可視化は速度の大きさ |u| をカラー表示
speed = np.sqrt(ux**2 + uy**2)
im = ax.imshow(speed, origin='lower', cmap='jet', vmin=0, vmax=u_in*1.5)
ax.set_title("Car flow demo (LBM)")
ax.set_xlabel("x")
ax.set_ylabel("y")

# 障害物部分を輪郭で表示
# True のところの輪郭線を描く
obs_y, obs_x = np.where(obstacle)
ax.scatter(obs_x, obs_y, s=1, c='k')

plt.tight_layout()

def lbm_step():
    """
    LBM の1ステップ（衝突＋流し込み＋境界条件）を計算する
    """
    global f, rho, ux, uy

    # --- マクロ量の計算（密度・速度） ---
    rho = np.sum(f, axis=0)
    ux = np.sum(f * cx[:, None, None], axis=0) / rho
    uy = np.sum(f * cy[:, None, None], axis=0) / rho

    # 障害物セルは速度ゼロ
    ux[obstacle] = 0.0
    uy[obstacle] = 0.0

    # --- 衝突ステップ（BGKモデル） ---
    feq = equilibrium(rho, ux, uy)
    f = f + omega * (feq - f)

    # --- 流し込み（ストリーミング） ---
    for i in range(9):
        f[i] = np.roll(np.roll(f[i], cx[i], axis=1), cy[i], axis=0)

    # --- 境界条件 ---
    # 入口（左端）：一定速度 u_in を与える
    ux[:, 0] = u_in
    uy[:, 0] = 0.0
    rho[:, 0] = rho0  # シンプルに一定密度
    feq_in = equilibrium(rho[:, 0:1], ux[:, 0:1], uy[:, 0:1])
    for i in range(9):
        f[i, :, 0] = feq_in[i, :, 0]

    # 出口（右端）：ゼロ勾配境界（簡易的に隣接セルをコピー）
    for i in range(9):
        f[i, :, -1] = f[i, :, -2]

    # 障害物セルに対するバウンスバック（ノースリップ条件）
    # D2Q9 で反対方向のペア:
    # 0-0, 1-3, 2-4, 5-7, 6-8
    bounce_pairs = [(1, 3), (2, 4), (5, 7), (6, 8)]
    for i, j in bounce_pairs:
        fi = f[i]
        fj = f[j]
        # 障害物セルでは反対方向の分布関数を入れ替える
        tmp = fi[obstacle].copy()
        fi[obstacle] = fj[obstacle]
        fj[obstacle] = tmp

def update(frame):
    """
    アニメーション用：毎フレーム LBM を数ステップ回してから描画を更新
    """
    steps_per_frame = 5  # 1フレームあたりの反復回数（粗さを調整）
    for _ in range(steps_per_frame):
        lbm_step()

    speed = np.sqrt(ux**2 + uy**2)
    im.set_data(speed)
    # 速さがNaNになっていないかチェック
    if not np.isfinite(speed).all():
        print("NaN detected! simulation stopped")
        raise RuntimeError

    return [im]

# アニメーションの設定
anim = animation.FuncAnimation(
    fig, update, frames=300, interval=30, blit=True
)

if __name__ == "__main__":
    # 実行すると、ウィンドウに流れのアニメーションが表示される
    plt.show()