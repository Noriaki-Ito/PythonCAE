"""
PINNによるバッテリーモジュール熱逆解析デモ

使い方
------
# 依存関係をインストール
pip install torch numpy matplotlib streamlit

# コンソールモードでの学習
python3 battery_pinn_inverse_jp.py --mode train --steps 5000

# UIを起動（Streamlit）
streamlit run battery_pinn_inverse.py -- --mode ui

# UI起動後にパラメータ調整してRun inverse estimationすると学習が始まる

"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 2次元拡散の陽解法での安定dtを計算
def stable_dt_2d(nx, ny, k, safety=0.4):
    # 格子幅: [0,1]を均等分割と仮定
    dx2 = (1.0/(nx-1))**2
    dy2 = (1.0/(ny-1))**2
    # 2D FTCS安定条件: dt <= (dx^2 * dy^2)/(2*k*(dx^2 + dy^2))
    dt_max = (dx2*dy2)/(2.0*k*(dx2+dy2))
    return safety*dt_max


# デバイス設定
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Fourier特徴 ----------
class FourierFeat(nn.Module):
    """
    x_norm, y_norm, t_norm (各[-1,1]) に対して、
    γ(x) = [sin(2π x B), cos(2π x B)] を連結して高周波成分を注入
    """
    def __init__(self, in_dim=3, m=8, scale=10.0):
        super().__init__()
        B = torch.randn(in_dim, m) * scale
        self.register_buffer("B", B)
        self.m = m
        self.in_dim = in_dim

    def forward(self, x):  # x: (N, in_dim) with values in [-1,1]
        proj = x @ self.B  # (N, m)
        return torch.cat([torch.sin(2 * torch.pi * proj), torch.cos(2 * torch.pi * proj)], dim=-1)  # (N, 2m)


# 温度場 u(x,y,t)を表現する多層パーセプトロンの定義(x,y,t)
class MLP(nn.Module):
    def __init__(self, din, dout, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(din, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, dout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


# パラメトリックな発熱源モデル（ガウス和）
# s(x,y) = sum_i a_i * exp(-||[x,y]-[cx_i,cy_i]||^2/(2*sig^2))
class GaussianSource(nn.Module):
    def __init__(self, M=2, init_amp=5.0, sigma=0.06, nonneg=True):
        super().__init__()
        self.M = M
        self.nonneg = nonneg
        self.sigma = nn.Parameter(torch.tensor(float(sigma)), requires_grad=False)
        cx = torch.rand(M, 1) * 0.8 + 0.1
        cy = torch.rand(M, 1) * 0.8 + 0.1
        a  = torch.ones(M, 1) * float(init_amp)
        self.centers = nn.Parameter(torch.cat([cx, cy], dim=1))  # (M,2)
        self.amps    = nn.Parameter(a)                           # (M,1)
    def forward(self, xy):
        N = xy.shape[0]
        diff = xy.view(N,1,2) - self.centers.view(1,self.M,2)
        r2 = (diff**2).sum(dim=-1, keepdim=True)  # (N,M,1)
        amps = self.amps
        if self.nonneg:
            amps = torch.nn.functional.softplus(amps)  # 非負化（安定のため）
        s_each = amps.view(1,self.M,1) * torch.exp(- r2 / (2*self.sigma**2))
        return s_each.sum(dim=1)


# 有限差分による順解析（合成データ生成）
# 実証のための単純な明示的スキーム（小さなdtに対して安定）
def forward_heat(nx, ny, nt, dt, k, h, ambient, source_fn):

    # 安定性チェック（陽解法）：大きすぎるdtは安全率付きで縮小
    dt_safe = stable_dt_2d(nx, ny, k, safety=0.4)
    print(f"[info] 安全dt目安: {dt_safe:.6f} （現行dt={dt}）")
    if dt > dt_safe:
        print(f"[info] 与えられたdt={dt} は陽解法の安定範囲を超えています。dtを {dt_safe:.6f} に自動調整します。")
        dt = dt_safe

    # 格子の生成
    xs = np.linspace(0,1,nx)
    ys = np.linspace(0,1,ny)
    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]

    # 温度場 (nx, ny)
    u = np.zeros((nx,ny), dtype=np.float32) + ambient  # start from ambient

    # グリッド上のソースを事前計算する（単純化のため時間に依存しない）
    xy = np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1).reshape(-1,2)
    xy_t = torch.from_numpy(xy).float()
    with torch.no_grad():
        s_grid = source_fn(xy_t).cpu().numpy().reshape(nx,ny)

    snaps = []
    alpha = k
    for t in range(nt):
        # ラプラス作用素
        uxx = (np.roll(u,-1,axis=0) - 2*u + np.roll(u,1,axis=0)) / (dx*dx)
        uyy = (np.roll(u,-1,axis=1) - 2*u + np.roll(u,1,axis=1)) / (dy*dy)
        rhs = alpha*(uxx+uyy) + s_grid
        u = u + dt*rhs

        # 数値の安全化（NaN/Infを回避）
        u = np.nan_to_num(u, nan=ambient, posinf=ambient, neginf=ambient)

        u[0,:]   = (k*u[1,:]/dx + h*ambient) / (k/dx + h)
        u[-1,:]  = (k*u[-2,:]/dx + h*ambient) / (k/dx + h)
        u[:,0]   = (k*u[:,1]/dy + h*ambient) / (k/dy + h)
        u[:,-1]  = (k*u[:,-2]/dy + h*ambient) / (k/dy + h)

        snaps.append(u.copy())
    snaps = np.stack(snaps, axis=0)  # (nt, nx, ny)
    return xs, ys, snaps, s_grid


# センサー位置の生成
def make_sensors(xs, ys, count, seed=0):
    rng = np.random.default_rng(seed)
    # avoid very boundary points
    xi = rng.integers(low=2, high=len(xs)-2, size=count)
    yi = rng.integers(low=2, high=len(ys)-2, size=count)
    return np.stack([xi, yi], axis=1)  # (count, 2)


# センサー位置のサンプリング
def sample_sensors(snaps, sensors, noise_std=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    nt = snaps.shape[0]
    meas = []
    for t in range(nt):
        vals = snaps[t, sensors[:,0], sensors[:,1]]
        if noise_std>0:
            vals = vals + rng.normal(0.0, noise_std, size=vals.shape)
        meas.append(vals)
    meas = np.stack(meas, axis=0)  # (nt, count)
    return meas


# ---------- 正規化ユーティリティ ----------
class Scaler:
    """ [0,1] → [-1,1] のアフィン変換（x_mu=0.5, x_sig=0.5） """
    def __init__(self, mu=0.5, sig=0.5):
        self.mu = mu
        self.sig = sig
    def forward(self, x):
        return (x - self.mu)/self.sig
    def __call__(self, x):
        return self.forward(x)


# ---------- PINNラッパ（スケーリング + Fourier特徴 + MLP） ----------
class PINNModel(nn.Module):
    """
    物理座標 (x,y,t) in [0,1]^3 を受け取り、
    1) [-1,1] に正規化 → 2) Fourier特徴を追加 → 3) MLP → 4) u を返す。
    ※ 出力uは温度スケーリング（u = u_mu + u_sig * û）を適用可能
    """
    def __init__(self, mlp_width=128, mlp_depth=6, fourier_m=8, fourier_scale=10.0, u_mu=25.0, u_sig=5.0):
        super().__init__()
        self.xs = Scaler(0.5, 0.5)  # x,y,t は [0,1]想定
        self.ys = Scaler(0.5, 0.5)
        self.ts = Scaler(0.5, 0.5)
        self.ff = FourierFeat(in_dim=3, m=fourier_m, scale=fourier_scale)
        din = 3 + 2*fourier_m
        self.mlp = MLP(din, 1, width=mlp_width, depth=mlp_depth)
        self.u_mu = nn.Parameter(torch.tensor(float(u_mu)), requires_grad=False)
        self.u_sig = nn.Parameter(torch.tensor(float(u_sig)), requires_grad=False)

    def forward(self, xyt_phys):
        xyt_norm = torch.stack([
            self.xs(xyt_phys[...,0]),
            self.ys(xyt_phys[...,1]),
            self.ts(xyt_phys[...,2])
        ], dim=-1)
        ff = self.ff(xyt_norm)
        feat = torch.cat([xyt_norm, ff], dim=-1)
        u_hat = self.mlp(feat)          # 正規化空間での出力
        u = self.u_mu + self.u_sig*u_hat # 物理スケールへ戻す
        return u


# PDE残差のサンプリング
def sample_collocation(n_col=20000):
    xy = np.random.rand(n_col,2).astype(np.float32)
    t  = np.random.rand(n_col,1).astype(np.float32)
    xyt = np.concatenate([xy,t], axis=1)
    return torch.from_numpy(xyt)


# 境界点のサンプリング
def sample_boundary(n_b=4000):
    t = np.random.rand(n_b,1).astype(np.float32)
    n_per = n_b//4
    y0 = np.random.rand(n_per,1).astype(np.float32)
    x0 = np.zeros_like(y0, dtype=np.float32)
    y1 = np.random.rand(n_per,1).astype(np.float32)
    x1 = np.ones_like(y1, dtype=np.float32)
    x2 = np.random.rand(n_per,1).astype(np.float32)
    y2 = np.zeros_like(x2, dtype=np.float32)
    x3 = np.random.rand(n_per,1).astype(np.float32)
    y3 = np.ones_like(x3, dtype=np.float32)
    bd = []
    bd += [np.concatenate([x0,y0,t[:n_per]], axis=1)]
    bd += [np.concatenate([x1,y1,t[:n_per]], axis=1)]
    bd += [np.concatenate([x2,y2,t[:n_per]], axis=1)]
    bd += [np.concatenate([x3,y3,t[:n_per]], axis=1)]
    bd = np.concatenate(bd, axis=0).astype(np.float32)
    return torch.from_numpy(bd)


# PINN用PDE残差計算
def pde_residual(u_net, src_model, xyt, k):
    xyt = xyt.requires_grad_(True)
    u = u_net(xyt)
    grads = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
    ux, uy, ut = grads[...,0:1], grads[...,1:2], grads[...,2:3]
    uxx = torch.autograd.grad(ux, xyt, torch.ones_like(ux), create_graph=True)[0][...,0:1]
    uyy = torch.autograd.grad(uy, xyt, torch.ones_like(uy), create_graph=True)[0][...,1:2]
    s  = src_model(xyt[...,:2])
    return ut - k*(uxx+uyy) - s


# 対流境界（Robin条件）の残差計算: k * grad(u).n + h*(u - u_inf) = 0
def boundary_residual(u_net, bd, k, h, u_inf):
    bd = bd.requires_grad_(True)
    u = u_net(bd)
    grads = torch.autograd.grad(u, bd, torch.ones_like(u), create_graph=True)[0]
    x, y = bd[...,0:1], bd[...,1:2]
    eps = 1e-6
    is_left   = (x<eps).float()
    is_right  = (x>1-eps).float()
    is_bottom = (y<eps).float()
    is_top    = (y>1-eps).float()
    un = (-grads[...,0:1])*is_left + (grads[...,0:1])*is_right + (-grads[...,1:2])*is_bottom + (grads[...,1:2])*is_top
    rob = k*un + h*(u - u_inf)
    return rob


# センサー観測データとの誤差損失
def sensor_loss(u_net, xs, ys, nt, sensors, meas, jitter_t=False, device=None):
    T = nt-1
    xyts = []
    ys_true = []
    for t in range(nt):
        tau = t / float(T if T>0 else 1)
        if jitter_t:
            tau = min(1.0, max(0.0, tau + np.random.uniform(-0.01,0.01)))
        for i,(ix,iy) in enumerate(sensors):
            xyts.append([xs[ix], ys[iy], tau])
            ys_true.append(meas[t,i])
    xyts = torch.tensor(xyts, dtype=torch.float32)
    ys_true = torch.tensor(ys_true, dtype=torch.float32).view(-1,1)
    # 入力テンソルをモデルと同じデバイスへ
    if device is None:
        device = next(u_net.parameters()).device
    xyts = xyts.to(device)
    ys_true = ys_true.to(device)
    u_pred = u_net(xyts)
    return ((u_pred - ys_true)**2).mean()


# 評価用の補助関数
def eval_grids(u_net, src_model, xs, ys, nt):
    # 評価フェーズ（推定結果をグリッド上で評価） u on grid across time and source on grid
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    xy = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    with torch.no_grad():
        device = next(u_net.parameters()).device
        xy_t = torch.from_numpy(xy).to(device)
        s_grid = src_model(xy_t).detach().cpu().numpy().reshape(nx, ny)

    T = nt-1
    snaps = []

    with torch.no_grad():
        device = next(u_net.parameters()).device
        for t in range(nt):
            tau = t/float(T if T>0 else 1)
            xyt = np.concatenate([xy, np.full((xy.shape[0],1), tau, dtype=np.float32)], axis=1)
            xyt_t = torch.from_numpy(xyt).to(device)
            u = u_net(xyt_t).detach().cpu().numpy().reshape(nx, ny)
            snaps.append(u)
    snaps = np.stack(snaps, axis=0)

    return snaps, s_grid


# 相対L2誤差
def relative_L2(a, b, eps=1e-8):
    num = np.linalg.norm(a-b)
    den = np.linalg.norm(a)+eps
    return float(num/den)


# 可視化用の補助関数
def plot_fields(xs, ys, s_true, s_est, u_true_t, u_est_t, save_dir, tag):
    fig, axes = plt.subplots(2,2, figsize=(8,6))
    vmin_s, vmax_s = s_true.min(), s_true.max()
    vmin_u, vmax_u = u_true_t.min(), u_true_t.max()
    im0 = axes[0,0].imshow(s_true.T, origin='lower',
                           extent=[xs[0],xs[-1],ys[0],ys[-1]], vmin=vmin_s, vmax=vmax_s, aspect='equal')
    axes[0,0].set_title("True source"); plt.colorbar(im0, ax=axes[0,0])
    im1 = axes[0,1].imshow(s_est.T, origin='lower',
                           extent=[xs[0],xs[-1],ys[0],ys[-1]], vmin=vmin_s, vmax=vmax_s, aspect='equal')
    axes[0,1].set_title("Estimated source"); plt.colorbar(im1, ax=axes[0,1])
    im2 = axes[1,0].imshow(u_true_t.T, origin='lower',
                           extent=[xs[0],xs[-1],ys[0],ys[-1]], vmin=vmin_u, vmax=vmax_u, aspect='equal')
    axes[1,0].set_title("True temperature")
    cbar2 = plt.colorbar(im2, ax=axes[1,0])
    cbar2.update_ticks()
    ot2 = cbar2.ax.yaxis.get_offset_text()
    ot2.set_x(3.)
    im3 = axes[1,1].imshow(u_est_t.T, origin='lower',
                           extent=[xs[0],xs[-1],ys[0],ys[-1]], vmin=vmin_u, vmax=vmax_u, aspect='equal')
    axes[1,1].set_title("Estimated temperature")
    cbar3 = plt.colorbar(im3, ax=axes[1,1])
    cbar3.update_ticks()
    ot3 = cbar3.ax.yaxis.get_offset_text()
    ot3.set_x(3.)
    fig.tight_layout()
    out = os.path.join(save_dir, f"fields_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# 逆解析全体（データ生成→学習→評価） (data synth -> train -> eval)
def run_inverse(args, device, ui_mode=False):
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1) 真の発熱源（合成データ生成用）を定義 (for data generation)
    true_src = GaussianSource(M=2, init_amp=8.0, sigma=0.05)  # used only for forward solver
    # Lock it (no grad)
    for p in true_src.parameters():
        p.requires_grad = False

    # 2) 順解析で真の温度データを生成 (truth) with chosen h
    xs, ys, snaps_true, s_true = forward_heat(
        nx=args.nx, ny=args.ny, nt=args.nt, dt=args.dt,
        k=args.k, h=args.h, ambient=args.ambient, source_fn=true_src
    )

    # 3) センサーを配置して測定値を取得 & sample noisy measurements
    sensors = make_sensors(xs, ys, args.sensor_count, seed=args.seed)
    meas = sample_sensors(snaps_true, sensors, noise_std=args.noise_std, rng=np.random.default_rng(args.seed))

    # 4) PINNモデル（温度場uネット＋発熱源モデル）を構築 (u-net) and learnable source model (initialized broadly)
    u_net = PINNModel(
        mlp_width=args.width, mlp_depth=args.depth,
        fourier_m=args.ff_m, fourier_scale=args.ff_scale,
        u_mu=args.ambient, u_sig=args.u_sig
    ).to(device)
    est_src = GaussianSource(M=args.M, init_amp=5.0, sigma=args.src_sigma).to(device)

    # 5) オプティマイザを設定（Adam）
    params = list(u_net.parameters()) + list(est_src.parameters())
    opt = optim.Adam(params, lr=args.lr)

    # 6) 学習ループ開始
    for step in range(1, args.steps+1):
        prog = step / args.steps
        w_sens = args.w_sens * prog
        w_pde = args.w_pde * (1 - prog) + 1.7 * prog
        w_bd = args.w_bd
        w_ic = args.w_ic
        opt.zero_grad()
        # PDE残差の選点
        xyt = sample_collocation(n_col=args.n_col).to(device)
        r_pde = pde_residual(u_net, est_src, xyt, k=args.k)
        loss_pde = (r_pde**2).mean()

        # 対流境界（Robin条件）の残差計算 (convective)
        bd = sample_boundary(n_b=args.n_b).to(device)
        r_bd = boundary_residual(u_net, bd, k=args.k, h=args.h,
                                 u_inf=torch.tensor([[args.ambient]], device=device))
        loss_bd = (r_bd**2).mean()

        # センサーに対してフィッティングを行う
        loss_sens = sensor_loss(u_net, xs, ys, args.nt, sensors, meas, jitter_t=True, device=device)
        xy0 = torch.rand(args.n_ic,2)
        t0  = torch.zeros(args.n_ic,1)
        xyt0 = torch.cat([xy0,t0], dim=1).to(device)
        u0 = u_net(xyt0)
        loss_ic = ((u0 - args.ambient)**2).mean()

        # ソース振幅のL2正規化
        reg_src = (est_src.amps**2).mean() * args.src_reg

        nx_tv, ny_tv = 60, 24
        X, Y = np.meshgrid(np.linspace(0, 1, nx_tv), np.linspace(0, 1, ny_tv), indexing='ij')
        xy_tv = torch.from_numpy(np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)).to(device)
        s_tv = est_src(xy_tv).view(nx_tv, ny_tv)

        dx = s_tv[1:, :] - s_tv[:-1, :]
        dy = s_tv[:, 1:] - s_tv[:, :-1]
        tv = (dx.abs().mean() + dy.abs().mean())

        loss = w_pde*loss_pde + w_bd*loss_bd + w_sens*loss_sens + w_ic*loss_ic + reg_src  + 1e-3 * tv
        loss.backward()
        # 勾配爆発の防止
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        # 損失に対するNaN検出
        if torch.isnan(loss) or torch.isinf(loss):
            print("[warn] 損失がNaN/Infになったため学習を早期停止します。学習率を下げる/重みを調整してください。")
            break
        opt.step()

        if (step % max(50, args.steps//10) == 0) or step==1:
            print(f"[{step:5d}] total={loss.item():.4e}  pde={loss_pde.item():.3e}  bd={loss_bd.item():.3e} "
                  f"sens={loss_sens.item():.3e}  ic={loss_ic.item():.3e}")

        # UIの応答性を高めるためにチェックポイントを設定する
        if ui_mode and (step % max(200, args.steps//3) == 0):
            torch.save({"u_net": u_net.state_dict(), "src": est_src.state_dict()},
                       os.path.join(args.save_dir, "ui_ckpt.pt"))

    # 7) 評価フェーズ（推定結果をグリッド上で評価）
    u_est_snaps, s_est = eval_grids(u_net, est_src, xs, ys, args.nt)
    rel_u = relative_L2(u_est_snaps[-1], snaps_true[-1])
    rel_s = relative_L2(s_est, s_true)
    print(f"Relative L2 (u @ last t): {rel_u:.3f},  (source): {rel_s:.3f}")

    # 8) 可視化出力
    fig_path = plot_fields(xs, ys, s_true, s_est, snaps_true[-1], u_est_snaps[-1], args.save_dir,
                           tag=f"h{args.h}_n{args.sensor_count}_noise{args.noise_std}")
    ckpt_path = os.path.join(args.save_dir, "ckpt.pt")
    torch.save({"u_net": u_net.state_dict(), "src": est_src.state_dict(), "args": vars(args)}, ckpt_path)

    # UIに渡すパラメータを纏める
    results = {
        "fig": fig_path,
        "ckpt": ckpt_path,
        "rel_u_last": rel_u,
        "rel_s": rel_s,
        "sensors": sensors.tolist(),
        "xs": xs.tolist(),
        "ys": ys.tolist(),
    }
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        import json
        json.dump(results, f, indent=2)

    return results


# StreamlitによるUI部分
def run_ui(default_args):
    import streamlit as st

    st.set_page_config(page_title="Battery Heat Inverse (PINN)", layout="wide")
    st.title("Battery Module Heat Inverse (PINN) — Demo")

    with st.sidebar:
        st.header("Scenario parameters")
        sensor_count = st.slider("Sensor count", 3, 16, default_args.sensor_count, step=1)
        noise_std = st.slider("Sensor noise (°C std)", 0.0, 1.0, float(default_args.noise_std), step=0.05)
        h = st.slider("Convective coef h", 0.0, 30.0, float(default_args.h), step=1.0)
        steps = st.slider("Training steps (UI)", 200, 5000, int(default_args.steps), step=100)
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=int(default_args.seed), step=1)
        st.caption("Tips: Increase steps for better estimates. This runs optimization live on your machine.")

        st.header("Model hyperparams")
        width = st.selectbox("MLP width", [32, 64,128,256], index=0)
        depth = st.selectbox("MLP depth", [4,6,8], index=0)
        M = st.selectbox("Source Gaussians M", [1,2,3], index=1)
        src_sigma = st.slider("Source sigma", 0.02, 0.12, float(default_args.src_sigma), step=0.01)
        src_reg = st.slider("Source L2 reg", 0.0, 1e-2, float(default_args.src_reg), step=1e-4, format="%.4f")

        st.header("Fourier Feature")
        ff_m = st.selectbox("Basis Num", [2,4,8,12,16], index=1)
        ff_scale = st.slider("Scale", 1.0, 20.0, float(default_args.ff_scale), step=1.0)

        st.header("Output Scale")
        u_sig = st.slider("Scale of u ( û → u )", 0.01, 1.0, float(default_args.u_sig), step=0.01)

        run_button = st.button("Run inverse estimation")

    st.markdown("""
    This demo generates synthetic temperature data from a hidden heat source
    and estimates the source and temperature field from sparse sensors using a PINN.
    **Loss** = w_pde·PDE + w_bd·BC(Robin) + w_sens·Sensors + w_ic·Initial + L2(source amps).
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ground Truth vs Estimate")
    with col2:
        st.subheader("Details")
        dt_safe = stable_dt_2d(default_args.nx, default_args.ny, default_args.k, safety=0.4)
        if default_args.dt > dt_safe:
            st.write(f"Grid: {default_args.nx}×{default_args.ny},  Time steps: {default_args.nt}, dt={dt_safe:.6f}")
        else:
            st.write(f"Grid: {default_args.nx}×{default_args.ny},  Time steps: {default_args.nt}, dt={default_args.dt:.6f}")
        st.write(f"k={default_args.k}, ambient={default_args.ambient}°C")
        st.code("k * du/dn + h * (u - u_inf) = 0   # convective (Robin) boundary", language="python")

    if run_button:
        # re-run with updated args (short inverse)
        run_args = argparse.Namespace(**vars(default_args))
        run_args.sensor_count = int(sensor_count)
        run_args.noise_std = float(noise_std)
        run_args.h = float(h)
        run_args.steps = int(steps)
        run_args.seed = int(seed)
        run_args.width = int(width)
        run_args.depth = int(depth)
        run_args.M = int(M)
        run_args.src_sigma = float(src_sigma)
        run_args.src_reg = float(src_reg)
        run_args.ff_m = int(ff_m)
        run_args.ff_scale = float(ff_scale)
        run_args.u_sig = float(u_sig)

        device = get_device()
        with st.spinner(f"Running inverse on {device}..."):
            results = run_inverse(run_args, device=device, ui_mode=True)

        # Show figure
        st.image(results["fig"], caption="Top: source (true vs est). Bottom: temperature @ last time (true vs est).")

        # Show metrics
        st.success(f"Relative L2 — u(last): {results['rel_u_last']:.3f},  source: {results['rel_s']:.3f}")
        st.json(results)


# コマンドライン実行設定
def build_argparser():
    p = argparse.ArgumentParser()
    # データ / 物理モデル
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--nt", type=int, default=60)
    p.add_argument("--k", type=float, default=0.5, help="熱拡散係数")
    p.add_argument("--h", type=float, default=10.0, help="対流係数")
    p.add_argument("--ambient", type=float, default=25.0)
    # センサー / ノイズ
    p.add_argument("--sensor_count", type=int, default=4)
    p.add_argument("--noise_std", type=float, default=0.15)
    # PINN
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--M", type=int, default=2, help="発熱源モデルのガウス分布の数")
    p.add_argument("--src_sigma", type=float, default=0.06)
    p.add_argument("--src_reg", type=float, default=1e-4)
    p.add_argument("--ff_m", type=int, default=4, help="Fourier基底数")
    # 訓練
    p.add_argument("--ff_scale", type=float, default=1.0, help="Fourierスケール")
    p.add_argument("--u_sig", type=float, default=0.01, help=" û → u のスケール")
    p.add_argument("--w_pde", type=float, default=1.5, help="PDE残差の重み")
    p.add_argument("--w_bd", type=float, default=0.1, help="境界条件（Robin）の重み")
    p.add_argument("--w_sens", type=float, default=30.0, help="センサー整合の重み")
    p.add_argument("--w_ic", type=float, default=0.1, help="初期条件の重み")
    p.add_argument("--steps", type=int, default=3500)
    p.add_argument("--n_col", type=int, default=20000)
    p.add_argument("--n_b", type=int, default=4000)
    p.add_argument("--n_ic", type=int, default=1024)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--seed", type=int, default=42)
    # I/O
    p.add_argument("--save_dir", type=str, default="outputs")
    # モード
    p.add_argument("--mode", type=str, choices=["train","ui"], default="train")
    return p

def main():
    args = build_argparser().parse_args()
    device = get_device()

    if args.mode == "train":
        run_inverse(args, device=device, ui_mode=False)
        print(f"Done. Outputs saved under: {args.save_dir}")
    elif args.mode == "ui":
        run_ui(args)

if __name__ == "__main__":
    main()