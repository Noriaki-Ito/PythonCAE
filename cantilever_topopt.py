"""
SIMP法によるトポロジー最適化デモ

使い方
------
# 依存関係をインストール
pip install numpy scipy streamlit

# UI起動（Streamlit）
streamlit run cantilever_topopt.py

# UI起動後、ブラウザが立ち上がる
# スライドバーを動かしてパラメータ調整とトポロジー最適化、グラフ描画を実施

"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import streamlit as st

st.set_page_config(page_title="Topology Optimization", layout="wide")
st.title("Topology Optimization (SIMP Method, Cantilever)")

# 要素剛性行列の定義
def lk(E=1.0, nu=0.3):

    # 各要素の形状係数
    k = np.array([
        0.5 - nu/6,   0.125 + nu/8,   -0.25 - nu/12, -0.125 + 3*nu/8,
        -0.25 + nu/12, -0.125 - nu/8,  nu/6,          0.125 - 3*nu/8
    ])

    # 要素剛性行列
    KE = np.array([
        [ k[0],  k[1],  k[2],  k[3],  k[4],  k[5],  k[6],  k[7]],
        [ k[1],  k[0],  k[7],  k[6],  k[5],  k[4],  k[3],  k[2]],
        [ k[2],  k[7],  k[0],  k[5],  k[6],  k[3],  k[4],  k[1]],
        [ k[3],  k[6],  k[5],  k[0],  k[7],  k[2],  k[1],  k[4]],
        [ k[4],  k[5],  k[6],  k[7],  k[0],  k[1],  k[2],  k[3]],
        [ k[5],  k[4],  k[3],  k[2],  k[1],  k[0],  k[7],  k[6]],
        [ k[6],  k[3],  k[4],  k[1],  k[2],  k[7],  k[0],  k[5]],
        [ k[7],  k[2],  k[1],  k[4],  k[3],  k[6],  k[5],  k[0]]
    ])

    return (E/(1-nu**2)) * KE


# インデックス配列生成
def prepare_indices(nelx, nely):
    # 平面なので各節点は2自由度
    ndof = 2*(nelx+1)*(nely+1)
    KE = lk()  # 要素剛性行列
    # 矩形要素なので各要素は8自由度（2自由度の接点が4つ）
    edofMat = np.zeros((nelx*nely, 8), dtype=np.int64)

    # インデックス配列を生成する
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx*nely
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3], dtype=np.int64)
            edofMat[el, :] = edof

    # 疎行列組み立て用の行・列インデックス（iK, jK）
    iK = np.kron(edofMat, np.ones((8,1), dtype=int)).flatten().astype(np.int64)
    jK = np.kron(edofMat, np.ones((1,8), dtype=int)).flatten().astype(np.int64)
    return KE, edofMat, iK, jK, ndof


# 境界条件及び荷重を定義
def support_and_loads(nelx, nely):
    ndof = 2*(nelx+1)*(nely+1)  # 各接点は2自由度
    F = np.zeros(ndof)  # 外力
    top_node = (nely+1)*nelx + nely  # 荷重が掛かる側の接点
    F[2*top_node+1] = -1.0  # 下向きに荷重を掛ける
    fix_nodes = np.arange(nely+1)  # 固定されている側の接点
    fix_dofs = np.union1d(2*fix_nodes, 2*fix_nodes+1)  # 固定端自由度
    all_dofs = np.arange(ndof)  # 全自由度
    free_dofs = np.setdiff1d(all_dofs, fix_dofs).astype(np.int64)  # 自由端自由度
    return F, free_dofs


# 空間平滑化前の前処理
def build_filter(nelx, nely, rmin):
    from scipy.sparse import coo_matrix
    n = nelx*nely
    iH = []
    jH = []
    vH = []
    fr = int(np.floor(rmin))
    # 感度フィルタのスパース行列 H と正規化係数 Hs を構築
    for i in range(nelx):
        for j in range(nely):
            row = j + i*nely
            imin = max(i - fr, 0)
            imax = min(i + fr, nelx-1)
            jmin = max(j - fr, 0)
            jmax = min(j + fr, nely-1)
            for k in range(imin, imax+1):
                for l in range(jmin, jmax+1):
                    col = l + k*nely
                    fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
                    if fac > 0:
                        iH.append(row); jH.append(col); vH.append(fac)
    H = coo_matrix((vH, (iH, jH)), shape=(n, n)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).flatten()
    # 感度フィルタのスパース行列 H と正規化係数 Hs を返す
    return H, Hs


# 全体コンプライアンスの計算
def fe_compliance(nelx, nely, x, penal, KE, edofMat, iK, jK, F, free_dofs, Emin=1e-9, Emax=1.0):
    x_penal = (Emin + x**penal * (Emax - Emin)).flatten(order='F')  # 各要素の有効ヤング率
    sK = ((KE.flatten()[None].T).dot(x_penal[None])).flatten(order='F')  # 各要素の KE をスケールして疎行列の値配列を作る
    K = coo_matrix((sK, (iK, jK))).tocsr()  # 全体剛性行列
    K = (K + K.T) * 0.5
    U = np.zeros(K.shape[0])  # 全自由度の変位配列
    Uf = spsolve(K[free_dofs, :][:, free_dofs], F[free_dofs])  # 拘束を除いた自由端自由度
    U[free_dofs] = Uf
    Ue = U[edofMat]  # 要素ごとの節点変位
    ce = np.sum((Ue @ KE) * Ue, axis=1)  # 歪みエネルギー密度 (>0)

    # 有効ヤング率を計算
    x_p = x ** penal  # 密度にペナルティを与える
    Eeff = Emin + x_p * (Emax - Emin)  # 中間密度の弾性を減衰させる

    c = np.sum(Eeff.flatten(order='F') * ce)  # 全体コンプライアンス
    dc = -penal * (Emax - Emin) * (x ** (penal - 1)) * ce.reshape(nely, nelx, order='F')  # 感度
    return c, dc, ce.reshape(nely, nelx, order='F')


# 空間平滑化
def apply_filter(dc, x, H, Hs):
    dc_vec = dc.flatten(order='F')
    x_vec  = x.flatten(order='F')
    # 感度フィルタを適用
    num = H @ (x_vec * dc_vec)
    # 0割り回避のために0以上の値になるようにする
    den = np.maximum(1e-9, Hs * x_vec)
    return num / den  # 正規化


# Optimality Criteria法による体積制約下の密度更新
def oc_update(x, dc_filtered_vec, volfrac, nely, nelx, move=0.3, tol=1e-3):
    x_fix = x.copy()  # 探索に用いる点
    dc = dc_filtered_vec.reshape(nely, nelx, order='F')  # 感度

    l1, l2 = 0.0, 1e9
    while (l2 - l1) / (l1 + l2 + 1e-12) > tol:
        lmid = 0.5*(l1 + l2)
        t = np.sqrt(np.maximum(1e-16, -dc / (lmid + 1e-16)))  # 中身を必ず非負・非ゼロに
        x_cand = np.clip(x_fix * t, x_fix - move, x_fix + move)
        x_cand = np.clip(x_cand, 0.0, 1.0)
        if x_cand.mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid

    # 決まったλで一回だけ更新
    lam = 0.5*(l1 + l2)
    t = np.sqrt(np.maximum(1e-16, -dc / (lam + 1e-16)))
    x_new = np.clip(x_fix * t, x_fix - move, x_fix + move)
    x_new = np.clip(x_new, 0.0, 1.0)
    return x_new


# トポロジー最適化処理
def topopt(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=2.0, max_iter=25, change_tol=0.02):
    KE, edofMat, iK, jK, _ = prepare_indices(nelx, nely)
    F, free_dofs = support_and_loads(nelx, nely)
    H, Hs = build_filter(nelx, nely, rmin)
    x = volfrac * np.ones((nely, nelx))
    history = []
    move0 = 0.3  # 更新幅
    decay = 0.92  # 減衰率, 0.9〜0.98の範囲で調整
    for it in range(1, max_iter + 1):
        c, dc, ce_map = fe_compliance(nelx, nely, x, penal, KE, edofMat, iK, jK, F, free_dofs)
        dc_f = apply_filter(dc, x, H, Hs)  # 感度フィルタを適用（空間平滑化）
        move = max(0.02, move0 * (decay ** (it - 1)))
        x_new = oc_update(x, dc_f, volfrac, nely, nelx, move=move, tol=1e-3)
        change = np.max(np.abs((x_new - x)))
        x = x_new.copy()
        history.append((it, c, change))
        if change < change_tol:
            break
    return x, history, ce_map  # 最終密度, (iter, compliance, change) の組, 最終要素エネルギーを返す


# 同じパラメータに対してはキャッシュを用いる
@st.cache_data(show_spinner=False, max_entries=64)
def solve_cached(nelx, nely, volfrac, penal, rmin, iters, tol):
    x, history, ce = topopt(nelx, nely, volfrac, penal, rmin, iters, tol)
    return x, history, ce


# UI部分（スライドバー）
with st.sidebar:
    st.subheader("Parameters (auto recompute)")
    nelx = st.slider("nelx", 20, 150, 60, 10)
    nely = st.slider("nely", 10, 100, 20, 2)
    iters = st.slider("Max iterations", 10, 60, 25, 5)
    tol = st.slider("Convergence tol", 0.005, 0.05, 0.02, 0.005)
    volfrac = st.slider("Volume fraction", 0.2, 1.0, 0.5, 0.05)
    penal = st.slider("SIMP penalty", 1.0, 5.0, 3.0, 0.5)
    rmin = st.slider("Filter radius", 1.0, 5.0, 2.0, 0.5)

x, history, ce = solve_cached(nelx, nely, volfrac, penal, rmin, iters, tol)

col1, col2 = st.columns([2,1])

# トポロジー最適化の対象（今回は片持梁）を描画
with col1:
    # 白=空(0), 黒=材料(1) にしたいなら 1-x で反転
    img = (255 * np.clip(1.0 - x, 0.0, 1.0)).astype(np.uint8)  # L(=グレースケール)画像
    st.image(img, caption=f"Design (nelx={nelx}, nely={nely}, volfrac={volfrac:.2f})",
             width='stretch', clamp=True)

# 全体コンプライアンスと設計変化量のグラフを描画
with col2:
    import pandas as pd, altair as alt

    df = pd.DataFrame(history, columns=["iter", "compliance", "change"])

    base = alt.Chart(df).encode(x=alt.X("iter:Q", title="iteration"))

    left = base.mark_line().encode(
        y=alt.Y("compliance:Q", title="compliance")
    ).properties(width=500, height=220)

    right = base.mark_line(color="#d62728").encode(
        y=alt.Y("change:Q", title="change", axis=alt.Axis(titleColor="#d62728"))
    ).properties(width=500, height=220)

    chart = alt.layer(left, right).resolve_scale(y='independent')
    st.altair_chart(chart, width='content')
