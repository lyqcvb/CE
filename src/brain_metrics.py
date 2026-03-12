"""
brain_metrics.py
================
Python 复现 MATLAB 脑网络指标计算流水线。

功能模块
--------
1. EEG 数据读取（EEGLAB .set/.fdt，MATLAB v7.3 HDF5 格式）
2. 功能连接矩阵计算（LOFC / HOFC / aHOFC）
3. 图论指标（全局效率、度中心性、介数中心性、接近中心性、聚类系数、模块度、参与系数）
4. Hub 节点识别（熵方法）
5. 富人俱乐部系数（归一化）
6. 靶向攻击模拟

对应 MATLAB 文件
---------------
- Remove_selfcon.m  →  remove_selfcon()
- untitled.m        →  所有指标计算函数
"""

from __future__ import annotations

import os
import numpy as np
import networkx as nx
from scipy.signal import hilbert
from scipy.stats import pearsonr
import h5py


# ─────────────────────────────────────────────────────────
# 1. EEG 数据读取
# ─────────────────────────────────────────────────────────

def read_eeg_set(set_path: str, n_samples: int = 25601) -> np.ndarray:
    """
    读取 EEGLAB .set/.fdt 文件对（MATLAB v7.3 / HDF5 格式）。

    Parameters
    ----------
    set_path : str
        .set 文件的完整路径。
    n_samples : int
        截取的时间点数（MATLAB 代码使用首 25601 点 = 256 Hz × 100 s + 1）。

    Returns
    -------
    eeg_data : ndarray, shape (n_channels, n_samples), float64
    """
    set_path = os.path.abspath(set_path)
    base_dir = os.path.dirname(set_path)

    with h5py.File(set_path, "r") as f:
        nbchan = int(np.array(f["nbchan"]).flatten()[0])
        pnts   = int(np.array(f["pnts"]).flatten()[0])
        # datfile 以 uint16 字符数组形式存储
        fdt_name = "".join(chr(c) for c in np.array(f["datfile"]).flatten())

    fdt_path = os.path.join(base_dir, fdt_name)
    raw = np.fromfile(fdt_path, dtype=np.float32)
    # MATLAB 以列优先（Fortran order）写入：每列=一个时间点的所有通道
    # 因此 binary 顺序为 (pnts × nbchan)，需转置为 (nbchan × pnts)
    eeg_data = raw.reshape(pnts, nbchan).T

    n_use = min(n_samples, pnts)
    return eeg_data[:, :n_use].astype(np.float64)


# ─────────────────────────────────────────────────────────
# 2. 功能连接矩阵
# ─────────────────────────────────────────────────────────

def compute_lofc(eeg_data: np.ndarray) -> np.ndarray:
    """
    低阶 FC（LOFC）：Hilbert 包络的 Pearson 相关（取绝对值）。

    对应 MATLAB untitled.m 第 18-34 行：
        analytic_signal = hilbert(signal);
        envelope = abs(analytic_signal);
        r = corr2(X, Y);  A(p,q) = r;
        A3 = abs(A);

    Parameters
    ----------
    eeg_data : ndarray, shape (n_ch, n_samples)

    Returns
    -------
    lofc : ndarray, shape (n_ch, n_ch)  — 包含对角线 1.0
    """
    envelopes = np.abs(hilbert(eeg_data, axis=1))
    lofc = np.abs(np.corrcoef(envelopes))
    return lofc


def _hofc_core(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    """
    [旧版简化算法，仅供参考/对比] 无 Fisher z 变换的 HOFC/aHOFC 计算。

    对每对 (i, j)：
    1. 取 in1 的第 i 列，删去对角线元素（第 i 行）→ 15 维向量 a
    2. 取 in2 的第 j 列，删去对角线元素（第 j 行）→ 15 维向量 b
    3. HOFC[i,j] = |pearsonr(b, a)|

    注意：此简化版本不含 Fisher z 变换和交叉元素删除，
    与 method 中的 Remove_selfcon.m 算法不一致。
    当前 compute_hofc() / compute_ahofc() 已切换为 remove_selfcon()。

    Parameters
    ----------
    in1, in2 : ndarray, shape (n, n)

    Returns
    -------
    FC : ndarray, shape (n, n)
    """
    n = in1.shape[0]
    FC = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            a = np.delete(in1[:, i], i)  # 删对角线元素
            b = np.delete(in2[:, j], j)  # 删对角线元素
            r, _ = pearsonr(b, a)
            FC[i, j] = r
            FC[j, i] = r
    return np.abs(FC)


def remove_selfcon(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    """
    高阶 FC 核心算法（直接移植 Remove_selfcon.m）。

    注意：此版本与 MATLAB untitled.m 中的 Remove_selfcon.m 逻辑一致，
    但生成的矩阵与 data/fc_matrices/ 参考 CSV 不同（参考 CSV 使用 _hofc_core）。

    对每对 (i, j)（j > i）：
    1. 取 in1 的第 i 列和 in2 的第 j 列。
    2. 删去自连接行（a 删第 i 行，b 删第 j 行）和交叉行
       （a 再删第 j-1 行，b 再删第 i 行）。
    3. Fisher z 变换：0.5 * (ln(1+x) - ln(1-x))。
    4. 计算 Pearson 相关，赋值并对称化，取绝对值。

    Parameters
    ----------
    in1, in2 : ndarray, shape (n, n)

    Returns
    -------
    HOFC : ndarray, shape (n, n)
    """
    n = in1.shape[0]
    HOFC = np.zeros((n, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            a = in1[:, i].copy()
            b = in2[:, j].copy()

            a = np.delete(a, i)
            b = np.delete(b, j)
            a = np.delete(a, j - 1)
            b = np.delete(b, i)

            a1 = 0.5 * (np.log(1.0 + a) - np.log(1.0 - a))
            b1 = 0.5 * (np.log(1.0 + b) - np.log(1.0 - b))

            r, _ = pearsonr(b1, a1)
            HOFC[i, j] = r
            HOFC[j, i] = r

    return np.abs(HOFC)


def compute_hofc(lofc: np.ndarray) -> np.ndarray:
    """
    高阶 FC（HOFC）：Remove_selfcon(LOFC, LOFC)。

    对应 MATLAB: [B3] = Remove_selfcon(A3, A3)
    含 Fisher z 变换和交叉元素删除，与 method 一致。
    """
    return remove_selfcon(lofc, lofc)


def compute_ahofc(lofc: np.ndarray, hofc: np.ndarray) -> np.ndarray:
    """
    非对称高阶 FC（aHOFC）：Remove_selfcon(LOFC, HOFC)。

    对应 MATLAB: Remove_selfcon_again(A3, B3)，其中 A3=LOFC, B3=HOFC。
    含 Fisher z 变换和交叉元素删除，与 method 一致。
    """
    return remove_selfcon(lofc, hofc)


def binarize(matrix: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    """
    二值化矩阵，并将对角线置零（去除自连接）。

    对应 MATLAB erzhihua(A, threshold)。
    """
    G = (matrix >= threshold).astype(np.float64)
    np.fill_diagonal(G, 0)
    return G


# ─────────────────────────────────────────────────────────
# 3. 图论指标
# ─────────────────────────────────────────────────────────

def _bfs_distance(G: np.ndarray) -> np.ndarray:
    """
    BFS 全源最短路径（二值图）。
    返回 D[i, j] = 最短路径长度，不可达为 inf。
    对应 BCT distance_bin.m。
    """
    n = G.shape[0]
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for src in range(n):
        visited = np.zeros(n, dtype=bool)
        visited[src] = True
        queue = [src]
        dist = 0
        while queue:
            nxt = []
            dist += 1
            for v in queue:
                for nb in np.where(G[v] > 0)[0]:
                    if not visited[nb]:
                        visited[nb] = True
                        D[src, nb] = dist
                        nxt.append(nb)
            queue = nxt
    return D


def global_efficiency_bin(G: np.ndarray) -> float:
    """
    二值图全局效率。
    E_glob = 1 / (N*(N-1)) * Σ_{i≠j} 1/d_ij
    对应 BCT efficiency_bin.m。
    """
    n = G.shape[0]
    if n < 2:
        return 0.0
    D = _bfs_distance(G)
    with np.errstate(divide="ignore"):
        inv_D = np.where(np.isinf(D), 0.0, 1.0 / np.where(D == 0, 1.0, D))
    np.fill_diagonal(inv_D, 0.0)
    return float(inv_D.sum() / (n * (n - 1)))


def _floyd_warshall_wei(W: np.ndarray) -> np.ndarray:
    """
    Floyd-Warshall 加权最短路径（edge length = 1/w）。
    对应 BCT distance_wei.m。
    """
    n = W.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(W > 0, 1.0 / W, np.inf)
    np.fill_diagonal(L, 0.0)
    D = L.copy()
    for k in range(n):
        D = np.minimum(D, D[:, k : k + 1] + D[k : k + 1, :])
    return D


def global_efficiency_wei(W: np.ndarray) -> float:
    """
    加权图全局效率（BCT efficiency_wei.m）。
    E = 1/(N*(N-1)) * Σ W_ij^(1/3) / d_ij
    """
    n = W.shape[0]
    if n < 2:
        return 0.0
    wmax = W.max()
    W_norm = W / wmax if wmax > 0 else W
    D = _floyd_warshall_wei(W_norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_D = np.where(D > 0, 1.0 / D, 0.0)
    np.fill_diagonal(inv_D, 0.0)
    W_cbrt = np.cbrt(W_norm)
    E_mat = W_cbrt * inv_D
    return float(E_mat.sum() / (n * (n - 1)))


def degree_centrality(G: np.ndarray, directed: bool = False) -> np.ndarray:
    """
    节点度中心性。
    - 无向（LOFC/HOFC）：sum(row)
    - 有向（aHOFC）：sum(row) + sum(col)

    对应 MATLAB: DC=sum(A3,2); DC2=sum(C3,2)+sum(C3,1)'.
    """
    if directed:
        return G.sum(axis=1) + G.sum(axis=0)
    return G.sum(axis=1)


def betweenness_bin(G: np.ndarray) -> np.ndarray:
    """
    二值图介数中心性（非归一化计数）。
    对应 BCT betweenness_bin.m。

    使用 networkx 计算，再乘以 2 使其与 BCT 有序对计数一致。
    """
    n = G.shape[0]
    Gx = nx.from_numpy_array(G)
    bc_dict = nx.betweenness_centrality(Gx, normalized=False, weight=None)
    # networkx 对无向图以无序对计数（除以2），BCT 以有序对计数 → ×2
    return np.array([bc_dict[i] * 2 for i in range(n)])


def betweenness_wei(W: np.ndarray) -> np.ndarray:
    """
    加权图介数中心性（非归一化计数）。
    对应 BCT betweenness_wei.m（edge length = 1/w）。
    """
    n = W.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(W > 0, 1.0 / W, 0.0)
    Gx = nx.from_numpy_array(L)
    bc_dict = nx.betweenness_centrality(Gx, normalized=False, weight="weight")
    return np.array([bc_dict[i] * 2 for i in range(n)])


def closeness_centrality(G: np.ndarray) -> np.ndarray:
    """
    接近中心性：CC[i] = 1 / Σ_j d(i,j)。
    对应 MATLAB: CCc = 1 ./ sum(D, 2)。
    """
    D = _bfs_distance(G)
    # 不可达设为 0（与 MATLAB 代码等价：sum 会很大 → 1/sum ≈ 0）
    D[np.isinf(D)] = 0.0
    row_sum = D.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cc = np.where(row_sum > 0, 1.0 / row_sum, 0.0)
    return cc


def clustering_coef_wu(W: np.ndarray) -> np.ndarray:
    """
    无向加权图聚类系数（BCT clustering_coef_wu.m）。
    C[i] = (W^(1/3) @ W^(1/3) @ W^(1/3))[i,i] / (K[i] * (K[i]-1))
    """
    W_sym = (W + W.T) / 2.0
    wmax = W_sym.max()
    W_norm = W_sym / wmax if wmax > 0 else W_sym
    K = (W_norm > 0).sum(axis=1).astype(float)
    W_cbrt = np.cbrt(W_norm)
    cyc3 = np.diag(W_cbrt @ W_cbrt @ W_cbrt)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(K >= 2, cyc3 / (K * (K - 1)), 0.0)
    return C


def clustering_coef_wd(W: np.ndarray) -> np.ndarray:
    """
    有向加权图聚类系数（BCT clustering_coef_wd.m）。
    """
    wmax = W.max()
    W_norm = W / wmax if wmax > 0 else W
    A = (W_norm > 0).astype(float)
    K = (A + A.T).sum(axis=1)  # in + out degree
    W_cbrt = np.cbrt(W_norm)
    S = W_cbrt + W_cbrt.T
    cyc3 = np.diag(S @ S @ S)
    denom = 2.0 * (K * (K - 1) - 2.0 * np.diag(A @ A))
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(denom > 0, cyc3 / denom, 0.0)
    return C


def modularity_louvain(
    G: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, float]:
    """
    Louvain 社区检测（近似 BCT modularity_louvain_dir.m）。

    Returns
    -------
    Ci : ndarray, shape (n,)  — 每个节点的社区标签（从 0 开始）
    Q  : float               — 模块度值
    """
    n = G.shape[0]
    Gx = nx.from_numpy_array(G)
    if Gx.number_of_edges() == 0:
        return np.zeros(n, dtype=int), 0.0

    communities = nx.community.louvain_communities(Gx, seed=seed)
    Ci = np.zeros(n, dtype=int)
    for idx, comm in enumerate(communities):
        for node in comm:
            Ci[node] = idx
    Q = nx.community.modularity(Gx, communities)
    return Ci, float(Q)


def participation_coef(G: np.ndarray, Ci: np.ndarray) -> np.ndarray:
    """
    参与系数（BCT participation_coef.m）。
    P[i] = 1 - Σ_m (k_im / k_i)^2
    """
    n = G.shape[0]
    K = G.sum(axis=1)
    P = np.zeros(n)
    for i in range(n):
        if K[i] == 0:
            continue
        sq_sum = 0.0
        for m in np.unique(Ci):
            mask = Ci == m
            k_im = G[i, mask].sum()
            sq_sum += (k_im / K[i]) ** 2
        P[i] = 1.0 - sq_sum
    return P


# ─────────────────────────────────────────────────────────
# 4. Hub 节点识别（CE 映射熵方法）
# ─────────────────────────────────────────────────────────

def compute_ce(G: np.ndarray, directed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    计算基于多属性决策（AHP-TOPSIS-Game Theory）结合映射熵的节点中心性熵（CE）。
    完全对应论文 2.6.2 节的数学推导步骤。
    
    Returns
    -------
    CE : ndarray, shape (n,)  — 最终的节点中心性熵
    S  : ndarray, shape (n,)  — 节点的综合中心性得分（中间产物）
    """
    n = G.shape[0]
    
    # --- Step 0: 提取基础中心性指标 ---
    DC = degree_centrality(G, directed=directed)
    BC = betweenness_bin(G)
    CC = closeness_centrality(G)
    
    # 构建决策矩阵 X 并进行 Min-Max 归一化（使数据无量纲化，范围 0-1）
    X = np.column_stack((DC, BC, CC))
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    range_val = X_max - X_min
    range_val[range_val == 0] = 1.0  # 防止除以零
    X_norm = (X - X_min) / range_val
    
    # --- Step 1: AHP 主观权重 ---
    w_s = np.array([0.077, 0.231, 0.692])
    
    # --- Step 2: TOPSIS 客观权重 ---
    # 寻找正理想解 (列最大值，归一化后即为 1.0)
    x_star = X_norm.max(axis=0)
    # 根据公式 (13) 最小化加权欧氏距离求解目标权重，解析解推导：w_o 与 1/D 成正比
    # D_j 为各指标到正理想解的距离平方和
    D = np.sum((x_star - X_norm)**2, axis=0)
    D[D == 0] = np.inf  # 防止完全一致的数据导致除零
    inv_D = 1.0 / D
    w_o = inv_D / np.sum(inv_D)
    
    # --- Step 3: Game Theory 组合权重 ---
    # 构建线性方程组求解 alpha1, alpha2
    A_mat = np.array([
        [np.dot(w_s, w_s), np.dot(w_s, w_o)],
        [np.dot(w_o, w_s), np.dot(w_o, w_o)]
    ])
    b_vec = np.array([np.dot(w_s, w_s), np.dot(w_o, w_o)])
    
    try:
        alphas = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError:
        alphas = np.array([0.5, 0.5]) # 奇异矩阵的降级方案
        
    # 归一化 alpha 并计算最终组合权重 w_c
    alphas = np.abs(alphas) / np.sum(np.abs(alphas))
    w_c = alphas[0] * w_s + alphas[1] * w_o
    w_c /= np.sum(w_c)
    
    # 计算节点综合中心性得分 S_i
    S = np.sum(X_norm * w_c, axis=1)
    
    # --- Step 4: 节点中心性结合映射熵 ---
    CE = np.zeros(n)
    # 获取邻接矩阵以确定局部邻居集合 Γ(i)
    A_adj = (G > 0) | (G.T > 0) if directed else (G > 0)
    np.fill_diagonal(A_adj, False)
    
    for i in range(n):
        neighbors = np.where(A_adj[i])[0]
        if len(neighbors) == 0 or S[i] == 0:
            CE[i] = 0.0
            continue
            
        S_neighbors = S[neighbors]
        sum_S_neighbors = np.sum(S_neighbors)
        
        if sum_S_neighbors == 0:
            CE[i] = 0.0
            continue
            
        # 计算映射概率 p_ik
        p_ik = S_neighbors / sum_S_neighbors
        valid_p = p_ik[p_ik > 0] # 仅保留大于0的概率参与对数计算
        
        # 计算局部香农信息熵
        H_i = -np.sum(valid_p * np.log2(valid_p))
        
        # 得出最终 CE 值
        CE[i] = S[i] * H_i

    return CE, S


def identify_hub_nodes(
    G: np.ndarray, directed: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    通过计算 CE 值来识别核心 Hub 节点。
    """
    CE, _ = compute_ce(G, directed=directed)
    threshold = CE.mean() + CE.std()
    is_hub = CE > threshold
    return is_hub, CE

# ─────────────────────────────────────────────────────────
# 5. 富人俱乐部系数（归一化）
# ─────────────────────────────────────────────────────────

def rich_club_bu(G: np.ndarray, klevel: int) -> np.ndarray:
    """
    无向二值图富人俱乐部系数（BCT rich_club_bu.m）。

    rc[k] = E_k / (N_k * (N_k - 1) / 2)，k = 1..klevel
    其中 N_k 为度 > k 的节点数，E_k 为其间的边数。
    """
    deg = G.sum(axis=1)
    rc = np.zeros(klevel)
    for k in range(1, klevel + 1):
        nodes_k = np.where(deg > k)[0]
        nk = len(nodes_k)
        if nk < 2:
            continue
        sub = G[np.ix_(nodes_k, nodes_k)]
        ek = sub.sum() / 2.0
        max_ek = nk * (nk - 1) / 2.0
        rc[k - 1] = ek / max_ek if max_ek > 0 else 0.0
    return rc


def rich_club_bd(G: np.ndarray, klevel: int) -> np.ndarray:
    """
    有向二值图富人俱乐部系数（BCT rich_club_bd.m）。

    rc[k] = E_k / (N_k * (N_k - 1))，使用总度（入度+出度）。
    """
    deg = G.sum(axis=1) + G.sum(axis=0)
    rc = np.zeros(klevel)
    for k in range(1, klevel + 1):
        nodes_k = np.where(deg > k)[0]
        nk = len(nodes_k)
        if nk < 2:
            continue
        sub = G[np.ix_(nodes_k, nodes_k)]
        ek = sub.sum()
        max_ek = nk * (nk - 1)
        rc[k - 1] = ek / max_ek if max_ek > 0 else 0.0
    return rc


def _randmio_und(G: np.ndarray, itr: int, rng: np.random.Generator) -> np.ndarray:
    """
    随机重连（无向，保持度序列）。
    移植自 BCT randmio_und_connected.m。
    """
    R = G.copy()
    rows, cols = np.where(np.triu(R, 1))
    edges = list(zip(rows.tolist(), cols.tolist()))
    ne = len(edges)
    if ne < 2:
        return R

    max_attempts = itr * ne
    for _ in range(max_attempts):
        idx = rng.integers(0, ne, size=2)
        if idx[0] == idx[1]:
            continue
        a, b = edges[idx[0]]
        c, d = edges[idx[1]]
        # 随机选互换方式
        if rng.random() > 0.5:
            i1, j1, i2, j2 = a, d, c, b
        else:
            i1, j1, i2, j2 = a, c, b, d

        if i1 == j1 or i2 == j2:
            continue
        # 归一化为上三角
        if i1 > j1:
            i1, j1 = j1, i1
        if i2 > j2:
            i2, j2 = j2, i2

        if R[i1, j1] == 0 and R[i2, j2] == 0:
            R[a, b] = R[b, a] = 0
            R[c, d] = R[d, c] = 0
            R[i1, j1] = R[j1, i1] = 1
            R[i2, j2] = R[j2, i2] = 1
            edges[idx[0]] = (i1, j1)
            edges[idx[1]] = (i2, j2)
    return R


def _randmio_dir(G: np.ndarray, itr: int, rng: np.random.Generator) -> np.ndarray:
    """
    随机重连（有向，保持度序列）。
    移植自 BCT randmio_dir_connected.m。
    """
    R = G.copy()
    rows, cols = np.where(R)
    edges = list(zip(rows.tolist(), cols.tolist()))
    ne = len(edges)
    if ne < 2:
        return R

    max_attempts = itr * ne
    for _ in range(max_attempts):
        e1 = int(rng.integers(0, ne))
        e2 = int(rng.integers(0, ne))
        if e1 == e2:
            continue
        a, b = edges[e1]
        c, d = edges[e2]
        if a == c or b == d or a == d or c == b:
            continue
        if R[a, d] == 0 and R[c, b] == 0:
            R[a, b] = 0; R[c, d] = 0
            R[a, d] = 1; R[c, b] = 1
            edges[e1] = (a, d)
            edges[e2] = (c, b)
    return R


def compute_rich_club(
    G_bin: np.ndarray,
    directed: bool = False,
    klevel: int = 16,
    n_rand: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    归一化富人俱乐部系数。

    对应 MATLAB untitled.m 第 170-215 行。

    Parameters
    ----------
    G_bin    : 二值邻接矩阵（已阈值化）
    directed : True → aHOFC（rich_club_bd）；False → LOFC/HOFC（rich_club_bu）
    klevel   : 最大度等级（LOFC/HOFC 为 16；aHOFC 为 32）
    n_rand   : 随机化次数（MATLAB 使用 1000）
    seed     : 随机种子（保证可重复）

    Returns
    -------
    rc_norm : ndarray, shape (klevel,)  — richclub / mean(random richclubs)
    """
    if directed:
        rc = rich_club_bd(G_bin, klevel)
    else:
        rc = rich_club_bu(G_bin, klevel)

    rng = np.random.default_rng(seed)
    ran_rc = np.zeros((n_rand, klevel))
    for ji in range(n_rand):
        if directed:
            R = _randmio_dir(G_bin, 2, rng)
            ran_rc[ji] = rich_club_bd(R, klevel)
        else:
            R = _randmio_und(G_bin, 2, rng)
            ran_rc[ji] = rich_club_bu(R, klevel)

    ran_mean = ran_rc.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rc_norm = np.where(ran_mean > 0, rc / ran_mean, 0.0)
    return rc_norm


# ─────────────────────────────────────────────────────────
# 6. 靶向攻击
# ─────────────────────────────────────────────────────────

def compute_target_attack(
    G_bin: np.ndarray,
    directed: bool = False,
    eg0: float | None = None,
) -> np.ndarray:
    """
    靶向攻击模拟：依据 CE (Node Centrality Entropy) 降序依次移除节点，记录剩余网络全局效率。
    对应论文 2.7 节方法描述。
    """
    n = G_bin.shape[0]
    if eg0 is None:
        eg0 = global_efficiency_bin(G_bin)

    # 计算 CE 值并生成降序索引
    CE, _ = compute_ce(G_bin, directed=directed)
    order = np.argsort(-CE)

    Eglob = np.zeros(n)
    working = G_bin.copy()
    for step, node in enumerate(order):
        # 移除节点（将其所有连边清零）
        working[:, node] = 0.0
        working[node, :] = 0.0
        e = global_efficiency_bin(working)
        Eglob[step] = e / eg0 if eg0 > 0 else 0.0
        
    return Eglob


# ─────────────────────────────────────────────────────────
# 7. 汇总指标（单被试，对应 all_scalar_metrics_summary.csv）
# ─────────────────────────────────────────────────────────

def compute_all_scalar_metrics(
    lofc: np.ndarray,
    hofc: np.ndarray,
    ahofc: np.ndarray,
    threshold: float = 0.35,
) -> dict:
    """
    计算标量汇总指标。
    修正了无向图的平均连接强度分母公式，确保与论文公式 (2) 一致。
    """
    result = {}
    n = lofc.shape[0]
    
    for fc_name, mat, is_directed in [
        ("LOFC",  lofc,  False),
        ("HOFC",  hofc,  False),
        ("aHOFC", ahofc, True),
    ]:
        G_bin = binarize(mat, threshold)
        
        # 严格按照公式区分有向和无向的平均连接强度计算
        if is_directed:
            MeanConn = float(G_bin.mean())
        else:
            # 仅取上三角矩阵计算总和，分母为 N(N-1)/2
            MeanConn = float(G_bin[np.triu_indices(n, 1)].sum() / (n * (n - 1) / 2))
            
        result[f"MeanConn_{fc_name}"] = MeanConn
        result[f"Eglob_{fc_name}"]    = global_efficiency_bin(G_bin)
        result[f"Cluster_{fc_name}"]  = float(
            clustering_coef_wd(G_bin).mean() if is_directed
            else clustering_coef_wu(G_bin).mean()
        )
    return result