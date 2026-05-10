import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置路径 =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(BASE_DIR, "result")
TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Global", "Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv")

OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "cascading_global_backbone.png")

# ================= 核心参数 =================
YEAR = 2021
MAX_ATTACK_NODES = 30
# 🚀 方案 B 的核心：设定贸易额阈值（单位：1000 USD）
# 我们过滤掉低于 100,000 USD (即 100 个单位) 的小额贸易，以显露网络骨架
TRADE_THRESHOLD = 1000
GAMMAS = [0.2, 0.4, 0.6, 0.8, 1.0]

def build_backbone_network():
    print(f"正在构建【全球贸易骨架】网络 (阈值 > {TRADE_THRESHOLD})...")
    df = pd.read_csv(TRADE_DATA_PATH, low_memory=False)
    year_col = f'Y{YEAR}'
    df_trade = df[df['Element'].str.contains('Export Value', case=False, na=False)].dropna(subset=[year_col])
    
    G = nx.DiGraph()
    edge_count_all = 0
    for _, row in df_trade.iterrows():
        edge_count_all += 1
        u, v, w = row['Reporter Countries'], row['Partner Countries'], row[year_col]
        # 仅保留高于阈值的强连接
        if w > TRADE_THRESHOLD:
            G.add_edge(u, v, weight=w)
            
    total_weight = sum(d['weight'] for _, _, d in G.edges(data=True))
    print(f"✅ 骨架提取完成: 保留了 {G.number_of_edges()}/{edge_count_all} 条主要连边.")
    return G, total_weight

# ... [此处省略与上一版本相同的 simulate_attack 和 get_attack_sequences 函数逻辑] ...
# 为了篇幅，我直接提供完整的 run 和 plot 逻辑

def get_attack_sequences(G):
    sequences = {}
    out_degrees = {n: sum(d['weight'] for _, _, d in G.out_edges(n, data=True)) for n in G.nodes()}
    sequences['Real Export (Degree)'] = sorted(out_degrees, key=out_degrees.get, reverse=True)
    try:
        v_df = pd.read_csv(os.path.join(RESULT_DIR, "vanilla_hits_scores_global_2021.csv"))
        sequences['Vanilla HITS'] = [n for n in v_df.sort_values(by='Hub Score (出)', ascending=False)['Country'].tolist() if n in G]
    except: pass
    for g in GAMMAS:
        try:
            g_df = pd.read_csv(os.path.join(RESULT_DIR, f"geobiased_gamma_{g:.1f}_hits_scores_global_2021.csv"))
            sequences[f'Geo-biased (γ={g:.1f})'] = [n for n in g_df.sort_values(by='Hub Score (出)', ascending=False)['Country'].tolist() if n in G]
        except: pass
    return sequences

def simulate_attack(G_original, attack_sequence, total_original_weight):
    G = G_original.copy()
    original_node_count = G.number_of_nodes()
    res = {'num_removed': [0], 'volume_ratio': [1.0], 'lcc_ratio': [1.0]}
    for i in range(1, min(MAX_ATTACK_NODES + 1, len(attack_sequence) + 1)):
        G.remove_node(attack_sequence[i-1])
        weight = sum(d['weight'] for _, _, d in G.edges(data=True))
        lcc = len(max(nx.weakly_connected_components(G), key=len)) / original_node_count if G.number_of_nodes() > 0 else 0
        res['num_removed'].append(i); res['volume_ratio'].append(weight/total_original_weight); res['lcc_ratio'].append(lcc)
    return pd.DataFrame(res)

def run_simulation():
    G, total_weight = build_backbone_network()
    sequences = get_attack_sequences(G)
    all_results = {name: simulate_attack(G, seq, total_weight) for name, seq in sequences.items()}
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = {'Real Export (Degree)': '#7f8c8d', 'Vanilla HITS': '#e74c3c'}
    geo_colors = ['#85c1e9', '#3498db', '#2980b9', '#2471a3', '#8e44ad']
    for i, g in enumerate(GAMMAS): colors[f'Geo-biased (γ={g:.1f})'] = geo_colors[i]
    
    for name, df in all_results.items():
        axes[0].plot(df['num_removed'], df['volume_ratio'], label=name, color=colors.get(name, 'black'))
        axes[1].plot(df['num_removed'], df['lcc_ratio'], label=name, color=colors.get(name, 'black'), linewidth=2 if '1.0' in name else 1)

    axes[0].set_title('Volume Degradation (Backbone Only)'); axes[1].set_title('Topological Fragmentation (Backbone Only)')
    axes[1].set_ylabel('LCC Ratio (Largest Component)'); axes[0].legend()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300); print(f"📈 骨架级联失效图已生成: {OUTPUT_IMG_PATH}")

if __name__ == "__main__": run_simulation()