import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(BASE_DIR, "result")

TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Global", "Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv")
if not os.path.exists(TRADE_DATA_PATH):
    TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Global", "Trade_DetailedTradeMatrix_E_All_Data.csv")

# 输出名字带上 reward_lg
OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "cascading_failure_reward_lg_global.png")

YEAR = 2021
MAX_ATTACK_NODES = 30  
GAMMAS = [0.2, 0.4, 0.6, 0.8, 1.0]  

def build_trade_network():
    print("Building global baseline network...")
    df = pd.read_csv(TRADE_DATA_PATH, low_memory=False)
    year_col = f'Y{YEAR}'
    df_trade = df[df['Element'].str.contains('Export Value', case=False, na=False)].dropna(subset=[year_col])
    
    G = nx.DiGraph()
    for _, row in df_trade.iterrows():
        u = row['Reporter Countries']
        v = row['Partner Countries']
        w = row[year_col]
        if w > 0:
            G.add_edge(u, v, weight=w)
            
    total_weight = sum(d['weight'] for _, _, d in G.edges(data=True))
    print(f"[Done] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G, total_weight

def get_attack_sequences(G):
    sequences = {}
    
    out_degrees = {n: sum(d['weight'] for _, _, d in G.out_edges(n, data=True)) for n in G.nodes()}
    sequences['Real Export (Degree)'] = sorted(out_degrees, key=out_degrees.get, reverse=True)
    
    # 读取 reward_lg 的 vanilla
    try:
        vanilla_df = pd.read_csv(os.path.join(RESULT_DIR, "reward_lg_vanilla_hits_scores_global_2021.csv"))
        vanilla_nodes = vanilla_df.sort_values(by='Hub Score (出)', ascending=False)['Country'].tolist()
        sequences['Vanilla HITS'] = [n for n in vanilla_nodes if n in G]
    except Exception as e:
        print(f"Error reading Vanilla HITS: {e}")

    # 读取 reward_lg 的 geobiased
    for gamma in GAMMAS:
        try:
            geo_file = f"reward_lg_geobiased_gamma_{gamma:.1f}_hits_scores_global_2021.csv"
            geo_df = pd.read_csv(os.path.join(RESULT_DIR, geo_file))
            geo_nodes = geo_df.sort_values(by='Hub Score (出)', ascending=False)['Country'].tolist()
            sequences[f'Reward Geo (γ={gamma:.1f})'] = [n for n in geo_nodes if n in G]
        except Exception as e:
            print(f"Error reading Reward Geo-biased gamma {gamma}: {e}")
            
    return sequences

def simulate_attack(G_original, attack_sequence, total_original_weight):
    G = G_original.copy()
    original_node_count = G.number_of_nodes()
    
    results = {
        'num_removed': [0],
        'volume_ratio': [1.0],
        'lcc_ratio': [1.0]
    }
    
    for i in range(1, min(MAX_ATTACK_NODES + 1, len(attack_sequence) + 1)):
        target = attack_sequence[i-1]
        G.remove_node(target)
        
        current_weight = sum(d['weight'] for _, _, d in G.edges(data=True))
        if G.number_of_nodes() > 0:
            lwcc = max(nx.weakly_connected_components(G), key=len)
            lcc_size = len(lwcc)
        else:
            lcc_size = 0
            
        results['num_removed'].append(i)
        results['volume_ratio'].append(current_weight / total_original_weight)
        results['lcc_ratio'].append(lcc_size / original_node_count)
        
    return pd.DataFrame(results)

def run_simulation():
    G, total_weight = build_trade_network()
    sequences = get_attack_sequences(G)
    
    all_results = {}
    for name, seq in sequences.items():
        print(f"Evaluating attack strategy: {name.ljust(22)}")
        all_results[name] = simulate_attack(G, seq, total_weight)
        
    plot_simulation(all_results)

def plot_simulation(all_results):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = {'Real Export (Degree)': '#7f8c8d', 'Vanilla HITS': '#e74c3c'}
    markers = {'Real Export (Degree)': 'o', 'Vanilla HITS': 's'}
    
    geo_colors = ['#1abc9c', '#2ecc71', '#27ae60', '#16a085', '#0e6655']
    geo_markers = ['v', '^', '<', '>', 'D']  
    
    for i, gamma in enumerate(GAMMAS):
        name = f'Reward Geo (γ={gamma:.1f})'
        colors[name] = geo_colors[i]
        markers[name] = geo_markers[i]

    for name, df in all_results.items():
        axes[0].plot(df['num_removed'], df['volume_ratio'], label=name, 
                     color=colors.get(name, 'black'), marker=markers.get(name, 'x'), 
                     linewidth=2.5 if 'Vanilla' in name or 'γ=1.0' in name else 1.5,
                     markersize=7)
        axes[1].plot(df['num_removed'], df['lcc_ratio'], label=name, 
                     color=colors.get(name, 'black'), marker=markers.get(name, 'x'), 
                     linewidth=2.5 if 'Vanilla' in name or 'γ=1.0' in name else 1.5,
                     markersize=7)

    axes[0].set_title('Reward Model: Volume Retained', fontsize=14)
    axes[0].set_xlabel('Number of Hubs Removed', fontsize=12)
    axes[0].set_ylabel('Remaining Trade Volume (Ratio)', fontsize=12)
    axes[0].legend(fontsize=10)

    axes[1].set_title('Reward Model: LCC Size (Fragmentation)', fontsize=14)
    axes[1].set_xlabel('Number of Hubs Removed', fontsize=12)
    axes[1].set_ylabel('Largest Connected Component (Ratio)', fontsize=12)
    axes[1].legend(fontsize=10)

    plt.suptitle('Global Network Robustness (Reward Distance Model)', fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, bbox_inches='tight', dpi=300)
    print(f"\n[Success] Image saved to: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    run_simulation()