import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(BASE_DIR, "result")

def compare_top_hubs(k=30):
    print(f"🔍 正在对比 Vanilla vs Geo-biased(γ=1.0) 的 Top {k} 名单...\n")
    
    # 读取 Vanilla
    v_df = pd.read_csv(os.path.join(RESULT_DIR, "vanilla_hits_scores_global_2021.csv"))
    v_top = v_df.sort_values(by='Hub Score (出)', ascending=False)['Country'].tolist()[:k]
    
    # 读取 Geo-biased (1.0)
    g_df = pd.read_csv(os.path.join(RESULT_DIR, "geobiased_gamma_1.0_hits_scores_global_2021.csv"))
    g_top = g_df.sort_values(by='Hub Score (出)', ascending=False)['Country'].tolist()[:k]
    
    # 打印对比
    print(f"{'排名':<5} | {'Vanilla 选出的 Hub':<25} | {'Geo-biased 选出的 Hub':<25}")
    print("-" * 65)
    for i in range(k):
        v_country = v_top[i]
        g_country = g_top[i]
        # 如果名字不一样，加个星星标记
        marker = "⭐" if v_country != g_country else "  "
        print(f"Top {i+1:<2} | {v_country:<25} | {marker} {g_country:<25}")
        
    # 计算重合度
    overlap = len(set(v_top).intersection(set(g_top)))
    print(f"\n📊 前 {k} 名的重合度: {overlap}/{k} ({(overlap/k)*100:.1f}%)")

if __name__ == "__main__":
    compare_top_hubs(30)