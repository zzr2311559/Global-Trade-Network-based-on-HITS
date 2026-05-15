import os
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(BASE_DIR, "result")
EXTERNAL_DATA_PATH = os.path.join(BASE_DIR, "evaluate", "external_alignment", "External_Metrics", "europe_lpi_2021.csv")
OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "spearman_topK_REWARD_lg_trend.png")

def run_evaluation():
    result_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('.csv') and 'reward_lg_geobiased' in f]
    
    if not result_files:
        print("未找到结果文件，请检查是否运行了 main_reward_lg.py")
        return

    ext_df = pd.read_csv(EXTERNAL_DATA_PATH)
    eval_results = []
    k_list = [30, 50, 100, None] 
    
    for file in result_files:
        df = pd.read_csv(os.path.join(RESULT_DIR, file))
        match = re.search(r'gamma_([0-9.]+)', file)
        gamma = float(match.group(1)) if match else None
        if gamma is None: continue
            
        merged_df = pd.merge(df, ext_df, on='Country', how='inner')
        
        for k in k_list:
            if k is not None:
                temp_df = merged_df.sort_values(by='Hub Score (出)', ascending=False).head(k)
                label = f"Top-{k}"
            else:
                temp_df = merged_df
                label = "All Countries"
                
            if len(temp_df) < 5: continue
            coef, p_value = spearmanr(temp_df['Hub Score (出)'], temp_df['LPI_Score'])
            
            eval_results.append({'Gamma': gamma, 'K_Value': label, 'Spearman_Coef': coef})

    results_df = pd.DataFrame(eval_results).sort_values(by=['K_Value', 'Gamma'])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=results_df, x='Gamma', y='Spearman_Coef', hue='K_Value', 
                 marker='s', linewidth=2, markersize=8, palette="Dark2")
    
    # 标题注明这是 Reward Distance 实验
    plt.title(r'Inverse Logic (Reward Distance): Impact of Top-K on LPI Correlation', fontsize=15, pad=15)
    plt.xlabel(r'Distance Reward Parameter ($\gamma$)', fontsize=12)
    plt.ylabel('Spearman Correlation (vs World Bank LPI)', fontsize=12)
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.legend(title="Country Scope", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n 奖励远距离的终极图表已保存至: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    run_evaluation()