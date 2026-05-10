import os
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

"""
    spearman相关系数用于检测Hub值排名与LPI值排名的一致性
"""

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RESULT_DIR = os.path.join(BASE_DIR, "result")

EXTERNAL_DATA_PATH = os.path.join(BASE_DIR, "evaluate", "external_alignment", "External_Metrics", "europe_lpi_2021.csv")
OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "spearman_trend.png")

def run_evaluation():
    if not os.path.exists(RESULT_DIR):
        print(f"cannot find {RESULT_DIR}")
        return

    result_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('.csv') and 'europe' in f]
    
    if not result_files:
        print("No related file found in 'result' dir")
        return

    if os.path.exists(EXTERNAL_DATA_PATH):
        print(f"Successfully read data from {EXTERNAL_DATA_PATH}")
        ext_df = pd.read_csv(EXTERNAL_DATA_PATH)
    else:
        raise FileNotFoundError

    # Compute the spearman correlation coefficient
    
    eval_results = []
    
    for file in result_files:
        file_path = os.path.join(RESULT_DIR, file)
        df = pd.read_csv(file_path)
        
        if 'vanilla' in file:
            gamma = 0.0 # gamma of vanilla HITS is 0.0
            model_type = 'Vanilla HITS'
        else:
            match = re.search(r'gamma_([0-9.]+)', file)
            gamma = float(match.group(1)) if match else None
            model_type = 'Geo-biased HITS'
            
        if gamma is None:
            continue
            
        merged_df = pd.merge(df, ext_df, on='Country', how='inner')
        
        if len(merged_df) < 5:
            print(f"countries in {file} matched less than 5, skip")
            continue
            
        # 计算 Hub Score 与 LPI_Score 的 Spearman 秩相关系数
        coef, p_value = spearmanr(merged_df['Hub Score (出)'], merged_df['LPI_Score'])
        
        eval_results.append({
            'Model': model_type,
            'Gamma': gamma,
            'Spearman_Coef': coef,
            'P_Value': p_value
        })

    results_df = pd.DataFrame(eval_results).sort_values(by='Gamma')
    print("\nSpearman Rank Correlation:")
    print(results_df.to_string(index=False))

    plot_trend(results_df)

def plot_trend(results_df):

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    geo_df = results_df[results_df['Model'] == 'Geo-biased HITS']
    
    sns.lineplot(data=geo_df, x='Gamma', y='Spearman_Coef', marker='o', 
                 linewidth=2, markersize=8, color="#2893da", label='Geo-biased HITS')
    
    vanilla_row = results_df[results_df['Model'] == 'Vanilla HITS']
    if not vanilla_row.empty:
        vanilla_coef = vanilla_row.iloc[0]['Spearman_Coef']
        plt.axhline(y=vanilla_coef, color="#e97d24", linestyle='--', linewidth=2, label='Vanilla HITS (Baseline)')

    plt.title(r'Impact of Geographic Penalty ($\gamma$) on Hub Ranking Alignment', fontsize=15, pad=15)
    plt.xlabel(r'Distance Penalty Parameter ($\gamma$)', fontsize=12)
    plt.ylabel('Spearman Correlation (vs World Bank LPI)', fontsize=12)
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n📈 result saved as: {OUTPUT_IMG_PATH}")
    # plt.show() 

if __name__ == "__main__":
    run_evaluation()