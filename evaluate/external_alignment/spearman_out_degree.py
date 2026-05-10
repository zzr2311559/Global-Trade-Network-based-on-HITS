import os
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置路径 =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(BASE_DIR, "result")

# 原始贸易数据路径 (用于计算真实出度)
# 注意：根据你的环境，这里可能是 TradeMatrix_Europe 或 TradeMatrix__Europe
TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Europe", "Trade_DetailedTradeMatrix_E_Europe_NOFLAG.csv")
if not os.path.exists(TRADE_DATA_PATH):
    # 尝试双下划线路径
    TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix__Europe", "Trade_DetailedTradeMatrix_E_Europe_NOFLAG.csv")

OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "spearman_out_degree_trend.png")

def calculate_real_out_degree(year=2021):
    """
    从原始 CSV 中直接计算每个国家的总出口额 (Weighted Out-degree)
    """
    print(f"正在从原始数据计算 {year} 年的真实出度...")
    df = pd.read_csv(TRADE_DATA_PATH)
    year_col = f'Y{year}'
    
    # 筛选出口数据 (Export Value)
    df_export = df[df['Element'].str.contains('Export Value', case=False, na=False)].copy()
    
    # 按发货国聚合
    out_degree_df = df_export.groupby('Reporter Countries')[year_col].sum().reset_index()
    out_degree_df.columns = ['Country', 'Real_Out_Degree']
    return out_degree_df

def run_evaluation():
    if not os.path.exists(RESULT_DIR):
        print(f"❌ 找不到结果目录: {RESULT_DIR}")
        return

    # 1. 获取真实出度作为指标
    try:
        out_degree_metric = calculate_real_out_degree(year=2021)
    except Exception as e:
        print(f"❌ 计算出度失败: {e}")
        return

    # 2. 扫描所有结果文件
    result_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('.csv') and 'europe' in f]
    
    eval_results = []
    for file in result_files:
        file_path = os.path.join(RESULT_DIR, file)
        df = pd.read_csv(file_path)
        
        # 识别模型类型和 Gamma
        if 'vanilla' in file:
            gamma = 0.0
            model_type = 'Vanilla HITS'
        else:
            match = re.search(r'gamma_([0-9.]+)', file)
            gamma = float(match.group(1)) if match else None
            model_type = 'Geo-biased HITS'
            
        if gamma is None: continue
            
        # 合并数据
        merged = pd.merge(df, out_degree_metric, on='Country', how='inner')
        
        # 计算 Spearman (Hub Score vs Real Out-degree)
        coef, _ = spearmanr(merged['Hub Score (出)'], merged['Real_Out_Degree'])
        
        eval_results.append({
            'Model': model_type,
            'Gamma': gamma,
            'Spearman_Coef': coef
        })

    # 3. 排序并展示
    results_df = pd.DataFrame(eval_results).sort_values(by='Gamma')
    print("\n📊 内部一致性检验结果 (Hub Score vs 真实总出口额):")
    print(results_df.to_string(index=False))

    # 4. 绘图
    plot_results(results_df)

def plot_results(results_df):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    geo_df = results_df[results_df['Model'] == 'Geo-biased HITS']
    sns.lineplot(data=geo_df, x='Gamma', y='Spearman_Coef', marker='s', 
                 linewidth=2.5, markersize=9, color='#8e44ad', label='Geo-biased HITS')
    
    vanilla_coef = results_df[results_df['Model'] == 'Vanilla HITS'].iloc[0]['Spearman_Coef']
    plt.axhline(y=vanilla_coef, color='#c0392b', linestyle='--', linewidth=2, label='Vanilla HITS (Baseline)')

    plt.title(r'Decoupling Effect: Hub Score vs. Raw Export Volume', fontsize=15, pad=15)
    plt.xlabel(r'Distance Penalty Parameter ($\gamma$)', fontsize=12)
    plt.ylabel('Spearman Correlation (with Total Export)', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n📈 对比趋势图已保存至: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    run_evaluation()