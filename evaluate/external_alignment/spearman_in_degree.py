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

# 原始贸易数据路径 (使用 Reporter-Partner 镜像关系计算入度)
TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix_Europe", "Trade_DetailedTradeMatrix_E_Europe_NOFLAG.csv")
if not os.path.exists(TRADE_DATA_PATH):
    TRADE_DATA_PATH = os.path.join(BASE_DIR, "data", "TradeMatrix__Europe", "Trade_DetailedTradeMatrix_E_Europe_NOFLAG.csv")

OUTPUT_IMG_PATH = os.path.join(os.path.dirname(__file__), "spearman_in_degree_trend.png")

def calculate_real_in_degree(year=2021):
    """
    计算每个国家的总进口额 (Weighted In-degree)。
    在贸易网中，国家的入度即为所有 Partner 发往该 Reporter 的出口额总和。
    """
    print(f"正在从原始数据计算 {year} 年的真实入度（总进口额）...")
    df = pd.read_csv(TRADE_DATA_PATH)
    year_col = f'Y{year}'
    
    # 统一使用 Export Value 来构建入度，确保与 adj_matrix 的逻辑一致
    df_trade = df[df['Element'].str.contains('Export Value', case=False, na=False)].copy()
    
    # 在 adj_matrix 中，Partner Countries 是接收方（Target），所以按 Partner 分组求和得到入度
    in_degree_df = df_trade.groupby('Partner Countries')[year_col].sum().reset_index()
    in_degree_df.columns = ['Country', 'Real_In_Degree']
    return in_degree_df

def run_evaluation():
    if not os.path.exists(RESULT_DIR):
        print(f"❌ 找不到结果目录: {RESULT_DIR}")
        return

    # 1. 获取真实入度指标
    try:
        in_degree_metric = calculate_real_in_degree(year=2021)
    except Exception as e:
        print(f"❌ 计算入度失败: {e}")
        return

    # 2. 扫描所有 europe 结果文件
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
        merged = pd.merge(df, in_degree_metric, on='Country', how='inner')
        
        # 计算 Spearman (Authority Score vs Real In-degree)
        # 注意：CSV 中的列名是 'Authority Score (进)'
        coef, _ = spearmanr(merged['Authority Score (进)'], merged['Real_In_Degree'])
        
        eval_results.append({
            'Model': model_type,
            'Gamma': gamma,
            'Spearman_Coef': coef
        })

    # 3. 排序并展示
    results_df = pd.DataFrame(eval_results).sort_values(by='Gamma')
    print("\n📊 内部一致性检验结果 (Authority Score vs 真实总进口额):")
    print(results_df.to_string(index=False))

    # 4. 绘图
    plot_results(results_df)

def plot_results(results_df):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    geo_df = results_df[results_df['Model'] == 'Geo-biased HITS']
    sns.lineplot(data=geo_df, x='Gamma', y='Spearman_Coef', marker='D', 
                 linewidth=2.5, markersize=9, color='#16a085', label='Geo-biased HITS (Authority)')
    
    vanilla_coef = results_df[results_df['Model'] == 'Vanilla HITS'].iloc[0]['Spearman_Coef']
    plt.axhline(y=vanilla_coef, color='#d35400', linestyle='--', linewidth=2, label='Vanilla HITS (Baseline)')

    plt.title(r'Decoupling Effect: Authority Score vs. Raw Import Volume', fontsize=15, pad=15)
    plt.xlabel(r'Distance Penalty Parameter ($\gamma$)', fontsize=12)
    plt.ylabel('Spearman Correlation (with Total Import)', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n📈 进口解耦趋势图已保存至: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    run_evaluation()