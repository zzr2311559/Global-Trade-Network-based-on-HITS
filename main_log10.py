import os
import pandas as pd
import numpy as np
from algo.HITS import hits_algorithm
# 导入刚刚新建的以 10 为底的对数算法
from algo.Geobiasd_HITS_log10 import GeobiasedHITS_log10 
from utils import align_coordinates
import argparse

NON_COUNTRY_ENTITIES = {
    'World', 'Africa', 'Eastern Africa', 'Middle Africa', 'Northern Africa', 
    'Southern Africa', 'Western Africa', 'Americas', 'Northern America', 
    'Central America', 'Caribbean', 'South America', 'Asia', 'Central Asia', 
    'Eastern Asia', 'Southern Asia', 'South-eastern Asia', 'Western Asia', 
    'Europe', 'Eastern Europe', 'Northern Europe', 'Southern Europe', 
    'Western Europe', 'Oceania', 'Australia and New Zealand', 'Melanesia', 
    'Micronesia', 'Polynesia', 'European Union (27)', 
    'Least Developed Countries (LDCs)', 'Low Income Food Deficit Countries (LIFDCs)', 
    'Net Food Importing Developing Countries (NFIDCs)', 
    'Small Island Developing States (SIDS)', 
    'Land Locked Developing Countries (LLDCs)',
    'USSR', 'Yugoslav SFR', 'Czechoslovakia', 'Sudan (former)', 
    'Serbia and Montenegro', 'Ethiopia PDR', 'Belgium-Luxembourg'
}

def build_trade_network(data_path, target_year=2021):
    df = pd.read_csv(data_path)
    year_col = f'Y{target_year}'
    df_value = df[df['Element'].str.contains('Export Value', case=False, na=False)].copy()
    df_value = df_value[~df_value['Reporter Countries'].isin(NON_COUNTRY_ENTITIES)]
    df_value = df_value[~df_value['Partner Countries'].isin(NON_COUNTRY_ENTITIES)]
    
    edge_list = df_value.groupby(['Reporter Countries', 'Partner Countries'])[year_col].sum().reset_index()
    edge_list.columns = ['Source', 'Target', 'Weight_1000_USD']
    edge_list = edge_list[edge_list['Source'] != edge_list['Target']]
    edge_list = edge_list[edge_list['Weight_1000_USD'] > 0]
    return edge_list

def edges_to_matrix(edge_list):
    all_countries = set(edge_list['Source']).union(set(edge_list['Target']))
    nodes = sorted(list(all_countries)) 
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    W = np.zeros((n, n), dtype=float)
    for _, row in edge_list.iterrows():
        u_idx = node_to_idx[row['Source']]
        v_idx = node_to_idx[row['Target']]
        W[u_idx, v_idx] = row['Weight_1000_USD']
    return W, nodes

class FaostatRunner:
    def __init__(self, data_path, year=2021):
        self.edge_list = build_trade_network(data_path, target_year=year)
        self.adj_matrix, self.node_names = edges_to_matrix(self.edge_list)
        
    def run_hits(self, use_geo_bias=True, gamma=1.0):
        print(f"\n=== running {'Geo-biased (Log10)' if use_geo_bias else 'vanilla'} HITS ===")
        if use_geo_bias:
            coords_df = pd.read_csv("./data/countries_coords.csv")
            coords_dict = align_coordinates(self.node_names, coords_df)
            # 使用新模型
            hits_model = GeobiasedHITS_log10(self.adj_matrix, coords_dict, self.node_names, gamma=gamma)
        else:
            hits_model = hits_algorithm(self.adj_matrix)
            
        auth_scores, hub_scores = hits_model.run()
        results = [{'Country': n, 'Hub Score (出)': hub_scores[i], 'Authority Score (进)': auth_scores[i]} 
                   for i, n in enumerate(self.node_names)]
        return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Geo', action='store_true')
    parser.add_argument('--scope', type=str, default='global')
    parser.add_argument('--year', type=int, default=2021)
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args()

    csv_path = "./data/TradeMatrix_Global/Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv"
    if not os.path.exists(csv_path):
        print(f"Error: cannot find {csv_path}")
        return

    runner = FaostatRunner(data_path=csv_path, year=args.year)
    final_scores = runner.run_hits(use_geo_bias=args.Geo, gamma=args.gamma)
    
    # 保存的前缀变成了 log10_，这样你的各种实验数据就井井有条了
    prefix = f"log10_geobiased_gamma_{args.gamma}" if args.Geo else "log10_vanilla"
    save_path = f"./result/{prefix}_hits_scores_{args.scope}_{args.year}.csv"
    
    if not os.path.exists("./result"):
        os.makedirs("./result")
        
    final_scores.to_csv(save_path, index=False)
    print(f"Result saved as '{save_path}'")

if __name__ == "__main__":
    main()