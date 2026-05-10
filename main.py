import os
import pandas as pd
import numpy as np
from algo.HITS import hits_algorithm
from algo.Geobiasd_HITS import GeobiasedHITS
from utils import align_coordinates
import argparse

# ================= Non-country-entities blacklist =================
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
    
    if year_col not in df.columns:
        raise ValueError(f"Cannot find year_col: {year_col}")
        
    df_value = df[df['Element'].str.contains('Export Value', case=False, na=False)].copy()
    
    # 剔除所有“非国家”的聚合节点
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
    
    print(f"{n} nodes in total.\nAdjacency Matrix shape: {n} * {n}")
    
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    W = np.zeros((n, n), dtype=float)
    
    for _, row in edge_list.iterrows():
        u_idx = node_to_idx[row['Source']]
        v_idx = node_to_idx[row['Target']]
        weight = row['Weight_1000_USD']
        W[u_idx, v_idx] = weight
        
    return W, nodes

class FaostatRunner:

    def __init__(self, data_path, year=2021):

        self.edge_list = build_trade_network(data_path, target_year=year)
        self.adj_matrix, self.node_names = edges_to_matrix(self.edge_list)
        
    def run_hits(self, use_geo_bias=True, gamma=1.0):
        print(f"\n=== running {'Geo-biased' if use_geo_bias else 'vanilla'} HITS ===")
        
        if use_geo_bias:
            coords_df = pd.read_csv("./data/countries_coords.csv")
            coords_dict = align_coordinates(self.node_names, coords_df)
            
            hits_model = GeobiasedHITS(
                self.adj_matrix, 
                coords_dict, 
                self.node_names, 
                gamma=gamma
            )
        else:
            hits_model = hits_algorithm(self.adj_matrix)
            
        authority_scores, hub_scores = hits_model.run()
        
        results = []
        for i, country in enumerate(self.node_names):
            results.append({
                'Country': country,
                'Hub Score (出)': hub_scores[i],
                'Authority Score (进)': authority_scores[i]
            })
            
        df_results = pd.DataFrame(results)
        return df_results

def main():
    parser = argparse.ArgumentParser(description="Run HITS algorithm on FAOSTAT trade networks.")
    parser.add_argument('--Geo', action='store_true', help="Use Geo-biased HITS or not")
    parser.add_argument('--scope', type=str, choices=['europe', 'global'], default='europe', 
                        help="Choose the dataset scope: 'europe' or 'global' (default: europe)")
    parser.add_argument('--year', type=int, default=2021, help="Target year for the network (default: 2021)")
    parser.add_argument('--gamma', type=float, default=1.0, help="Gravity model decay parameter")
    
    args = parser.parse_args()

    if args.scope == 'europe':
        csv_path = "./data/TradeMatrix_Europe/Trade_DetailedTradeMatrix_E_Europe_NOFLAG.csv"
    elif args.scope == 'global':
        csv_path = "./data/TradeMatrix_Global/Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv"

    if not os.path.exists(csv_path):
        print(f"Error: cannot find {csv_path}")
        return

    print(f"Task: Scope=[{args.scope}], Year=[{args.year}], Geo-biased=[{args.Geo}], Gamma=[{args.gamma}]")
    runner = FaostatRunner(data_path=csv_path, year=args.year)
    final_scores = runner.run_hits(use_geo_bias=args.Geo, gamma=args.gamma)
    
    print(f"\nTop 5 Hubs ({args.scope.capitalize()} Network):")
    print(final_scores.sort_values(by='Hub Score (出)', ascending=False).head(5).to_string(index=False))
    
    result_dir = "./result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    if args.Geo:
        prefix = f"geobiased_gamma_{args.gamma}"
    else:
        prefix = "vanilla"
        
    file_name = f"{prefix}_hits_scores_{args.scope}_{args.year}.csv"
    # =======================================================================
    
    save_path = os.path.join(result_dir, file_name)
    
    final_scores.to_csv(save_path, index=False)
    print(f"\nResult saved as '{save_path}'")
    
if __name__ == "__main__":
    """
    examples:
    1. python main.py
    2. python main.py --Geo
    3. python main.py --scope global
    4. python main.py --scope global --Geo --year 2020
    5. python main.py --scope global --Geo --gamma
    """
    main()