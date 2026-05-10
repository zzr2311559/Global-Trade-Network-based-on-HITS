import os
import pandas as pd
import numpy as np
from algo.HITS import hits_algorithm
from algo.Geobiasd_HITS import GeobiasedHITS
import argparse

def build_trade_network(data_path, target_year=2021):
    
    df = pd.read_csv(data_path)
    year_col = f'Y{target_year}'
    
    if year_col not in df.columns:
        raise ValueError(f"Cannot find year_col: {year_col} ")
        
    df_value = df[df['Element'].str.contains('Export Value', case=False, na=False)].copy()
    
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

        edge_list = build_trade_network(data_path, target_year=year)
        self.adj_matrix, self.node_names = edges_to_matrix(edge_list)
        
    def run_hits(self, use_geo_bias=True, gamma=1.0):
        print(f"\n=== running {'Geo-biased' if use_geo_bias else 'vanilla'} HITS  ===")
        
        if use_geo_bias:
            coords_df = pd.read_csv("./data/countries_coords.csv")
            coords_dict = {row['name']: (row['latitude'], row['longitude']) for _, row in coords_df.iterrows()}
            
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--Geo', action='store_true', help="Geo-biased HITS")

    args = parser.parse_args()

    csv_path = "./data/TradeMatrix__Europe/Trade_DetailedTradeMatrix_E_Europe_NOFLAG.csv"
    
    runner = FaostatRunner(data_path=csv_path, year=2021)
    final_scores = runner.run_hits(use_geo_bias = True if args.Geo else False)
    
    print("\nTop 5 Hubs:")
    print(final_scores.sort_values(by='Hub Score (出)', ascending=False).head(5).to_string(index=False))
    
    result_dir = "./result"
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    if args.Geo:
        save_path = os.path.join(result_dir, "geobiased_hits_scores_2021.csv")
    else:
        save_path = os.path.join(result_dir, "vanilla_hits_scores_2021.csv")

    
    final_scores.to_csv(save_path, index=False)
    
    print(f"\nResult saved as '{save_path}'")
    

if __name__ == "__main__":
    """
    Usage: Run 'python main.py --Geo' to use geobiased HITS.
    """
    main()