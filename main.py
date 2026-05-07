import os
import pandas as pd
import numpy as np
from hits import hits_algorithm

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
        
    def run_hits(self):

        hits_model = hits_algorithm(self.adj_matrix, max_iter=1000, tol=1e-8)
        
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



if __name__ == "__main__":

    csv_path = "./data/TradeMatrix__Europe/Trade_DetailedTradeMatrix_E_Europe_NOFLAG.csv"
    
    runner = FaostatRunner(data_path=csv_path, year=2021)
    final_scores = runner.run_hits()
    
    print("\nTop 5 Hubs:")
    print(final_scores.sort_values(by='Hub Score (出)', ascending=False).head(5).to_string(index=False))
    
    result_dir = "./result"
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    save_path = os.path.join(result_dir, "hits_scores_2021.csv")
    
    final_scores.to_csv(save_path, index=False)
    
    print(f"\nResult saved as '{save_path}'")