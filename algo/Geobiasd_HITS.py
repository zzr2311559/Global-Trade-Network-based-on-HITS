import numpy as np
import pandas as pd
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .HITS import hits_algorithm
from utils import calculate_trade_distance

class GeobiasedHITS(hits_algorithm):
    def __init__(self, adj_matrix, coords_dict, node_names, gamma=1.0, max_iter=1000, tol=1e-8):
        """
        :param coords_dict: {name: (lat, lon)}
        """
        biased_matrix = self._apply_gravity_model(adj_matrix, coords_dict, node_names, gamma)
        
        super().__init__(biased_matrix, max_iter=max_iter, tol=tol)

    def _apply_gravity_model(self, adj_matrix, coords_dict, node_names, gamma):
        """
        W' = W / (d + 1)^gamma
        """
        n = adj_matrix.shape[0]
        biased_matrix = adj_matrix.copy()
                
        for i in range(n):
            for j in range(n):
                if biased_matrix[i, j] > 0:
                    u_name = node_names[i]
                    v_name = node_names[j]
                    
                    if u_name in coords_dict and v_name in coords_dict:
                        dist = calculate_trade_distance(coords_dict[u_name], coords_dict[v_name])
                        penalty = (dist/1000 + 1) ** gamma
                        biased_matrix[i, j] = biased_matrix[i, j] / penalty
                    else:
                        print(f"找不到坐标，跳过匹配: {u_name} 或 {v_name}") # 加这一行
                        biased_matrix[i, j] = biased_matrix[i, j] * 0.0
                        
        return biased_matrix