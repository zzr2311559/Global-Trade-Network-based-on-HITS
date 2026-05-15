import numpy as np
from .HITS import hits_algorithm
from utils import calculate_trade_distance

class GeobiasedHITS_log10(hits_algorithm):
    def __init__(self, adj_matrix, coords_dict, node_names, gamma=1.0, max_iter=1000, tol=1e-8):
        biased_matrix = self._apply_gravity_model(adj_matrix, coords_dict, node_names, gamma)
        super().__init__(biased_matrix, max_iter=max_iter, tol=tol)

    def _apply_gravity_model(self, adj_matrix, coords_dict, node_names, gamma):
        n = adj_matrix.shape[0]
        biased_matrix = adj_matrix.copy()
                
        for i in range(n):
            for j in range(n):
                if biased_matrix[i, j] > 0:
                    u_name, v_name = node_names[i], node_names[j]
                    if u_name in coords_dict and v_name in coords_dict:
                        dist = calculate_trade_distance(coords_dict[u_name], coords_dict[v_name])
                        
                        # ================= 数学公式区 =================
                        # 使用以 10 为底的对数：np.log10
                        # 将平移量 10 放在内部：当 dist=0 时，log10(10) = 1
                        penalty = (np.log10(dist / 1000 + 10)) ** gamma
                        biased_matrix[i, j] = biased_matrix[i, j] / penalty
                        # ===================================================
                        
                    else:
                        biased_matrix[i, j] = 0.0
                        
        return biased_matrix