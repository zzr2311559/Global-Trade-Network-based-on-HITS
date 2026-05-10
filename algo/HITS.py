import numpy as np


class hits_algorithm(object):
    
    def __init__(self, adj_matrix, max_iter=1000, tol=1e-8):
        self.adj_matrix = adj_matrix
        self.max_iter=max_iter
        self.tol=tol
    
    def run(self):
        
        n = self.adj_matrix.shape[0]
        
        h = np.ones(n)
        a = np.ones(n)
        
        for i in range(self.max_iter):
            h_old = h.copy()
            a_old = a.copy()
            
            # 注意顺序：通常先用旧的 h 更新 a，再用新的 a 更新 h
            a = self.adj_matrix.T.dot(h_old) 
            h = self.adj_matrix.dot(a)
            
            # L2 Normalization
            a = a / np.linalg.norm(a, 2)
            h = h / np.linalg.norm(h, 2)
            
            a_diff = np.linalg.norm(a - a_old, 2)
            h_diff = np.linalg.norm(h - h_old, 2)
            
            if a_diff + h_diff < self.tol:
                print(f"Converge after {i+1} iters")
                break
        return a, h

if __name__ == "__main__":
 
    A = np.array([
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])

    hits_A = hits_algorithm(A)
    a, h = hits_A.run()
    
    print("authority scores: ", a)
    print("hub scores", h)
