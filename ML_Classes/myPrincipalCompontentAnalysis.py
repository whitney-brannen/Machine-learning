import numpy as np
import pandas as pd

class myPrincipalComponentAnalysis:
    
    def __init__(self, data):
        # make class take pandas or numpy array
        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy()
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            pass
        self.n_components = self.data.shape[1]   
        self.mean_val = None
        self.mc = None # X matrix, mean centered data
        self.cov = None
        self.eigenvalues = None # for scree plot
        self.eigenvectors = None # p matrix
        self.scores = None # y matrix
        self.var = None
    
    def __repr__(self):
        return np.array_str(self.data)
    
    def mean(self):
        self.mean_val = np.mean(self.data,axis=0, keepdims=True)
        return self.mean_val
    
    def mean_centered(self):
        if self.mean_val is None:
            self.mean() 
            
        self.mc = np.real(self.data - self.mean_val)

        return self.mc
    
    # p matrix
    def covariance_matrix(self):
        if self.mc is None:
            self.mean_centered()
            
        self.cov = np.cov(self.mc, rowvar=False) # rowvar False means rows are samples and columns are variables
        return self.cov
    
    def eigendecomposition(self):
        if self.cov is None:
            self.covariance_matrix()
        
        self.eigenvalues,self.eigenvectors = np.linalg.eig(self.cov)

        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[sorted_indices].tolist()
        self.eigenvectors = self.eigenvectors[:, sorted_indices]
        
    def variance_explained(self):
        if self.eigenvalues is None:
            self.eigendecomposition()
            
        var_total = sum(self.eigenvalues)
        self.var = []
        for x in self.eigenvalues:
            var = x/var_total
            self.var.append(var)
        self.var = np.real(self.var)
        return self.var 
    
    def project_data(self):
         # y matrix/ scores
        if self.eigenvectors is None:
            self.eigendecomposition()
        
        self.scores = np.real(np.dot(self.mc, self.eigenvectors))
        return self.scores
    
    def loading_plot(self):  
        x_values = np.real(self.eigenvectors[:,0])
        y_values = np.real(self.eigenvectors[:,1])
        plt.scatter(x_values,y_values,c='blue')
        plt.title("Loading Plot")
        plt.xlabel(f"PC1: {round(self.var[0]*100,2)}%")
        plt.ylabel(f"PC2: {round(self.var[1]*100,2)}%")
        return plt
    
    def scree_plot(self):
        plt.plot(np.real(self.eigenvalues), marker="o", linewidth=0,markerfacecolor='blue')
        plt.title('Scree Plot') 
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        return plt
    
    def scores_plot(self):
        pca_df = self.transform()
        x1 = pca_df[:,0]
        x2 = pca_df[:,1]
        l,d = np.shape(pca_df)
        ll = int(l//2) # dimensions
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(x1[:ll], x2[:ll], color = "blue")
        ax.scatter(x1[ll:], x2[ll:], color = "red")
        ax.set_title("Scores Plot")
        ax.set_xlabel(f"PC1: {round(self.var[0]*100,2)}%")
        ax.set_ylabel(f"PC2: {round(self.var[1]*100,2)}%")
        return 
    
    def fit(self):
        # return learned p matrix/ loadings
        self.mean_centered()
        self.covariance_matrix()
        self.eigendecomposition()
        self.variance_explained()
        return f"PCA fit: n_components={self.n_components}"
    
    def transform(self):
        # return scores (y=xp)
        self.project_data()
        return self.scores
    
    def fit_transform(self):
        self.fit()
        return self.transform()
