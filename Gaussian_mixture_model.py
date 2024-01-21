import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')


class Gaussian_mixture_model:
    def __init__(self):
        self.n_clusters = None
    # data as a matrix format
    def data_matrix(self,data):
        X = []
        for i in range(len(data)):
            X.append(np.array(data.iloc[i]))
        return (np.matrix(X))

    # defining the mean
    def mean(self,prob_mat,data):
        X = self.data_matrix(data)
        means = []
        for i in range(len(prob_mat)):
            x = 0
            y = 0
            for j in range(len(prob_mat[0])):
                x += prob_mat[i][j] * X[j].T
                y += prob_mat[i][j]
            means.append(x/y)
        return means

    def covariance(self,prob_mat,data):
        X = self.data_matrix(data)
        cov = []
        means = self.mean(prob_mat,data)
        for i in range(len(prob_mat)):
            x = 0
            y = 0
            for j in range(len(prob_mat[0])):
                x += prob_mat[i][j] * np.matmul((X[j].T - means[i]),(X[j].T - means[i]).T)
                y += prob_mat[i][j]
            cov.append(x/(y-1))
        return cov

    
    #defining multivariate gaussian distribution
    def multi_gauss_dis(self,prob_mat , data):
        means = self.mean(prob_mat , data)
        cov = self.covariance(prob_mat , data)
        X = self.data_matrix(data)
        Q = np.empty([len(prob_mat) , len(prob_mat[0])])
        for i in range(len(prob_mat)):
            for j in range(len(prob_mat[0])):
                Q[i][j]  = float(1/(2 * math.pi * np.linalg.det(cov[i])*np.matmul((X[j] - means[i].T),np.matmul(np.linalg.inv(cov[i]),(X[j].T - means[i])))))
        return Q


    #defining probability of each class
    def class_probability(self,prob_mat , data):
        class_prob = []
        for i in range(len(prob_mat)):
            x = 0
            for j in range(len(prob_mat[0])):
                x += prob_mat[i][j]
            class_prob.append(x/len(data))
        return class_prob


    # allocate the class
    def calculate_class(self,data , prob_mat):
        t = np.zeros([len(prob_mat),len(prob_mat[0])])
        X = self.data_matrix(data)
        Q = self.multi_gauss_dis(prob_mat , data)
        prob_class = self.class_probability(prob_mat , data)
        for i in range(len(Q)):
            for j in range(len(Q[0])):
                Q[i][j] = prob_class[i] * Q[i][j]
        x = []
        for i in range(len(Q[0])):
            x.append(np.argmax(Q[:,i]))
        j = 0
        for i in range(len(prob_mat[0])):
            t[:,i][x[j]] = 1.0
            j +=1
        return t



    def fit(self,prob_mat , data):
        for i in range(10):
            # E step
            prob_mat = self.calculate_class(data , prob_mat)
            print(prob_mat)
        return prob_mat
    

    def cluster_alloc(self,prob_mat,data):
        X = self.data_matrix(data)
        cluster_1 = []
        cluster_2 = []
        cluster_3 = []
        cluster_4 = []
        for i in range(len(data)):
            if prob_mat[:,i][0] == 1:
                cluster_1.append(X[i])
            if prob_mat[:,i][1] == 1:
                cluster_2.append(X[i])
            if prob_mat[:,i][2] == 1:
                cluster_3.append(X[i])
            if prob_mat[:,i][3] == 1:
                cluster_4.append(X[i])
        return cluster_1,cluster_2,cluster_3,cluster_4
