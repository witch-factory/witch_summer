import random  # import random for random sampling first centroids
import numpy as np  # import numpy for 2d matrix processing
import pandas as pd  # pandas for overall processing
import matplotlib.pyplot as plt  # for showing plots
import warnings  # deprecation issue handler

LIMIT = 900  # Limit the maximum iteration(for get result much faster)
seed_num = 777  # set random seed
np.random.seed(seed_num)  # seed setting
iteration = 300  # if value unchage untill 300 times


class k_means:
    def __init__(self,cluster_num, data, iteration): #2차원 데이터를 받자.
        self.cluster_num=cluster_num
        self.data=data
        self.iteration=iteration

    def random_seeds(self):
        sample_data=self.data.to_numpy().transpose()
        data_item_num=sample_data.shape[0]
        seeds=np.zeros((data_item_num, 1))
        for i in range(data_item_num):
            seeds[i]=random.random()*max(sample_data[i, :])
        return seeds

    def Euclidean_norm(self, data, seeds, clusters):
        for ins in data:
            mu=min([(i[0], np.linalg.norm(ins-seeds[i[0]])) for i in enumerate(seeds)], key=lambda x: x[1])
            #데이터들과 seed간의 유클리드 거리 측정
            clusters.setdefault(mu, []).append(ins)
            #유클리드 거리가 가장 작은 시드에 각 데이터를 배정

        """for result in clusters:
            if not result:
                result.append(data[])"""
        return clusters

    def update_stop(self, seeds, prev_seeds, iters): #업데이트 할거냐 말거냐
        if iters>LIMIT:return 1
        return seeds==prev_seeds #실수 저장 방식 등을 감안해서 시드 업데이트에서 prev_seed와의 차이 등으로 개조하자

    def Assignment(self):
        data=self.data.to_numpy()
        seed=self.random_seeds() #initial seeds
        prev_seed=[[] for i in range(self.cluster_num)]
        iter_num=0
        while not self.update_stop(seed, prev_seed, iter_num): #업데이트 stop 전까지
            iter_num+=1
            clusters=[[] for i in range(self.cluster_num)]
            old_result=[[] for i in range(self.cluster_num)]
            clusters=self.Euclidean_norm(data, seed, clusters)
            idx=0
            for result in clusters:
                prev_seed[idx]=seed[idx]
                seed[idx]=np.mean(result, axis=0).tolist()
                idx+=1
            if np.array_equal(old_result,result):
                iter_num=0
            iteration=self.iteration
            old_result=result
        return clusters, iteration

    def Train(self):
        iteration=0
        result, iteration=self.Assignment()
        self.iteration=iteration
        return result



colorlist = ['r','c','k','g','m','b','y']
# Set color list (set this pallet because white and yellow is hard to congize)
data = pd.read_csv("data.csv")
model1=k_means(cluster_num=3, data=data, iteration=iteration)
clusters=model1.Train()
result = [] #result list for set diff colors
for i in range(int(model1.k)): # for k case
    result = np.array(clustsers[i]) # i control for reslut
    result_x = result[:,0] # Assign x
    result_y = result[:,1] # Assign y
    plt.scatter(result_x,result_y,c=str((colorlist[i]))) #plt scatter for each clusters
plt.xlabel('sepal length (cm)') # set label
plt.ylabel('sepal width (cm)') # set label
plt.title("implementaion") # set title
plt.show() # show plot


print(data.to_numpy().transpose().shape)



