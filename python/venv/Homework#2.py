import random  # import random for random sampling first centroids
import numpy as np  # import numpy for 2d matrix processing
import pandas as pd  # pandas for overall processing
import matplotlib.pyplot as plt  # for showing plots

LIMIT = 100  # Limit the maximum iteration(for get result much faster)
seed_num = 777  # set random seed
np.random.seed(seed_num)  # 난수 seed setting
iteration = 300  # if value doesn't change until 300 iterations


class K_means:
    def __init__(self,cluster_num, data, limit): #2차원 데이터를 받자.
        self.cluster_num=cluster_num
        self.data=data
        self.limit=limit

    def make_random_seeds(self):
        sample_data=self.data.to_numpy()
        idx=np.random.randint(sample_data.shape[0], size=self.cluster_num)
        random_seed=[]
        for i in idx:
            random_seed.append(sample_data[i])
        #print(np.array(random_seed))
        return np.array(random_seed)

    def Euclidean_norm(self, data, seeds, clusters): #clusters are dict
        for ins in data:
            mu=(min([(i[0], np.linalg.norm(ins-seeds[i[0]])) for i in enumerate(seeds)], key=lambda x: x[1]))[0]
            #데이터들과 seed간의 유클리드 거리 측정
            clusters.setdefault(mu, []).append(ins)
            #유클리드 거리가 가장 작은 시드에 각 데이터를 배정
        return clusters

    def update_stop(self, seeds, prev_seeds, iter_num): #업데이트 할거냐 말거냐
        if iter_num>self.limit:return 1
        return np.array_equal(seeds, prev_seeds)
        #return np.linalg.norm(seeds-prev_seeds)<1e-7 #업데이트된 seed와의 차이가 특정 norm 이하라면

    def assignment(self):
        data=self.data.to_numpy()
        seed=self.make_random_seeds() #initial seeds

        prev_seed=np.zeros((self.cluster_num, data.shape[1]))
        clusters = {i: [] for i in range(self.cluster_num)}
        iter_num=0
        while not self.update_stop(seed, prev_seed, iter_num): #업데이트 stop 전까지
            print(seed)
            iter_num+=1
            clusters={i: [] for i in range(self.cluster_num)}
            #old_result=[[] for i in range(self.cluster_num)]
            clusters=self.Euclidean_norm(data, seed, clusters)

            for i in clusters:
                print(i, clusters[i])

            idx=0
            for result in clusters:
                prev_seed[idx]=seed[idx]
                seed[idx]=np.mean(clusters[result])
                idx+=1

            """if np.array_equal(old_result,result):
                iter_num=0
            iteration=self.iteration
            old_result=result"""
        return clusters, iter_num

    def Train(self):
        iteration=0
        result_clusters, iteration=self.assignment()
        self.iteration=iteration
        return result_clusters



colorlist = ['red','purple','green','pink','blue','brown']
# Set color list (set this pallet because white and yellow is hard to congize)
data = pd.read_csv("data.csv")
model1=K_means(cluster_num=2, data=data, limit=LIMIT)
clusters=model1.Train()
result = [] #result list for set diff colors
for i in range(int(model1.cluster_num)):
    result = np.array(clusters[i]) # i control for result
    result_x = result[:,0] # Assign x
    result_y = result[:,1] # Assign y
    plt.plot(result_x,result_y, linestyle='None', color=colorlist[i], marker='o') #plt scatter for each clusters
plt.grid()
plt.xlabel('sepal length (cm)') # set label
plt.ylabel('sepal width (cm)') # set label
plt.title('K-means clustering') # set title
plt.show() # show plot