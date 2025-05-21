#Extra Functions

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from numpy import std
import random
from numpy import mean

from minisom import MiniSom
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from sklearn.preprocessing import MinMaxScaler #normalisation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #scoring
from sklearn.metrics import confusion_matrix
from matplotlib import animation, colors




#Principle Component Analysis Function
def PCA(data, RedDim):
    #Subtract Mean of each feature from respective column
    Xmean = data - np.mean(data, axis=0)
    #Calculate covariance matrix of mean centered matrix
    covMatrix = np.dot(Xmean.T, Xmean)
    #Calculate eigenvalues and eigenvectors of covariance matrix
    eigenvalue, eigenvector = np.linalg.eig(covMatrix)

    #Sort eigenvalues and eigenvectors
    idx = eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[idx]
    eigenvector = eigenvector[:, idx]

    #get feature vector
    featureVector = eigenvector[:, :RedDim]
    #Transform data using top eigenvectors
    dataTrans = np.dot(data, featureVector)

    #return transformed data
    return dataTrans

#plot the obtained data from principle component analysis
def PCAPlot(dataTrans, RedDim):
    if RedDim == 1:
        #Plots a Scatter Plot for a single dimension, x-axis is first dimension, y-axis is second dimension
        plt.scatter(dataTrans[:,0], np.zeros(dataTrans.shape[0]))
        return plt
    elif RedDim == 2:
        #plot original data in 2 dimensions
        plt.scatter(dataTrans[:,0], dataTrans[:,1])
        return plt
    elif RedDim == 3:
        #plot data in 3 dimension scatter plot
        #Setup 3D graph
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #Plot transformed data
        ax.scatter(dataTrans[:,0], dataTrans[:,1], dataTrans[:,2])
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        return plt
    else:
        print("Error, cannot plot more than 3 dimensions")


#kmeans cluster
def kmeans(data, k, maxIterations):
    # Initialize centroids randomly
    centroids = random.sample(data.tolist(), k)
    # Initialize array to store cluster assignments
    clusters = np.zeros(len(data))

    # Repeat until max iterations reached
    for _ in range(maxIterations):
        # Assign each point to closest centroid
        for i, point in enumerate(data):
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            clusterAs = np.argmin(distances)
            clusters[i] = clusterAs

        # Update centroids based on mean of points in each cluster
        for cluster in range(k):
            points = data[clusters == cluster]
            if len(points) > 0:
                centroids[cluster] = np.mean(points, axis=0)
    return clusters.astype(int)

#Plot the kmeans clusters
def kmeansPlot(clusters, data):
    # Generate scatter plot
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-Means Clustering')
    return plt

#Parallel Coordinates plot
def PCP(data, clusters, dataTop):
    num_clusters = len(set(clusters))
    fig, ax = plt.subplots()
    
    # Normalize the data
    NormData = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    
    # Assign colors to clusters
    color_map = plt.cm.get_cmap('tab10', num_clusters)
    
    # Plot each line
    for i, point in enumerate(NormData):
        cluster_color = color_map(clusters[i])
        ax.plot(range(len(point)), point, color=cluster_color)
    
    # Set the labels for each axis
    ax.set_xticks(range(len(data[0])))
    ax.set_xticklabels([i for i in dataTop])
    return plt


#KFold cross validation
#model type is the type of machine learning used
#X is the feature matrix of the dataset
#y is the target vector of the dataset
#k is the number of folds for the cross validation
def kfold(X, y, k):
    #prepare cross-validation procedure
    cv = KFold(n_splits = k, random_state=1, shuffle=True)
    #create model
    model = get_model()
    #evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    model.fit(X, y)
    #report performance
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    return model, scores
#Get model type
#Strictly using logistic regression here, can use others if needed
def get_model():
    model = LogisticRegression()
    return model

#Self organising maps
def SOM(data, target, Size, MaxMdis, MaxLrate, MaxStep):
    NoRow = Size
    NoCol = Size
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size = 0.2, random_state=42)
    trainXnorm = minmaxS(Xtrain)

    #init SOM
    NumDim = trainXnorm.shape[1]#number of dimensions in input
    som = np.random.random_sample(size=(NoRow, NoCol, NumDim)) #map construction

    #train iterations
    for step in range(MaxStep):
        lrate, nrate = decay(step, MaxStep, MaxLrate, MaxMdis)

        t = np.random.randint(0, high=trainXnorm.shape[0]) #select random input vector
        winner = Winning(trainXnorm, t, som, NoRow, NoCol)
        for row in range(NoRow):
            for col in range(NoCol):
                if ManDis([row,col], winner) <= nrate:
                    som[row][col] += lrate * (trainXnorm[t]-som[row][col]) #update neighbour weight

    #label data
    labelD = Ytrain
    mapp = np.empty(shape=(NoRow, NoCol), dtype=object)

    for row in range(NoRow):
        for col in range(NoCol):
            mapp[row][col] = []

    for t in range(trainXnorm.shape[0]):
        winner = Winning(trainXnorm, t, som, NoRow, NoCol)
        mapp[winner[0]][winner[1]].append(labelD[t]) #label of winning neuron

    #Construct map
    labelMap = np.zeros(shape=(NoRow, NoCol), dtype=np.int64)
    for row in range(NoRow):
        for col in range(NoCol):
            labelL = mapp[row][col]
            if len(labelL)==0:
                Label = 2
            else:
                Label = max(labelL, key=labelL.count)
            labelMap[row][col]=Label
    
    cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange'])
    plt.imshow(labelMap, cmap=cmap)
    plt.colorbar()
    return plt




#Data normalisation
def minmaxS(data):
    #normalise data between [0,1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled

#Euclidean distance
def EucDis(x,y):
    return distance.euclidean(x,y)

#Manhatten distance
def ManDis(x,y):
    return distance.cityblock(x,y)

#Best matching unit search
def Winning(data, t, som, NoRow, NoCol):
    winner = [0,0]
    #shortest distance, init with max distance
    Sdis = np.sqrt(data.shape[1])
    inp = data[t]
    for row in range(NoRow):
        for col in range(NoCol):
            distance = EucDis(som[row][col], data[t])
            if distance < Sdis:
                Sdis = distance
                winner = [row,col]
    return winner

#Learning rate and neighbourhood range calculation
def decay(step, MaxStep, MaxLrate, MaxMdis):
    coefficient = 1.0 - (np.float64(step)/MaxStep)
    lrate = coefficient*MaxLrate
    neigh = ceil(coefficient * MaxMdis)
    return lrate, neigh







































