import argparse
import numpy as np
import h5py
import csv
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
     
parser = argparse.ArgumentParser(description='Predict.')
parser.add_argument('-inputtrain', type=str, help='File with the input train data', required=True)

parser.add_argument('-inputtest', type=str, help='File with the input test data', required=True)

args = parser.parse_args()
print(args.inputtrain)


train_dataset = h5py.File(args.inputtrain, "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set imagenes
train_set_y = np.array(train_dataset["train_set_y"][:]).reshape(train_set_x_orig.shape[0], 1) # train set etiquetas

test_dataset = h5py.File(args.inputtest, "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set imagenes
test_set_y = np.array(test_dataset["test_set_y"][:]).reshape(test_set_x_orig.shape[0], 1) # test set etiquetas

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)


train_set_x_flatten = train_set_x_flatten/255.
test_set_x_flatten = test_set_x_flatten/255.

print('Imagenes de entrenamiento:', train_set_x_flatten.shape)
print('Etiquetas de entrenamiento:', train_set_y.shape)
print('Imagenes de prueba:', test_set_x_flatten.shape)



k = 30
pca = PCA(n_components = k)
pca.fit(train_set_x_flatten)
train_set_x_flatten= pca.transform(train_set_x_flatten)
print('Imagenes de entrenamiento:', train_set_x_flatten.shape)
test_set_x_flatten= pca.transform(test_set_x_flatten)
print('Imagenes de prueba:', test_set_x_flatten.shape)



def estimate_gaussian(dataset):
    #print('estimate_gaussian')
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma


def multivariate_gaussian(dataset, mu, sigma):
    #print('multivariate_gaussian')
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)


def select_eps(probs, dataset):
    #print('select_eps')
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs), max(probs), stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(dataset, predictions, average='binary')
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon

    return best_f1, best_epsilon
def calculate_fscore (eps,probs,dataset):
    predictions = (probs > eps)
    f = f1_score(dataset, predictions, average='binary')
    return f   

mu, sigma = estimate_gaussian(test_set_x_flatten)
p = multivariate_gaussian(test_set_x_flatten,mu,sigma)


p_cv = multivariate_gaussian(train_set_x_flatten,mu,sigma)
fscore, ep = select_eps(p_cv,train_set_y)
print(fscore, ep)

outliers = np.asarray(np.where(p < ep))

with open('cats_anomaly.csv', 'w') as csvfile:
    pwriter = csv.writer(csvfile)
    for i in range(test_set_x_flatten.shape[0]):
        if i in outliers[0]:
            pwriter.writerow([i, 1]) #anomaly
        else: 
            pwriter.writerow([i, 0]) #normal
        