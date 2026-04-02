
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn.model_selection as selection
import matplotlib.pyplot as plt
from sklearn import metrics


# modelling the loss function as a MSE from pytorch as we will be working tensors and to prevent any incompatablity from an MSE function from other libaraies
mse = nn.MSELoss()
def loss_function(predictions, actual):
    return mse(predictions, actual)

#function to plot computation graph with gradient descent
def plot_loss(loss_curve):
    plt.plot(list(range(len(loss_curve))), loss_curve)

# defining the Linear Regression model with weight and constant variables 
class LinearRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        w = th.tensor(np.random.random(size=n_features), dtype=th.float32)
        c = th.tensor(np.random.random(), dtype=th.float32)
        self.w = nn.Parameter(w)
        self.c = nn.Parameter(c)

    def predict(self, X):
     #transponsing the feature matix so that it can be use to find the dot product of w and x 
        return self.w @ th.transpose(X, 0, 1) + self.c
      

    def forward(self, X):
        return self.predict(X)

    #calculates the loss of a pridiction by calling the loss_function
    def loss(self, X, y):
        predictions = self.predict(X)
        return loss_function(predictions, y)

# then nmber of columns ncols,features, and the number of row nrows, sample house
nrows, ncols = X.shape 
#with the data being scaled, it produce large numbers, indicating numerical instablity
y_scaled = (y - y.mean()) / y.std()

#splitting the dataset into training and testing sets using a 80/20 split with random selection 
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42,shuffle=True)

#the dataset were in pandas serires while torch requires numpy array to convert to tensors
X_train =X_train.to_numpy()
X_test =X_test.to_numpy()

#converting the x training and test data to tensors so develop the computational graph
X_train = th.tensor(X_train, dtype=th.float32)
y_train = th.tensor(y_train, dtype=th.float32)

#assigning the linear regression model  
model = LinearRegression(ncols)
#learning 01rates were tested but anything lr less that 0.05 was within .1 of the miniume value
lr = 0.001
# Modified SGD
optimizer = optim.RMSprop(model.parameters(), lr=lr) D
num_iters = 2000
loss_curve = []

for i in range(num_iters):
    optimizer.zero_grad()
    loss_value = model.loss(X_train, y_train)
    loss_curve.append(loss_value.data.item())
    loss_value.backward()
    optimizer.step()

plot_loss(loss_curve)
w = model.w.detach()
c = model.c.detach()
print(w, c)

X_test = th.tensor(X_test, dtype=th.float32)
predictions = model.predict(X_test)
predictions = predictions.clone().detach()

y_pred_real = predictions * y.std() + y.mean()
y_test_real = y_test * y.std() + y.mean()


print("MSE", metrics.mean_squared_error(y_test_real, y_pred_real))
print("RMSE", metrics.root_mean_squared_error(y_test_real, y_pred_real))
print("MAE", metrics.mean_absolute_error(y_test_real, y_pred_real))
print("R²:", metrics.r2_score(y_test_real, y_pred_real))


plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, y_pred_real, alpha=0.4, s=15)
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min(), y_test_real.max()],
         'r--', label='Perfect prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('PyTorch Linear Regression: Predicted vs Actual')
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.decomposition import PCA

#fitting pca on the training data while keeping 95% of variance
pca =PCA(n_components=0.95)

X_train_pca= pca.fit_transform(X_train)

X_test_pca =pca.transform(X_test)

print(f'Original features: {X_train.shape[1]}')
print(f'PCA features (95% variance): {X_train_pca.shape[1]}')
print(f'Reduction: {1 - X_train_pca.shape[1]/X_train.shape[1]:.0%} fewer features')

plt.figure(figsize=(8, 4))
cumulative = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumulative)+1), cumulative, 'b.-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#converting the x training and test data to tensors so develop the computational graph
X_trainp = th.tensor(X_train_pca, dtype=th.float32)
y_train = th.tensor(y_train, dtype=th.float32)

#assigning the linear regression model  
model1 = LinearRegression(X_train_pca.shape[1])
#learning 01rates were tested but anything lr less that 0.05 was within .1 of the miniume value
lr = 0.001
# Modified SGD
optimizer = optim.RMSprop(model1.parameters(), lr=lr) 
num_iters = 2000
loss_curve = []

for i in range(num_iters):
    optimizer.zero_grad()
    loss_value = model1.loss(X_trainp, y_train)
    loss_curve.append(loss_value.data.item())
    loss_value.backward()
    optimizer.step()

plot_loss(loss_curve)
w = model1.w.detach()
c = model1.c.detach()
print(w, c)

X_test = th.tensor(X_test_pca, dtype=th.float32)
predictions = model1.predict(X_test)
predictions = predictions.clone().detach()

y_pred_real = predictions * y.std() + y.mean()
y_test_real = y_test * y.std() + y.mean()

print("MSE", metrics.mean_squared_error(y_test_real,y_pred_real))
print("RMSE", metrics.root_mean_squared_error(y_test_real,y_pred_real))
print("MAE", metrics.mean_absolute_error(y_test_real,y_pred_real))
print("R²:", metrics.r2_score(y_test_real,y_pred_real))

plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, y_pred_real, alpha=0.4, s=15)
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min(), y_test_real.max()],
         'r--', label='Perfect prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('PyTorch Linear Regression: Predicted vs Actual')
plt.legend()
plt.tight_layout()
plt.show()


  
