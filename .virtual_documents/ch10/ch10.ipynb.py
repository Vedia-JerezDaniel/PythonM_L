from IPython.display import Image
get_ipython().run_line_magic("matplotlib", " inline")


Image(filename='images/10_01.png', width=500) 


Image(filename='images/10_15.png', width=500) 


import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-3rd-edition/'
                 'master/ch10/housing.data.txt',
                 header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()


import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix


cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

scatterplotmatrix(df[cols].values, figsize=(8, 6), names=cols, alpha=0.5)
plt.tight_layout()

plt.show()


import numpy as np
from mlxtend.plotting import heatmap

cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)

plt.show()


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


X = df[['RM','LSTAT']].values
X1 = df[['RM']].values
y = df['MEDV'].values


from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
X1_std = sc_x.fit_transform(X1)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten() ## np.newaxis adds a new dimention


lr = LinearRegressionGD()
lr.fit(X1_std, y_std)


plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')

plt.show()


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 


lin_regplot(X1_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')

#plt.savefig('images/10_06.png', dpi=300)
plt.show()


print('Slope: get_ipython().run_line_magic(".3f'", " % lr.w_[1])")
print('Intercept: get_ipython().run_line_magic(".3f'", " % lr.w_[0])")


num_rooms_std = sc_x.transform(np.array([[5.0]])) # 5 rooms
price_std = lr.predict(num_rooms_std)

print("Price in $1000s: get_ipython().run_line_magic(".3f"", " % sc_y.inverse_transform(price_std))")


from sklearn.linear_model import LinearRegression


slr = LinearRegression()
slr.fit(X1, y)
y_pred = slr.predict(X1)
print('Slope: get_ipython().run_line_magic(".3f'", " % slr.coef_[0])")
print('Intercept: get_ipython().run_line_magic(".3f'", " % slr.intercept_)")


lin_regplot(X1, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')

#plt.savefig('images/10_07.png', dpi=300)
plt.show()


# adding a column vector of "ones"
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))

print('Slope: get_ipython().run_line_magic(".3f'", " % w[1])")
print('Intercept: get_ipython().run_line_magic(".3f'", " % w[0])")


http://rasbt.github.io/mlxtend/user_guide/regressor/LinearRegression/
    
https://www.statsmodels.org/stable/examples/index.html#regression


line_X = np.arange(3, 10, 1)
line_X[:, np.newaxis].shape
z = np.zeros(7)[:,np.newaxis].shape
line_X[:, np.newaxis]+z


from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(LinearRegression(), 
                         max_trials=50, 
                         min_samples=100, 
                         loss='absolute_loss', 
                         residual_threshold=5.0, 
                         random_state=0)

ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:,np.newaxis])

plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white', 
            marker='o', label='Inliers')

plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white', 
            marker='s', label='Outliers')

plt.plot(line_X, line_y_ransac, color='black', lw=2)   
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')

plt.show()


print('Slope: get_ipython().run_line_magic(".3f'", " % ransac.estimator_.coef_[0])")
print('Intercept: get_ipython().run_line_magic(".3f'", " % ransac.estimator_.intercept_)")


from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


import numpy as np
import scipy as sp

ary = np.array(range(100000))


get_ipython().run_line_magic("timeit", " np.linalg.norm(ary) # one of eight different matrix norms")


get_ipython().run_line_magic("timeit", " sp.linalg.norm(ary)")


get_ipython().run_line_magic("timeit", " np.sqrt(np.sum(ary**2))")


plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='cyan',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='green',
            label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: get_ipython().run_line_magic(".3f,", " test: %.3f' % (")
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: get_ipython().run_line_magic(".3f,", " test: %.3f' % (")
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01)

lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
print(lasso.coef_)


print('MSE train: get_ipython().run_line_magic(".3f,", " test: %.3f' % (")
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: get_ipython().run_line_magic(".3f,", " test: %.3f' % (")
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)

print('MSE train: get_ipython().run_line_magic(".3f,", " test: %.3f' % (")
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.05)

lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print('MSE train: get_ipython().run_line_magic(".3f,", " test: %.3f' % (")
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))


from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)

elanet.fit(X_train, y_train)
y_train_pred = elanet.predict(X_train)
y_test_pred = elanet.predict(X_test)

print('MSE train: get_ipython().run_line_magic(".3f,", " test: %.3f' % (")
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))


X = np.array([258.0, 270.0, 294.0, 
              320.0, 342.0, 368.0, 
              396.0, 446.0, 480.0, 586.0])[:, np.newaxis]

y = np.array([236.4, 234.4, 252.8, 
              298.6, 314.2, 342.2, 
              360.8, 368.0, 391.2, 390.8])


from sklearn.preprocessing import PolynomialFeatures

lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)


# fit linear features
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# fit quadratic features
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# plot results
plt.scatter(X, y, label='Training points')
plt.plot(X_fit, y_lin_fit, label='Linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='Quadratic fit')
plt.xlabel('Explanatory variable')
plt.ylabel('Predicted or known target values')
plt.legend(loc='upper left')

plt.tight_layout()

plt.show()


y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)


print('Training MSE linear: get_ipython().run_line_magic(".3f,", " quadratic: %.3f' % (")
        mean_squared_error(y, y_lin_pred),
        mean_squared_error(y, y_quad_pred)))
print('Training R^2 linear: get_ipython().run_line_magic(".3f,", " quadratic: %.3f' % (")
        r2_score(y, y_lin_pred),
        r2_score(y, y_quad_pred)))


X = df[['LSTAT']].values
y = df['MEDV'].values

regr = LinearRegression()

# create quadratic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# fit features
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))


# plot results
plt.scatter(X, y, label='Training points', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='Linear (d=1), $R^2=get_ipython().run_line_magic(".2f$'", " % linear_r2, ")
         color='blue', lw=2, 
         linestyle=':')

plt.plot(X_fit, y_quad_fit, 
         label='Quadratic (d=2), $R^2=get_ipython().run_line_magic(".2f$'", " % quadratic_r2,")
         color='red', lw=2,
         linestyle='-')

plt.plot(X_fit, y_cubic_fit, 
         label='Cubic (d=3), $R^2=get_ipython().run_line_magic(".2f$'", " % cubic_r2,")
         color='green', lw=2, 
         linestyle='--')

plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')

plt.show()


X = df[['LSTAT']].values
y = df['MEDV'].values

# transform features
X_log = np.log(X)
y_sqrt = np.log(np.sqrt(y))

# fit features
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]

regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# plot results
plt.scatter(X_log, y_sqrt, label='Training points', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='Linear (d=1), $R^2=get_ipython().run_line_magic(".2f$'", " % linear_r2, ")
         color='blue', 
         lw=2)

plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='lower left')

plt.tight_layout()
#plt.savefig('images/10_13.png', dpi=300)
plt.show()


from sklearn.tree import DecisionTreeRegressor

X = df[['LSTAT']].values
y = df['MEDV'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

sort_idx = X.flatten().argsort() 
# the array collapsed into one dimension and sort by an algorithm

lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')

plt.show()


X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000, 
                               criterion='mse', 
                               random_state=1, 
                               n_jobs=4)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE train: get_ipython().run_line_magic(".3f,", " test: %.3f' % (")
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: get_ipython().run_line_magic(".3f,", " test: %.3f' % (")
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


plt.scatter(y_train_pred,  
            y_train_pred - y_train, 
            c='steelblue',
            edgecolor='white',
            marker='o', 
            s=35,
            alpha=0.9,
            label='Training data')
plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='limegreen',
            edgecolor='white',
            marker='s', 
            s=35,
            alpha=0.9,
            label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.tight_layout()

#plt.savefig('images/10_15.png', dpi=300)
plt.show()


! python ../.convert_notebook_to_script.py --input ch10.ipynb --output ch10.py
