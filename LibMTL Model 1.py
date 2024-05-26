import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


#Create simulation data
def create_simulated_data(n_samples=100, n_features=20, n_tasks=5, noise=0.1):
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features, n_tasks)
    Y = X @ true_coef + noise * np.random.randn(n_samples, n_tasks)
    return X, Y


X, Y = create_simulated_data()

# Define parameters
lam1_seq = 10 ** np.arange(1, -5, -1, dtype=float)
lam2 = 0
nfolds = 5

# Cross-validation
kf = KFold(n_splits=nfolds)
cv_errors = []

for lam1 in lam1_seq:
    fold_errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model = Ridge(alpha=lam1)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        fold_errors.append(mean_squared_error(Y_test, Y_pred))

    cv_errors.append(np.mean(fold_errors))

cv_errors = np.array(cv_errors)

# The minimum cross-validation error corresponds to lam1

min_index = np.argmin(cv_errors)
best_lam1 = lam1_seq[min_index]

print("Lam1 sequence:", lam1_seq)
print("Lam2:", lam2)
print(f"Estimated lam1: {best_lam1}")

# Plot cross-validation error

plt.plot(np.log10(lam1_seq), cv_errors, marker='o')
plt.xlabel('log10(Lam1)')
plt.ylabel('CV Error')
plt.title('CV Error across Lam1 sequence')
plt.show()

