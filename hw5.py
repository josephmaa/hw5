import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
input = rng.multivariate_normal(np.zeros(2), cov=np.identity(2), size=(2, 1000))[..., 0]
# print(input.shape)
# print(input[1])
# print(input)
# print(input.shape)


# fig, ax = plt.subplots()
# ax.scatter(input[0, :], input[1, :])
# ax.set_ylabel('W2')
# ax.set_xlabel('W1')
# ax.set_title("Random multivariate distribution")
# plt.axis('equal')

# plt.show()

r_x = np.array([[2, -1.2], [-1.2, 1]])
w, v = np.linalg.eig(r_x)
x_p = np.matmul(np.sqrt(np.abs(v)), input)
print(x_p.shape)
print(w)

# fig, ax = plt.subplots()
# ax.scatter(x_p[0, :], x_p[1, :])
# ax.set_title('X_i tilde')
# ax.set_adjustable
# fig.show()
# plt.axis('equal')
# plt.show()

X_i = v @ x_p
# print(X_i.shape)

# fig, ax = plt.subplots()
# ax.scatter(X_i[0, :], X_i[1, :])
# ax.set_title('X_i')
# ax.set_adjustable
# fig.show()
# plt.axis('equal')
# plt.show()

mean = 1/1000 * np.sum(X_i, axis=1)
R_hat = 0
for i in range(1000):
    R_hat += (X_i[:, i]-mean)[:, np.newaxis]@np.transpose(X_i[:, i]-mean)[np.newaxis, :]
R_hat *= 1/(1000-1)
print(R_hat)

eigenvalues, eigenvectors = np.linalg.eig(R_hat)
X_i_tilde = np.transpose(eigenvectors) @ input
W = np.sqrt(np.abs(eigenvectors)) @ np.transpose(eigenvectors) @ X_i_tilde
# print(W)

fig, ax = plt.subplots()
ax.scatter(X_i_tilde[0, :], X_i_tilde[1, :])
ax.set_title('X_i_tilde')
ax.set_adjustable
fig.show()
plt.axis('equal')

fig, ax = plt.subplots()
ax.scatter(W[0, :], W[1, :])
ax.set_title('W')
ax.set_adjustable
fig.show()
plt.axis('equal')
plt.show()

def estimate_covariance(vector: np.array, n: int) -> np.array:
    mean = 1/n * np.sum(W, axis=1)
    cov = 0
    for i in range(n):
        cov += (W[:, i]-mean)[:, np.newaxis]@np.transpose(W[:, i]-mean)[np.newaxis, :]
    cov *= 1/(n-1)
    return cov

R_w = estimate_covariance(W, 1000)
print(R_w)