from read_data import read_data
import numpy as np
import matplotlib.pyplot as plt

data = read_data()
print(data)

n = 1000

Mu_hat = np.mean(data, axis=1)
print(Mu_hat.shape)
rows, cols = data.shape
Z = np.zeros((rows, cols))
Z = np.subtract(data, Mu_hat[:, np.newaxis])

u, s, vh = np.linalg.svd(Z)

# fig, ax = plt.subplots(3, 4)
# for i in range(12):
#     image = np.reshape(u[:, i], (64, 64))
#     ax[i//4, i%4].imshow(image)
# fig.suptitle("Top 12 eigen images")

Y = np.transpose(u) @ np.subtract(data, Mu_hat[:, np.newaxis])
fig, ax = plt.subplots()
print(Y.shape)
for i in range(4):
    # print(Y[10, i])
    ax.plot(Y[0:10, i], label = f"This is letter {chr(ord('a') +  i)}")
fig.legend()
fig.suptitle("Projection coefficients vs eigenvector number")
ax.set_xlabel("Projection coefficient")
ax.set_ylabel("Eigenvector number")
plt.show()

print('u', u.shape, Y.shape)
fig, ax = plt.subplots(3, 2)
img = np.zeros(4096)
for idx, i in enumerate([1, 5, 10, 15, 20, 30]):
    ax[idx//2, idx%2].imshow(np.add((u[0, 0:i] @ Y[0:i, 0]), Mu_hat).reshape(64, 64))

plt.show()

plt.imshow(data[:, 0].reshape(64, 64))
plt.show()


