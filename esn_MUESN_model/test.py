import numpy as np

rng = np.random.default_rng()


N_x = 100
N_u = 14
input_scale = 0.75
mask = np.random.rand(N_x, N_u)
mask[mask < 0.1] = 0
mat = np.random.randint(0, 2, (N_x, N_u)) * 2 - 1
W_in = np.multiply(mat, mask)
W_in_np = rng.uniform(-input_scale, input_scale, (N_x, N_u))*input_scale

np.set_printoptions(threshold=np.inf)
#print(mask)
#print(mask.shape)
#print(mat)
#print(mat.shape)
#print(W_in)
print(W_in_np)