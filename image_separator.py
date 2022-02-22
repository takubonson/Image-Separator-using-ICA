import imageio
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

image1 = np.asarray(imageio.imread("data/image1.png"))
image2 = np.asarray(imageio.imread("data/image2.png"))
image1 = image1.flatten().astype(np.float64)
image2 = image2.flatten().astype(np.float64)
image1 -= np.mean(image1)
image2 -= np.mean(image2)
sample_num = len(image1)
X = np.matrix([image1, image2])

sum = np.matrix([[0., 0.], [0., 0.]])
for i in range(sample_num):
    x = X[:, i]
    sum += x@x.T
sigma = sum/sample_num
D, E = LA.eig(sigma)
root_D = np.diag(D**(-1/2))
V = E@root_D@(E.T)


w = np.matrix([[1.], [0.]])
size_w = LA.norm(np.asarray(w).flatten())
w /= size_w
ratio_a = 1000
ratio_b = 1000
while ((abs(abs(ratio_a)-1)>0.001) or (abs(abs(ratio_b)-1)>0.001)):
    sum = np.matrix([[0.], [0.]])
    for j in range(sample_num):
        x = X[:, j]
        z = V@x
        sum += z @ (LA.matrix_power((w.T @ z), 3))
    E__ = sum / sample_num
    new_w = E__ - 3*w
    new_size_w = LA.norm(np.asarray(new_w).flatten())
    new_w/=new_size_w
    ratio_a = float(new_w[0]/w[0])
    ratio_b = float(new_w[1]/w[1])
    w = new_w
w1 = w
#白黒反転させないように
if (abs(float(w1[0]))<abs(float(w1[1])) and (float(w1[1]) < 0)):
    w1 *= -1.

w = np.matrix([[0.], [1.]])
size_w = LA.norm(np.asarray(w).flatten())
w /= size_w
ratio_a = 1000
ratio_b = 1000
while ((abs(abs(ratio_a)-1)>0.001) or (abs(abs(ratio_b)-1)>0.001)):
    sum = np.matrix([[0.], [0.]])
    for j in range(sample_num):
        x = X[:, j]
        z = V@x
        sum += z @ (LA.matrix_power((w.T @ z), 3))
    E__ = sum / sample_num
    new_w = E__ - 3*w
    new_size_w = LA.norm(np.asarray(new_w).flatten())
    new_w/=new_size_w
    ratio_a = float(new_w[0]/w[0])
    ratio_b = float(new_w[1]/w[1])
    w = new_w
    
w2 = w
#白黒反転させないように
if (abs(float(w2[0]))<abs(float(w2[1])) and (float(w2[1]) < 0)):
    w2 *= -1.

y1 = []
y2 = []
for m in range(sample_num):
    x = X[:, m]
    z = V@x
    y1.append(float(w1.T @ z))
    y2.append(float(w2.T @ z))
y1 = np.array(y1).reshape((512, 512))
y2 = np.array(y2).reshape((512, 512))

plt.imshow(y1, cmap="gray")
plt.imshow(y2, cmap="gray")

plt.show()
