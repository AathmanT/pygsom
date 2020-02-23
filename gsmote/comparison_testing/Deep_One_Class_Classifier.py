from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from sklearn.mixture import GaussianMixture

from keras.datasets import mnist
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime


vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(56, 56, 3))
conv_out = vgg_conv.get_layer('block2_conv2').output
avgpool = GlobalAveragePooling2D()(conv_out)
model = Model(inputs=vgg_conv.input, outputs=avgpool)


gmm = GaussianMixture(n_components=1)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_ok = x_train[y_train == 1]  # 6742 筆
x_test = x_test[(y_test == 7) | (y_test == 1)]  # 1135 筆 "1", 1028 筆 "7"
y_test = y_test[(y_test == 7) | (y_test == 1)]


def reshape_x(x):
    new_x = np.empty((len(x), 56, 56))
    for i, e in enumerate(x):
        new_x[i] = cv2.resize(e, (56, 56))

    new_x = np.expand_dims(new_x, axis=-1)
    new_x = np.repeat(new_x, 3, axis=-1)
    return new_x


x_ok = reshape_x(x_ok)
x_test = reshape_x(x_test)

features = model.predict(x_ok)
gmm.fit(features)

OKscore = gmm.score_samples(features)
thred = OKscore.mean() - 3 * OKscore.std()

test_features = model.predict(x_test)
score = gmm.score_samples(test_features)

qwr = score[(y_test == 1)]
qwr2 = score[(score > thred)]
qwrr = score[(y_test == 7)]
qwrr2 = score[(score < thred)]

print('normal accuracy: %.2f' % (len(score[(y_test == 1) & (score > thred)]) / 1135))
print('abnormal accuracy: %.2f' % (len(score[(y_test == 7) & (score < thred)]) / 1028))
asfs=x_test[(y_test == 1) & (score > thred)]
for i in range(0,asfs.shape[0]):
    plt.imshow(asfs[i,:,:,:], cmap='brg')
    plt.show()

plt.scatter(range(len(x_test)), score, c=['skyblue' if x == 1 else 'pink' for x in y_test])
plt.plot(range(len(x_test)), [thred]*len(x_test), c='black')


plt.title("Deep One Class Classifier")
plt.savefig("output/one_class_classifier_"+datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")+".png",dpi=1200)