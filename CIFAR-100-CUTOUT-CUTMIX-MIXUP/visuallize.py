import pickle as p
import numpy as np
from PIL import Image
import random


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='bytes')
        # 以字典的形式取出数据
        X = datadict[b'data']
        Y = datadict[b'fine_labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        print(Y.shape)
        return X, Y


def visualize(method, imgX, i):
    imgs = imgX[i].copy()
    loc_x = random.randint(0, 16)
    loc_y = random.randint(0, 16)
    height = random.randint(6, 10)
    width = random.randint(6, 10)
    lam = random.randint(2, 8) * 0.1

    if method == 'cutout':
        imgs[:, loc_x:loc_x + width, loc_y:loc_y + height] = 0
    elif method == 'cutmix':
        imgs[:, loc_x:loc_x + width, loc_y:loc_y + height] = imgX[random.randint(50, 100)][:, loc_x:loc_x + width,
                                                             loc_y:loc_y + height]
    elif method == 'mixup':
        imgs = (imgs * lam + imgX[random.randint(50, 100)] * (1 - lam)).astype(int)
    else:
        imgs = imgs

    imgs = np.clip(imgs, 0, 255)

    i0 = Image.fromarray(imgs[0]).convert('L')
    i1 = Image.fromarray(imgs[1]).convert('L')
    i2 = Image.fromarray(imgs[2]).convert('L')
    img = Image.merge("RGB", (i0, i1, i2))
    name = "img" + str(i) + "_" + method + ".png"
    img.save("./pic1/" + name, "png")
    print(method + '_图片存储成功！')


if __name__ == "__main__":
    imgX, imgY = load_CIFAR_batch("./data/cifar-100-python/test")
    with open('pic1/img_label.txt', 'a+') as f:
        for i in range(imgY.shape[0]):
            f.write('img' + str(i) + ' ' + str(imgY[i]) + '\n')

    for i in range(10):
        for method in ['baseline', 'cutout', 'cutmix', 'mixup']:
            visualize(method, imgX, i)
