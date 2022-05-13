from yolo3.model import yolo_body
from keras.layers import Input
from yolo import YOLO
from PIL import Image
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
yolo = YOLO()

while True:
    img = input('Input imagename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error!')
        continue
    else:
        rel_image = yolo.detect_image(image)
        rel_image.show()
yolo.close_session()

