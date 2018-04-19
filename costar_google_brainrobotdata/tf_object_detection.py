"""Configure and run tf object detection API training

Author: Andrew Hundt <ATHundt@gmail.com>

Works with [tf object detection
API](https://github.com/tensorflow/models/blob/master/research/object_detection),
for details see README.md
"""
import tensorflow as tf
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras._impl.keras.utils.data_utils import _hash_file

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


def main(argv):
    get_file(
        'ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
        extract=True)
    get_file(
        'faster_rcnn_nas_coco_2018_01_28.tar.gz',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz',
        extract=True)
    get_file(
        'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz',
        extract=True)



if __name__ == '__main__':
    tf.app.run(main=main)
