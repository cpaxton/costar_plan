
import h5py
import os
import io
import glob
from PIL import Image

import numpy as np
import json
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        print(indexes)

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        #Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_Ids):
        """ Generates data containing batch_size samples

        # Arguments

        list_Ids: a list of file paths to be read
        """

        def JpegToNumpy(jpeg):
            stream = io.BytesIO(jpeg)
            im = Image.open(stream)
            return np.asarray(im, dtype=np.uint8)

        def ConvertImageListToNumpy(data, format='numpy', data_format='NHWC'):
            """ Convert a list of binary jpeg or png files to numpy format.

            # Arguments

            data: a list of binary jpeg images to convert
            format: default 'numpy' returns a 4d numpy array,
                'list' returns a list of 3d numpy arrays
            """
            length = len(data)
            imgs = []
            for raw in data:
                img = JpegToNumpy(raw)
                if data_format == 'NCHW':
                    img = np.transpose(img, [2, 0, 1])
                imgs.append(img)
            if format == 'numpy':
                imgs = np.array(imgs)
            return imgs
        # Initialization
        X = {}
        y = []


        # Generate data
        for i, ID in enumerate(list_Ids):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            data = h5py.File(ID, 'r')
            rgb_images = list(data['depth_image'])
            rgb_images = ConvertImageListToNumpy(np.squeeze(rgb_images), format='numpy')
            #indices = [0]
            indices = [0]+ list(np.random.randint(1,len(rgb_images),10))
            print(indices)
            X['image'] = (rgb_images[indices])
            X['pose'] = np.array(data['pose'][indices])
            #X.append(np.array(data['pose'])[indices])

            # Store class
            for items in list(data['all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json']):
                json_data = json.loads(items.decode('UTF-8'))
                y.append(np.array(json_data['camera_rgb_frame']))
        y = np.array(y)

        return X, y


filenames = glob.glob(os.path.expanduser("~/JHU/LAB/Projects/costar_block_stacking_dataset_v0.3/*success.h5f"))
#print(filenames)
training_generator = DataGenerator(filenames,batch_size=1)
# X,y=training_generator.__getitem__(1)
# print(X.keys())
# print(X['image'].shape)