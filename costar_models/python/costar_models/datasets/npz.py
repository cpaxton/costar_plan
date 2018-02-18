from __future__ import print_function

import numpy as np
import os


class NpzDataset(object):
    '''
    Write out NPZ datasets one file at a time, with information in filenames
    for easy access and data aggregation.

    This is a fairly slow way to do things but prevents a lot of the problems
    with massive amounts of data being held in memory.
    '''

    def __init__(self, name):
        '''
        Create a folder to hold different archives in
        '''
        self.name = os.path.expanduser(name)
        try:
            os.mkdir(name)
        except OSError:
            pass

    def write(self, example, i, r):
        '''
        Write an example out to disk.
        '''
        if r > 0.:
            status = "success"
        else:
            status = "failure"
        filename = "example%06d.%s.npz"%(i,status)
        filename = os.path.join(self.name, filename)
        np.savez(filename, **example)

    def load(self,success_only=False):
        '''
        Read a whole set of data files in. This part could definitely be
        faster/more efficient.

        Parameters:
        -----------
        success_only: exclude examples without the "success" tag in their
                      filename when loading. Good when learning certain types
                      of models.

        Returns:
        --------
        data: dict of features and other information.
        '''
        files = os.listdir(self.name)
        files.sort()
        data = {}
        i = 0
        for f in files:
            if not f[0] == '.':
                i += 1
                print("%d:"%i, f)
                if success_only:
                    name = f.split('.')
                    if name[0] == 'failure':
                        continue
                fdata = np.load(os.path.join(self.name,f))
                for key, value in fdata.items():
                    if key not in data:
                        data[key] = value
                    if value.shape[0] == 0:
                        continue
                    data[key] = np.concatenate([data[key],value],axis=0)
        return data

    def preprocess(self, train=0.6, val=0.2):
        '''
        TODO(cpaxton) 
        Create train/val/test splits
        '''
        assert train+val <= 1.0 and train+val > 0.
        test = 1.0 - train - val
