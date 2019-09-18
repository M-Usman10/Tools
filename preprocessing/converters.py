import glob

import h5py
import numpy as np
import skimage.io as io


class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)
    
    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype
    
    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)
        
    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """
    def __init__(self, datapath, dataset, shape, dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0
        
        with h5py.File(self.datapath, mode='w') as h5f:
            self.dset = h5f.create_dataset(
                dataset,
                shape=(0, ) + shape,
                maxshape=(None, ) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len, ) + shape)
    
    def append(self, values):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1,) + self.shape)
            dset[self.i] = [values]
            self.i += 1
            h5f.flush()

def saveAsHdf5(inputPath,OutputFile,preprocess=None,dataName='images',inpFormat='.jpg',reader=io.imread,shape=(224,224,224)):
    """
        Saves all files in inputPath that ends with inpFormat to OutputFile
    """
    hdf5_store = HDF5Store(OutputFile,dataName, shape=shape)
    files=glob.glob(inputPath+'/*'+inpFormat)
    if preprocess==None:
        for i in files:
            file=reader(i)
            hdf5_store.append(file)
    else:
        for i in files:
            file=reader(i)
            file=preprocess(file)
            hdf5_store.append(file)

def dense_box_to_file_box(inpPath,outPath):
    """
       Convert Dense Box Format (1 file for all images) used in my Object Detection work to standard File format,
       one file per image
       Params:
       path: path of dense format file
     """
    from .readers import read_boxes
    import os
    boxes,names,img_names=read_boxes(inpPath,allowed=None)
    for i in range(len(names)):
        with open(os.path.join(outPath ,os.path.basename(img_names[i])+'.txt'),'w+') as file:
            curr_boxes,curr_names=boxes[i],names[i]
            for j in range(len(curr_boxes)):
                file.write(curr_names[j]+','+','.join(curr_boxes[j].astype(str))+'\n')
