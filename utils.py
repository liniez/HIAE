import math
import h5py
import numpy as np
import scipy.io as sio

class AverageMeter(object):

    #-------- Computing and Storing the Avearge and Current Value --------
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Calculate_Size(object):
    
    #-------- Computing the Feature Map Size and Receptive Field Size -------- 
    def __init__(self, input_size, kernel_size):
        self.input_size = input_size
        self.kernel_size = kernel_size

    def featmap(self):
        featmap_size = [self.input_size]
        for layer in range(len(self.kernel_size)):
            featmap_size.append(math.floor(((featmap_size[2*layer]+2*int((self.kernel_size[layer]-1)/2)-(self.kernel_size[layer]-1)-1)/1)+1))
            featmap_size.append(math.floor(((featmap_size[2*layer+1]-2)/2)+1))
        return featmap_size

    def receptfield(self):
        receptfield_size = [1]
        for layer in range(0,len(self.kernel_size)):
            receptfield_size.append((receptfield_size[layer]-1)*2+self.kernel_size[layer])
        return receptfield_size

class Dataset(object):
    
    def __init__(self, fname, attrname, test_attrname = ['whole_brain'], test = False, sample_group_flag = False):
        self.fname = fname
        self.attrname = attrname
        self.test_attrname = test_attrname
        self.test = test

        if sample_group_flag:
            self.sample_group = np.array(h5py.File(self.fname,'r')[self.attrname[2]]) 
        else:
            self.prepare_data()

    
    def prepare_data(self):

        # Input Size: [Instances_Num*Channels_Num*Time_Series_Length]
        if self.test:
            self.test_data = np.array(h5py.File(self.fname,'r')[self.test_attrname[0]]) 
            self.test_data = np.expand_dims(self.test_data,1).transpose(2,1,0)

        else:
            self.train_data = np.array(h5py.File(self.fname,'r')[self.attrname[0]]) 
            self.train_data = np.expand_dims(self.train_data,1).transpose(2,1,0)
            self.val_data = np.array(h5py.File(self.fname,'r')[self.attrname[1]]) 
            self.val_data = np.expand_dims(self.val_data,1).transpose(2,1,0)
            

def savemat(fname, x):
    # x is a list object.
    varname = list('abcdefghijklmnopqrstuvwxyz')
    var = dict(zip(varname[0:len(x)],x))
    sio.savemat(fname,var)



