#-------- Initialization --------
import os
import time
import yaml
import argparse
import numpy as np

import torch
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from audtorch import metrics
from model import Autoencoder
from utils import AverageMeter, Dataset, savemat


parser = argparse.ArgumentParser(description = 'Autoencoder for fMRI Representation', epilog = 'Created by Lin Zhao')
parser.add_argument('--config', default='configs/config.yaml')
args = parser.parse_args()


class Autoencoder_Run():

    def __init__(self):
    
        global args
        with open(args.config) as f:
            config = yaml.load(f)
        for key in config:
            for k, v in config[key].items():
                setattr(args, k, v)        

        self.model_path = args.model_dir
            
    def main(self):

	#-------- Loading and Preparing Data--------
        model = torch.load('{}/BestModel.pkl'.format(self.model_path))
        model.cuda().double()

        # Training Data
        train_dataset = Dataset(args.fname,args.attrname,sample_group_flag = True)
        sample_group = train_dataset.sample_group

	#-------- Loss Function --------
        cudnn.benchmark = True		
        loss_func = torch.nn.MSELoss()


	#-------- Tesing --------
 
        for sub in sample_group:

            dataset = Dataset("{}/{}.mat".format(args.test_data_dir,int(sub[0])),args.attrname,test = True)
            subj_data = dataset.test_data
            data_set = Data.TensorDataset(torch.from_numpy(subj_data))
            loader = Data.DataLoader(dataset=data_set,batch_size=args.batch_size,shuffle=False,num_workers=2)            
            test_loss,test_corr, test_run_time,SA = self.test(loader, model, loss_func)
            print('Subj:{0}  Test_Loss: {1:.6f}, Test_Corr: {2:.6f}, Test_Time: {3:.6f}'.format(int(sub[0]),test_loss, test_corr, test_run_time))
            savemat(args.result_dir+'/'+str(int(sub[0]))+'.mat', SA)
            
    def test(self,test_loader, model, loss_func, layers = 4):

        Loss = AverageMeter()
        Corr = AverageMeter()
        Latent = {}
        Spatial_Activation = []
        for i in range(layers):
            Latent[i] = []

        model.eval() # Switch to evaluate mode
        start = time.time()
		
        for i,test_batch in enumerate(test_loader):
		
	    #-------- Data Preparing --------
            test_batch = test_batch[0].double()
            with torch.no_grad():
                test_batch = Variable(test_batch.cuda())
            
	    #-------- Computing Output, Loss and Correlation --------
            out = model(test_batch)
            loss = loss_func(out[0], test_batch)
            corr = metrics.functional.pearsonr(out[0],test_batch).sum()/test_batch.size(0)
            
            for i in range(layers):
                Latent[i].append(out[1][i])
           
            #-------- Recording Loss and Correlation --------
            Loss.update(loss.item())
            Corr.update(corr.item())

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Corr {corr.val:.3f} ({corr.avg:.3f})'.format(i, len(test_loader), loss=Loss, corr=Corr))

        end = time.time()
        test_time = end-start

        print('Test Time: {0:.3f}'.format(test_time))
        
        for i in range(layers):
            Spatial_Activation.append(np.concatenate(Latent[i],axis = 0))

        return Loss.avg, Corr.avg, test_time, Spatial_Activation


	
  
if __name__ == '__main__':
    AE = Autoencoder_Run()
    AE.main()
    



