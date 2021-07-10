#-------- Initialization --------
import os
import time
import yaml
import argparse

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from audtorch import metrics
from model import Autoencoder
from utils import AverageMeter, Dataset
from pytorchtools import EarlyStopping


parser = argparse.ArgumentParser(description = 'Autoencoder for fMRI Representation', epilog = 'Created by Lin Zhao')
parser.add_argument('--config', default='configs/config.yaml')
args = parser.parse_args()



class AutoEncoder():

    def __init__(self):
    
        global args
        with open(args.config) as f:
            config = yaml.load(f)
        for key in config:
            for k, v in config[key].items():
                setattr(args, k, v)        

        self.model_path = args.model_dir

    def main(self):

        model = Autoencoder(args)
        model.cuda()
        model = model.double()

        writer = SummaryWriter('runs')


	#-------- Loading and Preparing Data--------
        dataset = Dataset(args.fname,args.attrname)

        train_data,val_data = dataset.train_data, dataset.val_data
		
        train_data = torch.from_numpy(train_data)
        train_set = Data.TensorDataset(train_data)
        train_loader = Data.DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True,num_workers=2) 
		
        val_data = torch.from_numpy(val_data)
        val_set = Data.TensorDataset(val_data)
        val_loader = Data.DataLoader(dataset=val_set,batch_size=args.batch_size,shuffle=True,num_workers=2)
		

        print('# Model Total Parameters:', sum(param.numel() for param in model.parameters()))

	#-------- Optimizer and Loss Function --------	
        cudnn.benchmark = True	
        optimizer = torch.optim.Adam(model.parameters())
        loss_func = torch.nn.MSELoss()
        early_stopping = EarlyStopping(patience = 5, verbose = True)


	#-------- Training and Tesing --------
        for epoch in range(args.epochs):

            train_loss, train_corr = self.train(train_loader, model, loss_func, optimizer, epoch)
            val_loss, val_corr, val_run_time = self.validate(val_loader, model, loss_func, optimizer, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Correlation/train', train_corr, epoch)
            writer.add_scalar('Correlation/val', val_corr, epoch)


            print('Epoch: {0} Loss: {1:.6f}, Corr: {2:.6f}, Val_Loss: {3:.6f}, Val_Corr: {4:.6f}'.format(epoch, train_loss, train_corr, val_loss, val_corr))
            

            early_stopping(val_loss,model,'{}/BestModel.pkl'.format(self.model_path))
            if early_stopping.early_stop:
                print('Early Stopping')
                writer.close()
                break
        writer.close()

    def train(self,train_loader, model, loss_func, optimizer, epoch):

        Batch_Time = AverageMeter()
        Loss = AverageMeter()
        Corr = AverageMeter()

		# Switch to train mode
        model.train() 
        
        start = time.time()
		
        for i,train_batch in enumerate(train_loader):
            batch_start = time.time()	
            #-------- Data Preparing --------
            train_batch = Variable(train_batch[0].cuda())
            train_batch = train_batch.double()

            #-------- Computing Output, Loss and Correlation --------
            out = model(train_batch)
            loss = loss_func(out[0], train_batch)
            corr = metrics.functional.pearsonr(out[0],train_batch).sum()/train_batch.size(0)

            #-------- Recording Loss and Correlation --------
            Loss.update(loss.item())
            Corr.update(corr.item())
			
            #-------- Backpropagation and Updating Weights --------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Batch_Time.update(time.time()-batch_start)
			
            if i % args.print_freq == 0:
                current = time.time()-start
                print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {3:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Corr {corr.val:.3f} ({corr.avg:.3f})'.format(epoch, i, len(train_loader), current, batch_time=Batch_Time,loss=Loss, corr=Corr))
        Epoch_Time = time.time()-start
        print('Epoch Time: {0:.3f}'.format(Epoch_Time))	
        torch.save(model, '{}AE_Epoch_{:0>2d}.pkl'.format(self.model_path, epoch))
        return Loss.avg, Corr.avg


    def validate(self,val_loader, model, loss_func, optimizer, epoch):

        Loss = AverageMeter()
        Corr = AverageMeter()
		
        model.eval() # Switch to evaluate mode
        start = time.time()
		
        for i,val_batch in enumerate(val_loader):
		
	    #-------- Data Preparing --------
            val_batch = val_batch[0].double()
            with torch.no_grad():
                val_batch = Variable(val_batch.cuda())
            
	    #-------- Computing Output, Loss and Correlation --------
            out = model(val_batch)
            loss = loss_func(out[0], val_batch)
            corr = metrics.functional.pearsonr(out[0],val_batch).sum()/val_batch.size(0)

            #-------- Recording Loss and Correlation --------
            Loss.update(loss.item())
            Corr.update(corr.item())

            if i % args.print_freq == 0:
                print('Val: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Corr {corr.val:.3f} ({corr.avg:.3f})'.format(i, len(val_loader), loss=Loss, corr=Corr))

        end = time.time()
		
        val_time = end-start
        print('Validation Time: {0:.3f}'.format(val_time))	
			
        return Loss.avg, Corr.avg, val_time
		
			
  
if __name__ == '__main__':
    AE = AutoEncoder()
    AE.main()
    

