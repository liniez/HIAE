import torch
import torch.nn as nn
from utils import Calculate_Size

class conv_block(nn.Module):

    def __init__(self, nin, nout, kernel_size):
        super(conv_block, self).__init__()
        self.conv = nn.Conv1d(nin, nout, kernel_size, padding = int((kernel_size-1)/2))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(nout)
        
    def forward(self, x):

        x = self.conv(x)
        x = self.relu(x)
        out = self.bn(x)

        return x


class deconv_block(nn.Module):

    def __init__(self, nin, nout, kernel_size):
        super(deconv_block, self).__init__()
        self.conv = nn.Conv1d(nin, nout, kernel_size, padding = int((kernel_size-1)/2))
        self.tanh = nn.Tanh() 
        self.bn = nn.BatchNorm1d(nout)
        
    def forward(self, x):

        x = self.conv(x)
        x = self.tanh(x)
        out = self.bn(x)

        return x

class kernel(nn.Module):

    def __init__(self, channel, input_size, latent_size):
        super(kernel, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.channel = channel
        self.fcen = nn.Linear(self.input_size*self.channel,self.latent_size)
        self.fcde = nn.Linear(self.latent_size,self.input_size*self.channel)      
        self.tanh = nn.Tanh() 
        
    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.fcen(x)        
        latent = self.tanh(x)
        x = self.fcde(latent)
        x = self.tanh(x)   
        x = x.view(x.size(0),self.channel,self.input_size)

        return x, latent



class Autoencoder(nn.Module):

    def __init__(self, args):
        super(Autoencoder, self).__init__()

        self.layers = len(args.kernel_size)

        
        calsize = Calculate_Size(args.task_dict[args.task],args.kernel_size)
        outsize = calsize.featmap()

        self.conv1 = conv_block(1,args.kernel_num[0],args.kernel_size[0])
        self.conv2 = conv_block(args.kernel_num[0],args.kernel_num[1],args.kernel_size[1])
        self.conv3 = conv_block(args.kernel_num[1],args.kernel_num[2],args.kernel_size[2])
        self.conv4 = conv_block(args.kernel_num[2],args.kernel_num[3],args.kernel_size[3])
        
        self.kernel1 = kernel(args.kernel_num[0],outsize[1],args.latent_size[0])
        self.kernel2 = kernel(args.kernel_num[1],outsize[3],args.latent_size[1])
        self.kernel3 = kernel(args.kernel_num[2],outsize[5],args.latent_size[2])
        self.kernel4 = kernel(args.kernel_num[3],outsize[7],args.latent_size[3])

        self.deconv1 = deconv_block(args.kernel_num[3], args.kernel_num[2], args.kernel_size[3]) 
        self.deconv2 = deconv_block(args.kernel_num[2]*2, args.kernel_num[1], args.kernel_size[2]) 
        self.deconv3 = deconv_block(args.kernel_num[1]*2, args.kernel_num[0], args.kernel_size[1]) 
        self.deconv4 = deconv_block(args.kernel_num[0]*2, 1, args.kernel_size[0]) 

        self.maxpool = nn.MaxPool1d(2,2)
        self.upsample = nn.Upsample(scale_factor=2)
        

    def forward(self, x):
    
        featmap = []
        
        c1 = self.conv1(x)
        p1 = self.maxpool(c1)

        c2 = self.conv2(p1)
        p2 = self.maxpool(c2)
        
        c3 = self.conv3(p2)
        p3 = self.maxpool(c3) 
       
        c4 = self.conv4(p3)
   
        k1,l1= self.kernel1(c1)
        k2,l2 = self.kernel2(c2)
        k3,l3 = self.kernel3(c3)
        k4,l4 = self.kernel4(c4)

        dc1 = self.deconv1(k4)

        up2 = self.upsample(dc1)
        merge2 = torch.cat([up2,k3],1)
        dc2 = self.deconv2(merge2)
        
        up3 = self.upsample(dc2)
        merge3 = torch.cat([up3,k2],1)
        dc3 = self.deconv3(merge3)
        
        up4 = self.upsample(dc3)
        merge4 = torch.cat([up4,k1],1)
        dc4 = self.deconv4(merge4)

        featmap.append(l1.cpu().detach().numpy())
        featmap.append(l2.cpu().detach().numpy()) 
        featmap.append(l3.cpu().detach().numpy())
        featmap.append(l4.cpu().detach().numpy())

        return dc4, featmap


