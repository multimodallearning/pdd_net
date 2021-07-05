from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import torch.nn as nn
import torch.optim as optim

import time

import torch.nn.functional as F
#import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from torch.utils.checkpoint import checkpoint

from scipy.ndimage.interpolation import zoom as zoom
from scipy.ndimage.interpolation import map_coordinates as mapcoord





list_train = torch.Tensor([2,3,5,6,8,9,21,22,24,25,27,28,30,31,33,34,36,37,39,40]).long()


B = 20#len(list_train)
H = 192; W = 160; D = 256;
H2 = H//3; W2 = W//3; D2 = D//3###



o_m = H//3
o_n = W//3
o_o = D//3

#grid for features
ogrid_xyz = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,o_m,o_n,o_o)).view(1,1,-1,1,3).cuda()

alpha_data = torch.Tensor([1.1337,  0.0761,  1.0533, -0.7469,  0.0728,  1.1337])
class GridNet(nn.Module):
    def __init__(self,grid_x,grid_y,grid_z):
        super(GridNet, self).__init__()
        self.params = nn.Parameter(torch.randn(1,3,grid_x,grid_y,grid_z))

    def forward(self):
        return self.params


def countParameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class OBELISK(nn.Module):
    def __init__(self):

        super(OBELISK, self).__init__()
        channels = 24#16
        self.offsets = nn.Parameter(torch.randn(2,channels*2,3)*0.05)
        self.layer0 = nn.Conv3d(1, 4, 5, stride=2, bias=False, padding=2)
        self.batch0 = nn.BatchNorm3d(4)

        self.layer1 = nn.Conv3d(channels*8, channels*4, 1, bias=False, groups=1)
        self.batch1 = nn.BatchNorm3d(channels*4)
        self.layer2 = nn.Conv3d(channels*4, channels*4, 3, bias=False, padding=1)
        self.batch2 = nn.BatchNorm3d(channels*4)
        self.layer3 = nn.Conv3d(channels*4, channels*1, 1)


    def forward(self, input_img):
        img_in = F.avg_pool3d(input_img,3,padding=1,stride=2)
        img_in = F.relu(self.batch0(self.layer0(img_in)))
        sampled = F.grid_sample(img_in,ogrid_xyz + self.offsets[0,:,:].view(1,-1,1,1,3),align_corners=True).view(1,-1,o_m,o_n,o_o)
        sampled -= F.grid_sample(img_in,ogrid_xyz + self.offsets[1,:,:].view(1,-1,1,1,3),align_corners=True).view(1,-1,o_m,o_n,o_o)
    
        x = F.relu(self.batch1(self.layer1(sampled)))
        x = F.relu(self.batch2(self.layer2(x)))
        features = self.layer3(x)
        return features


def fit_sparse2dense(pred_xyz,grid_xyz,soft_cost,H,W,D,disp_range=0.4,displacement_width=15,lambda_weight=0.15):

    sample_loss = np.zeros(100)
    regular_loss = np.zeros(100)

    t0 = time.time()
    with torch.enable_grad():
        net = GridNet(H2,W2,D2)
        net.params.data = pred_xyz.permute(0,4,1,2,3).detach()#+torch.randn_like(pred_xyz.permute(0,4,1,2,3))*0.05#F.interpolate(dense_grid+torch.randn_like(dense_grid)*0.01,size=(64,64),mode='bilinear').detach()
        net.cuda()
        avg5 = nn.AvgPool3d((3,3,3),stride=(1,1,1),padding=(1,1,1)).cuda()


        optimizer = optim.Adam(net.parameters(), lr=0.02)

        for iter in range(50):
            optimizer.zero_grad()
            fitted_grid = (avg5(avg5(net())))
            sampled_net = F.grid_sample(fitted_grid,grid_xyz,align_corners=True).permute(2,0,3,4,1)/disp_range
            #print(sampled_net.size())soft_cost
            cost3d = soft_cost.view(-1,1,displacement_width,displacement_width,displacement_width)
            sampled_cost = F.grid_sample(cost3d,sampled_net,align_corners=True)
            loss = (-sampled_cost).mean()
            sample_loss[iter] = (-sampled_cost).mean().item()
            reg_loss = lambda_weight*((fitted_grid[0,:,:,1:,:]-fitted_grid[0,:,:,:-1,:])**2).mean()+            lambda_weight*((fitted_grid[0,:,1:,:,:]-fitted_grid[0,:,:-1,:,:])**2).mean()+            lambda_weight*((fitted_grid[0,:,:,:,1:]-fitted_grid[0,:,:,:,:-1])**2).mean()
            regular_loss[iter] = reg_loss.item()

            (reg_loss+loss).backward()

            optimizer.step()
    dense_flow_fit = F.interpolate(fitted_grid.detach(),size=(H,W,D),mode='trilinear',align_corners=True)

    return dense_flow_fit

def warpImageFloat(img_input,def_x2):
    B, C, H, W, D = img_input.size() #expects 'depth' in 'channel'
    grid_xyz = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D),align_corners=True).to(def_x2.device)
    x_grid = grid_xyz + def_x2.permute(0,2,3,4,1)
    output_im = F.grid_sample(img_input,x_grid,align_corners=True)#.cpu()
    return output_im



class deeds(nn.Module):
    def __init__(self):

        super(deeds, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([1,.1,1,1,.1,1]))#.cuda()

        self.pad1 = nn.ReplicationPad3d(3)#.cuda()
        self.avg1 = nn.AvgPool3d(3,stride=1)#.cuda()
        self.max1 = nn.MaxPool3d(3,stride=1)#.cuda()
        self.pad2 = nn.ReplicationPad3d(2)#.cuda()##



    def forward(self, feat00,feat50,grid_xyz,grid_size,displacement_width,shift_xyz):
        
        #deeds correlation layer (slightly unrolled)
        deeds_cost = torch.zeros(1,grid_size**3,displacement_width,displacement_width,displacement_width).cuda()
        xyz8 = grid_size**2
        for i in range(grid_size): 
            moving_unfold = F.grid_sample(feat50,grid_xyz[:,i*xyz8:(i+1)*xyz8,:,:,:] + shift_xyz,padding_mode='border',align_corners=True)
            fixed_grid = F.grid_sample(feat00,grid_xyz[:,i*xyz8:(i+1)*xyz8,:,:,:],align_corners=True)
            deeds_cost[:,i*xyz8:(i+1)*xyz8,:,:,:] = self.alpha[1]+self.alpha[0]*torch.sum(torch.pow(fixed_grid-moving_unfold,2),1).view(1,-1,displacement_width,displacement_width,displacement_width)

        # remove mean (not really necessary)
        #deeds_cost = deeds_cost.view(-1,displacement_width**3) - deeds_cost.view(-1,displacement_width**3).mean(1,keepdim=True)[0]
        deeds_cost = deeds_cost.view(1,-1,displacement_width,displacement_width,displacement_width)
    
        # approximate min convolution / displacement compatibility
        #cost = 1*deeds_cost+0#
        cost = self.avg1(self.avg1(-self.max1(-self.pad1(deeds_cost))))
   
        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2,3,4,0,1).view(1,displacement_width**3,grid_size,grid_size,grid_size)
        cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0,2,3,4,1).view(1,-1,displacement_width,displacement_width,displacement_width)
        
        # second path
        cost = self.alpha[4]+self.alpha[2]*deeds_cost+self.alpha[3]*cost_avg
        cost = self.avg1(self.avg1(-self.max1(-self.pad1(cost))))
        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2,3,4,0,1).view(1,displacement_width**3,grid_size,grid_size,grid_size)
        cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0,2,3,4,1).view(grid_size**3,displacement_width**3)
        #cost = alpha[4]+alpha[2]*deeds_cost+alpha[3]*cost.view(1,-1,displacement_width,displacement_width,displacement_width)
        #cost = avg1(avg1(-max1(-pad1(cost))))
        
        #probabilistic and continuous output
        cost_soft = F.softmax(-self.alpha[5]*cost_avg,1)
#        pred_xyz = torch.sum(F.softmax(-5self.alpha[2]*cost_avg,1).unsqueeze(2)*shift_xyz.view(1,-1,3),1)
        pred_xyz = torch.sum(cost_soft.unsqueeze(2)*shift_xyz.view(1,-1,3),1)



        return cost_soft,pred_xyz

    
def pdd(input_img_fixed,input_img_moving,output_field_npz,input_model_pth):
    
    displacement_width = 11; disp_range = 0.25
    disp_range = 0.25#0.25
    displacement_width = 11#11#17
    shift_xyz = F.affine_grid(disp_range*torch.eye(3,4).unsqueeze(0),(1,1,displacement_width,displacement_width,displacement_width),align_corners=True).view(1,1,-1,1,3).cuda()

    #_,_,H,W,D = img00.size()
    grid_size = 29#32#25#30
    grid_xyz = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,grid_size,grid_size,grid_size),align_corners=True).view(1,-1,1,1,3).cuda()


    displacement_width2 = 11; disp_range2 = 0.15
    shift_xyz2 = F.affine_grid(disp_range2*torch.eye(3,4).unsqueeze(0),(1,1,displacement_width2,displacement_width2,displacement_width2),align_corners=True).view(1,1,-1,1,3).cuda()

    grid_size2 = 39#25#30
    grid_xyz2 = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,grid_size2,grid_size2,grid_size2),align_corners=True).view(1,-1,1,1,3).cuda()



    
    net = OBELISK()
    net.load_state_dict(torch.load(input_model_pth))
    net.eval()
    net.cuda()
    reg = deeds()
    reg.alpha.data = alpha_data
    reg.eval()
    reg.cuda()
    
    reg2 = deeds()
    reg2.alpha.data = alpha_data
    reg2.eval()
    reg2.cuda()

    
#imgs = torch.zeros(B,1,H,W,D)
#segs = torch.zeros(B,H,W,D).long()
#label_select = torch.Tensor([0,1,2,3,4,5,6,7,0,0,8,9]).long()

    img00 = (torch.from_numpy(nib.load(input_img_fixed).get_data())/500.0).unsqueeze(0).unsqueeze(0).float().cuda()
    img50 = (torch.from_numpy(nib.load(input_img_moving).get_data())/500.0).unsqueeze(0).unsqueeze(0).float().cuda()
    
    
    
    with torch.no_grad():
        
        feat00 = net(img00)#net(img00)# #00 is fixed
        feat50 = net(img50)#net(img50)# #50 is moving
        cost_soft,pred_xyz =  reg(feat00,feat50,grid_xyz,grid_size,displacement_width,shift_xyz)#
        pred_xyz = 1*pred_xyz.view(1,grid_size,grid_size,grid_size,3)
        dense_flow_fit = fit_sparse2dense(pred_xyz,grid_xyz,cost_soft,H,W,D,disp_range,displacement_width,1.5)
        pred_xyz_full = F.interpolate(dense_flow_fit,(H,W,D))

        warped50_ = warpImageFloat(img50,pred_xyz_full)

        feat50_ = net(warped50_)
        cost_soft,pred_xyz =  reg2(feat00,feat50_,grid_xyz2,grid_size2,displacement_width2,shift_xyz2)#
        pred_xyz = pred_xyz.view(1,grid_size2,grid_size2,grid_size2,3)
        dense_flow_fit = fit_sparse2dense(pred_xyz,grid_xyz2,cost_soft,H,W,D,disp_range2,displacement_width2,1.5)
        pred_xyz_full_ = F.interpolate(dense_flow_fit,(H,W,D))
        combined = warpImageFloat(pred_xyz_full,pred_xyz_full_)+pred_xyz_full_

        def_xyz_full = combined + F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D),align_corners=True).permute(0,4,1,2,3).cuda() 
        x0,y0,z0 = torch.meshgrid(torch.arange(H).float(),torch.arange(W).float(),torch.arange(D).float())
        x0 = x0.data.numpy()
        y0 = y0.data.numpy()
        z0 = z0.data.numpy()

        x = (def_xyz_full[0,2,:,:,:].data.cpu().numpy()+1)*(H-1)/2
        y = (def_xyz_full[0,1,:,:,:].data.cpu().numpy()+1)*(W-1)/2
        z = (def_xyz_full[0,0,:,:,:].data.cpu().numpy()+1)*(D-1)/2

        x1 = zoom(x-x0,1/2,order=2).astype('float16')
        y1 = zoom(y-y0,1/2,order=2).astype('float16')
        z1 = zoom(z-z0,1/2,order=2).astype('float16')

        #filename = path+'/disp_'+str(int(list_test[idx[i,0]])).zfill(4)+'_'+str(int(list_test[idx[i,1]])).zfill(4)+'.npz'
        #print(filename)
        np.savez_compressed(output_field_npz,np.stack((x1,y1,z1),0))




    


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input_img_fixed",
                        type=str,
                        help="path to input fixed nifti-image")
    
    parser.add_argument("--input_img_moving",
                        type=str,
                        help="path to input moving nifti-image")
    
    parser.add_argument("--output_field_npz",
                        type=str,
                        help="path to output displacement file")
    
    
    parser.add_argument("--input_model_pth",
                        type=str,
                        help="path to obelisk model for PDD-Net")
    
    
   
    pdd(**vars(parser.parse_args()))