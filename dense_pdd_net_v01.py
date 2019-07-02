import torch
import torch.nn as nn
import torch.optim as optim

import time

import torch.nn.functional as F
#import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
from torch.utils.checkpoint import checkpoint




fold = 1#int(sys.argv[1])
list_train = torch.Tensor([1,4,5,6,9,10]).long()
if(fold==2):
    list_train = torch.Tensor([2,3,5,6,7,8,9]).long()
if(fold==3):
    list_train = torch.Tensor([1,2,3,4,7,8,10]).long()


B = 10#len(list_train)
H = 233; W = 168; D = 286;
imgs = torch.zeros(B,1,H,W,D)
segs = torch.zeros(B,H,W,D).long()
label_select = torch.Tensor([0,1,2,3,4,5,6,7,0,0,8,9]).long()

for i in range(B):
    #case_train = int(list_train[i])
    imgs[i,0,:,:,:] = torch.from_numpy(nib.load('/share/data_zoe2/heinrich/DatenPMBV/img'+str(i+1)+'v2.nii.gz').get_data())/500.0#.unsqueeze(0).unsqueeze(0)
    segs[i,:,:,:] = label_select[torch.from_numpy(nib.load('/share/data_zoe2/heinrich/DatenPMBV/seg'+str(i+1)+'v2.nii.gz').get_data()).long()]


#img00 = torch.from_numpy(nib.load('/share/data_zoe2/heinrich/DatenPMBV/img10v2.nii.gz').get_data()).unsqueeze(0).unsqueeze(0)
#img50 = torch.from_numpy(nib.load('/share/data_zoe2/heinrich/DatenPMBV/img5v2.nii.gz').get_data()).unsqueeze(0).unsqueeze(0)

#seg00 = torch.from_numpy(nib.load('/share/data_zoe2/heinrich/DatenPMBV/seg10v2.nii.gz').get_data()).long().unsqueeze(0)
#seg50 = torch.from_numpy(nib.load('/share/data_zoe2/heinrich/DatenPMBV/seg5v2.nii.gz').get_data()).long().unsqueeze(0)
#seg00 = label_select[seg00]
#seg50 = label_select[seg50]

def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


print(np.unique(segs.view(-1).numpy()))
#    mask_train[i,:,:,:] = torch.from_numpy(nib.load('/share/data_zoe2/heinrich/DatenPMBV/mask'+str(case_train)+'v2.nii.gz').get_data())#.long()
   
d0 = dice_coeff(segs[9,:,:,:], segs[4,:,:,:], 10)
print(d0.mean(),d0)


o_m = H//3
o_n = W//3
o_o = D//3
print('numel_o',o_m*o_n*o_o)
ogrid_xyz = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,o_m,o_n,o_o)).view(1,1,-1,1,3).cuda()

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.xavier_normal(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.0)

def countParameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class OBELISK(nn.Module):
    def __init__(self):

        super(OBELISK, self).__init__()
        channels = 16#16
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
        sampled = F.grid_sample(img_in,ogrid_xyz + self.offsets[0,:,:].view(1,-1,1,1,3)).view(1,-1,o_m,o_n,o_o)
        sampled -= F.grid_sample(img_in,ogrid_xyz + self.offsets[1,:,:].view(1,-1,1,1,3)).view(1,-1,o_m,o_n,o_o)
    
        x = F.relu(self.batch1(self.layer1(sampled)))
        x = F.relu(self.batch2(self.layer2(x)))
        features = self.layer3(x)
        return features




disp_range = 0.4#0.25
displacement_width = 15#11#17
shift_xyz = F.affine_grid(disp_range*torch.eye(3,4).unsqueeze(0),(1,1,displacement_width,displacement_width,displacement_width)).view(1,1,-1,1,3).cuda()

#_,_,H,W,D = img00.size()
grid_size = 32#25#30
grid_xyz = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,grid_size,grid_size,grid_size)).view(1,-1,1,1,3).cuda()

#    print('moving_unfold',torch.numel(moving_unfold)*4e-6,'MBytes')  
#print('deeds_cost',torch.numel(deeds_cost)*4e-6,'MBytes')
     #minconv_seq = nn.Sequential(pad1,max1,avg1,avg1)
    #cost = checkpoint(deeds_cost,minconv_seq)
#print('cost',torch.numel(cost)*4e-6,'MBytes')   
#    
#    avg_seq = nn.Sequential(pad2,avg1,avg1)
#>>> input_var = checkpoint_sequential(model, chunks, input_var)
    
    #cost_avg =  checkpoint(avg_seq,cost_permute)#
# print('cost_avg',torch.numel(cost_avg)*4e-6,'MBytes') 
def augmentAffine(img_in, seg_in, strength=0.05):
    """
    3D affine augmentation on image and segmentation mini-batch on GPU.
    (affine transf. is centered: trilinear interpolation and zero-padding used for sampling)
    :input: img_in batch (torch.cuda.FloatTensor), seg_in batch (torch.cuda.LongTensor)
    :return: augmented BxCxTxHxW image batch (torch.cuda.FloatTensor), augmented BxTxHxW seg batch (torch.cuda.LongTensor)
    """
    B,C,D,H,W = img_in.size()
    affine_matrix = (torch.eye(3,4).unsqueeze(0) + torch.randn(B, 3, 4) * strength).to(img_in.device)

    meshgrid = F.affine_grid(affine_matrix,torch.Size((B,1,D,H,W)))

    img_out = F.grid_sample(img_in, meshgrid,padding_mode='border')
    seg_out = F.grid_sample(seg_in.float().unsqueeze(1), meshgrid, mode='nearest').long().squeeze(1)

    return img_out, seg_out

class deeds(nn.Module):
    def __init__(self):

        super(deeds, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([1,.1,1,1,.1,1]))#.cuda()

        self.pad1 = nn.ReplicationPad3d(3)#.cuda()
        self.avg1 = nn.AvgPool3d(3,stride=1)#.cuda()
        self.max1 = nn.MaxPool3d(3,stride=1)#.cuda()
        self.pad2 = nn.ReplicationPad3d(2)#.cuda()##



    def forward(self, feat00,feat50):
        
        #deeds correlation layer (slightly unrolled)
        deeds_cost = torch.zeros(1,grid_size**3,displacement_width,displacement_width,displacement_width).cuda()
        xyz8 = grid_size**2
        for i in range(grid_size): 
            moving_unfold = F.grid_sample(feat50,grid_xyz[:,i*xyz8:(i+1)*xyz8,:,:,:] + shift_xyz,padding_mode='border')
            fixed_grid = F.grid_sample(feat00,grid_xyz[:,i*xyz8:(i+1)*xyz8,:,:,:])
            deeds_cost[:,i*xyz8:(i+1)*xyz8,:,:,:] = self.alpha[1]+self.alpha[0]*torch.sum(torch.pow(fixed_grid-moving_unfold,2),1).view(1,-1,displacement_width,displacement_width,displacement_width)

        # remove mean (not really necessary)
        #deeds_cost = deeds_cost.view(-1,displacement_width**3) - deeds_cost.view(-1,displacement_width**3).mean(1,keepdim=True)[0]
        deeds_cost = deeds_cost.view(1,-1,displacement_width,displacement_width,displacement_width)
    
        # approximate min convolution / displacement compatibility
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




net = OBELISK()
net.apply(init_weights)
net.cuda()
net.train()

class_weight = torch.sqrt(1.0/(torch.bincount(segs.view(-1)).float()))
class_weight = class_weight/class_weight.mean()
class_weight[0] = 0.15
class_weight = class_weight.cuda()
print('inv sqrt class_weight',class_weight)
criterion = nn.CrossEntropyLoss(class_weight)

t0 = time.time() 

reg = deeds()
reg.cuda()
print('alpha_before',reg.alpha)


list_train = torch.Tensor([1,4,5,6,9,10]).long()-1



#img00.requires_grad = True
#img50.requires_grad = True
iterations = 1000 
lambda_weight = 2#2.5#1.5
run_labelloss = torch.zeros(iterations)#/0
run_diffloss = torch.zeros(iterations)#/0

optimizer = optim.Adam(list(net.parameters())+list(reg.parameters()),lr=0.005)

for i in range(iterations):
    
    idx = list_train[torch.randperm(6)].view(2,3)[:,0]
    #print(idx)
    optimizer.zero_grad()
    label_moving = torch.zeros(size=(1,10,H,W,D)).cuda()
    label_moving = label_moving.scatter_(1, segs[idx[1]:idx[1]+1,:,:,:].unsqueeze(1).cuda(), 1).detach()
    
    img00_in = imgs[idx[0]:idx[0]+1,:,:,:,:].cuda()
    img50 = imgs[idx[1]:idx[1]+1,:,:,:,:].cuda()
    
    img00, seg50 = augmentAffine(img00_in,segs[idx[0]:idx[0]+1,:,:,:].cuda(),0.0375)
    img00.requires_grad = True
    img50.requires_grad = True
    
    label_fixed = torch.zeros(size=(1,10,H,W,D)).cuda()
    label_fixed = label_fixed.scatter_(1, seg50.unsqueeze(1), 1).detach()
    
    # get features (regular grid)
    feat00 = checkpoint(net,img00)#net(img00)# #00 is fixed
    feat50 = checkpoint(net,img50)#net(img50)# #50 is moving
    # run differentiable deeds (regular grid)
    cost_soft,pred_xyz =  checkpoint(reg,feat00,feat50)#reg(feat00,feat50)#
    pred_xyz = pred_xyz.view(1,grid_size,grid_size,grid_size,3)
    # evaluate diffusion regularisation loss
    diffloss = lambda_weight*((pred_xyz[0,:,1:,:,:]-pred_xyz[0,:,:-1,:,:])**2).mean()+\
            lambda_weight*((pred_xyz[0,1:,:,:,:]-pred_xyz[0,:-1,:,:,:])**2).mean()+\
            lambda_weight*((pred_xyz[0,:,:,1:,:]-pred_xyz[0,:,:,:-1,:])**2).mean()
    run_diffloss[i] = diffloss.item()


    # evaluate non-local loss
    nonlocal_label = (F.grid_sample(label_moving,grid_xyz+shift_xyz,padding_mode='border')\
                          *cost_soft.view(1,-1,grid_size**3,displacement_width**3,1)).sum(3,keepdim=True)
    fixed_label = F.grid_sample(label_fixed,grid_xyz,padding_mode='border').detach()#.long().squeeze(1)
    
    labelloss = ((nonlocal_label-fixed_label)**2)*class_weight.view(1,-1,1,1,1)
    labelloss = labelloss.mean()
    #labelloss = criterion(nonlocal_label,fixed_label)
    run_labelloss[i] = labelloss.item()
    (labelloss+diffloss).backward()

    optimizer.step()
    
    if(i%50==49):
        print('epoch',i,'time',time.time()-t0)

        #print('grad',reg.layer1.weight.grad.norm().item())

        loss_avg = F.avg_pool1d(run_labelloss.view(1,1,-1),5,stride=1).squeeze().numpy()[:i]
        print('run_labelloss',loss_avg[-1])
        loss_avg = F.avg_pool1d(run_diffloss.view(1,1,-1),5,stride=1).squeeze().numpy()[:i]
        print('run_diffloss',loss_avg[-1])

        #plt.plot(F.avg_pool1d(run_labelloss.view(1,1,-1),5,stride=1).squeeze().numpy()[:i])
        #plt.plot(F.avg_pool1d(run_diffloss.view(1,1,-1),5,stride=1).squeeze().numpy()[:i])

        #plt.show()
        #plt.imshow(pred_xyz[0,:,12,:,0].cpu().data.numpy())
        #plt.colorbar()
        #plt.show()
        
        torch.save(net.cpu().state_dict(),'/data_supergrover2/heinrich/dense_reg3_feat_epoch'+str(i)+'.pth')
        torch.save(reg.cpu().state_dict(),'/data_supergrover2/heinrich/dense_reg3_deeds_epoch'+str(i)+'.pth')

        net.cuda()
        reg.cuda()

    
   
    #

torch.cuda.synchronize()

print('time',time.time()-t0)
print('grad_alpha',reg.alpha.grad.norm())
print('grad_obelisk',net.layer1.weight.grad.norm())

print('alpha_after',reg.alpha)


