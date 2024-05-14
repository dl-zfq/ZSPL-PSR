import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)  

		number_f = 32 
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*3,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        
		self.e_conv7 = nn.Conv2d(number_f*2,12,3,1,1,bias=True) 
        
		self.e_conv8 = nn.Conv2d(number_f*2,6,3,1,1,bias=True)
		self.e_conv9 = nn.Conv2d(number_f*2,12,3,1,1,bias=True)

		self.e_conv10 = nn.Conv2d(24,24,3,1,1,bias=True)       
		self.e_conv11 = nn.Conv2d(12,12,3,1,1,bias=True)
		self.e_conv12 = nn.Conv2d(24,24,3,1,1,bias=True)

		self.e_conv13 = nn.Conv2d(60,60,3,1,1,bias=True)
        
		self.convdown = nn.Conv2d(number_f,number_f,5,1,0,bias=True)   
        
		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False) 
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2) 


		
	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))
		down1 = self.convdown(x1)
		x2 = self.relu(self.e_conv2(x1))
		down2 = self.relu(self.e_conv2(down1))
		x3 = self.relu(self.e_conv3(x2))
		down3 = self.relu(self.e_conv3(down2))
		up3 = nn.Upsample(size=(x3.size(2),x3.size(3)), mode='bilinear',align_corners=True)(down3)
		xdown3 = nn.Upsample(size=(down3.size(2),down3.size(3)), mode='bilinear',align_corners=True)(x3)        
		x3 = torch.cat([x3,up3],1) 
		down3 = torch.cat([down3,xdown3],1)
		x4 = self.relu(self.e_conv4(x3))
		down4 = self.relu(self.e_conv4(down3)) 
		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		down5 = self.relu(self.e_conv5(torch.cat([down3,down4],1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		down6 = self.relu(self.e_conv6(torch.cat([down2,down5],1)))
        
		xa = self.relu(self.e_conv1(x))
		downa = self.convdown(xa)
		xb = self.relu(self.e_conv2(xa))
		downb = self.relu(self.e_conv2(downa))
		xc = self.relu(self.e_conv3(xb))
		downc = self.relu(self.e_conv3(downb))
		upc = nn.Upsample(size=(xc.size(2),xc.size(3)), mode='bilinear',align_corners=True)(downc)
		xdownc = nn.Upsample(size=(downc.size(2),downc.size(3)), mode='bilinear',align_corners=True)(xc)        
		xc = torch.cat([xc,upc],1) 
		downc = torch.cat([downc,xdownc],1)
		xd = self.relu(self.e_conv4(xc))
		downd = self.relu(self.e_conv4(downc)) 
		xe = self.relu(self.e_conv5(torch.cat([xc,xd],1)))
		downe = self.relu(self.e_conv5(torch.cat([downc,downd],1)))
		xf = self.relu(self.e_conv6(torch.cat([xb,xe],1)))
		downf = self.relu(self.e_conv6(torch.cat([downb,downe],1)))
        
		xA = self.relu(self.e_conv1(x))
		downA = self.convdown(xA)
		xB = self.relu(self.e_conv2(xA))
		downB = self.relu(self.e_conv2(downA))
		xC = self.relu(self.e_conv3(xB))
		downC = self.relu(self.e_conv3(downB))
		upC = nn.Upsample(size=(xC.size(2),xC.size(3)), mode='bilinear',align_corners=True)(downC)
		xdownC = nn.Upsample(size=(downC.size(2),downC.size(3)), mode='bilinear',align_corners=True)(xC)        
		xC = torch.cat([xC,upC],1) 
		downC = torch.cat([downC,xdownC],1)
		xD = self.relu(self.e_conv4(xC))
		downD = self.relu(self.e_conv4(downC)) 
		xE = self.relu(self.e_conv5(torch.cat([xC,xD],1)))
		downE = self.relu(self.e_conv5(torch.cat([downC,downD],1)))
		xF = self.relu(self.e_conv6(torch.cat([xB,xE],1)))
		downF = self.relu(self.e_conv6(torch.cat([downB,downE],1)))
        
		dx_r = self.relu(self.e_conv7(torch.cat([down1,down6],1))) 
		dx_t = self.relu(self.e_conv8(torch.cat([downa,downf],1))) 
		dx_m = self.relu(self.e_conv9(torch.cat([downA,downF],1))) 
		dx_r = nn.Upsample(size=(dx_r.size(2)+4,dx_r.size(3)+4), mode='bilinear',align_corners=True)(dx_r)
		dx_t = nn.Upsample(size=(dx_t.size(2)+4,dx_t.size(3)+4), mode='bilinear',align_corners=True)(dx_t)
		dx_m = nn.Upsample(size=(dx_m.size(2)+4,dx_m.size(3)+4), mode='bilinear',align_corners=True)(dx_m)  
        
		x_r = self.relu(self.e_conv7(torch.cat([x1,x6],1)))        
		x_t = self.relu(self.e_conv8(torch.cat([xa,xf],1))) 
		#x_m = F.tanh(self.e_conv8(torch.cat([xa,xf],1))) 
		x_m = self.relu(self.e_conv9(torch.cat([xA,xF],1))) 
       
		x_r = torch.cat([x_r,dx_r],1)   
		x_t = torch.cat([x_t,dx_t],1)
		x_m = torch.cat([x_m,dx_m],1)   

		x_r = F.tanh(self.e_conv10(x_r))  
		x_t = F.tanh(self.e_conv11(x_t)) 
		x_m = F.tanh(self.e_conv12(x_m))   

		cate = F.sigmoid(self.e_conv13(torch.cat([x_r,x_t,x_m],1)))     
		a,b,c,d,e,f,g,h,i,j = torch.split(cate, 6, dim=1)
		aa = torch.cat([a,b,c,d],1)
		bb = torch.cat([e,f],1)
		cc = torch.cat([g,h,i,j],1)
		x_r = aa*x_r
		x_t = bb*x_t
		x_m = cc*x_m
        
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1) 
		t1,t2,t3,t4 = torch.split(x_t, 3, dim=1)
		#m1,m2,m3,m4 = torch.split(x_m, 3, dim=1)
		#m1,m2 = torch.split(x_m, 3, dim=1)
		m1,m2,m3,m4,m5,m6,m7,m8 = torch.split(x_m, 3, dim=1)

        
		x = x/(1-t1)   
		x = x/(1-t2)  
		x = x/(1-t3)  
		x = x/(1-t4) 
		avg = torch.mean(x, dim=1) #avg: torch.Size([8, 256, 256]) x: torch.Size([8, 3, 256, 256])
		avg = avg.unsqueeze(1)
		#print("avg:",avg.size())
		x = avg + (x-avg)/(1-m1/100)
		avg = torch.mean(x, dim=1)
		avg = avg.unsqueeze(1)
		x = avg + (x-avg)/(1-m2/100)    
        
		avg = torch.mean(x, dim=1) #avg: torch.Size([8, 256, 256]) x: torch.Size([8, 3, 256, 256])
		avg = avg.unsqueeze(1)
		#print("avg:",avg.size())
		x = avg + (x-avg)/(1-m3/100)
		avg = torch.mean(x, dim=1)
		avg = avg.unsqueeze(1)
		x = avg + (x-avg)/(1-m4/100) 

		avg = torch.mean(x, dim=1) #avg: torch.Size([8, 256, 256]) x: torch.Size([8, 3, 256, 256])
		avg = avg.unsqueeze(1)
		#print("avg:",avg.size())
		x = avg + (x-avg)/(1-m5/100)
		avg = torch.mean(x, dim=1)
		avg = avg.unsqueeze(1)
		x = avg + (x-avg)/(1-m6/100) 

		avg = torch.mean(x, dim=1) #avg: torch.Size([8, 256, 256]) x: torch.Size([8, 3, 256, 256])
		avg = avg.unsqueeze(1)
		#print("avg:",avg.size())
		x = avg + (x-avg)/(1-m7/100)
		avg = torch.mean(x, dim=1)
		avg = avg.unsqueeze(1)
		x = avg + (x-avg)/(1-m8/100) 
# 		x = x + (x - 100)*(1 / (1 - m1 / 255) -1 ) # RGB + (RGB - Threshold) * (1 / (1 - Contrast / 255) - 1)  
# 		x = x + (x - 100)*(1 / (1 - m2 / 255) -1 )        
# 		x = m1*x + f1
# 		x = m2*x + f2
# 		x = (1-m1)*x + m1 
# 		x = (1-m2)*x + m2  
# 		x = (1-m3)*x + m3 
# 		x = (1-m4)*x + m4 
#		x = x/(1-m1) - m1/(1-m1)
# 		x = x/(1-m2) - m2/(1-m2)
# 		x = x + m1*5*x*torch.exp(-14*torch.pow(x,1.6))
# 		x = x - m2*5*(1-x)*torch.exp(-14*torch.pow((1-x),1.6)) 
        
		x = x + r1*(torch.pow(x,2)-x)
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)	
		x = x + r7*(torch.pow(x,2)-x) 
		enhance_image = x + r8*(torch.pow(x,2)-x) 
		r = torch.cat([t1,t2,t3,t4,m1,m2,m3,m4,m5,m6,m7,m8,r1,r2,r3,r4,r5,r6,r7,r8],1) 
		return enhance_image_1,enhance_image,r



