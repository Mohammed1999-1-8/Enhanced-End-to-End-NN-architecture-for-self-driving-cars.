import torch 
import torch.nn as nn

class DoubleConv(nn.Module):
	def __init__(self, in_channels , out_channels , affine = True):
		super(DoubleConv , self).__init__()

		self.conv_1 = nn.Conv2d(in_channels,out_channels, kernel_size = 3 , stride = 1 , padding = 1)
		self.norm_1 = nn.BatchNorm2d(out_channels, affine=affine)
		self.act_1 = nn.ReLU(inplace = True)
		self.conv_2 = nn.Conv2d(out_channels,out_channels, kernel_size = 3 , stride = 1 , padding = 1)
		self.norm_2 = nn.BatchNorm2d(out_channels, affine=affine)
		self.act_2 = nn.ReLU(inplace = True)

	def forward(self , x):
		
		x = self.conv_1(x)
		x = self.norm_1(x)
		x = self.act_1(x)
		x = self.conv_2(x)
		x = self.norm_2(x)
		x = self.act_2(x)

		return x


class DownConv(nn.Module):
	def __init__(self , in_channels , out_channels , affine = True):
		super(DownConv, self).__init__()

		self.doubleconv = DoubleConv(in_channels = in_channels , out_channels = out_channels , affine = affine)
		self.pool = nn.MaxPool2d(2)

	def forward(self , x):

		skip = self.doubleconv(x)
		output = self.pool(skip)

		return output,skip

class UpConv(nn.Module):
	def __init__(self , in_channels , out_channels , affine = True):
		super(UpConv, self).__init__()

		self.upsample = nn.Upsample(scale_factor = 2)
		self.doubleconv = DoubleConv(in_channels = 2*in_channels , out_channels = out_channels , affine = affine)
		

	def forward(self , Input , skip):
		
		x = self.upsample(Input)
		x = torch.cat((x,skip),1)
		x = self.doubleconv(x)
		return x 



class MobileUNet(nn.Module) :
	def __init__(self , in_channels = 3 , affine = True , internal_channels = 32 , out_channels = 16):
		super(MobileUNet , self).__init__()

		self.out_channels = out_channels
		self.in_channels = in_channels
		self.affine = affine
		self.internal_channels = internal_channels

		self.conv_block_1 = DownConv( in_channels = self.in_channels , out_channels = self.internal_channels ,
			affine = self.affine)

		self.conv_block_2 = DownConv(in_channels = self.internal_channels , out_channels = 2*self.internal_channels ,
			affine = self.affine)

		self.conv_block_3 = DoubleConv(in_channels = 2*self.internal_channels,out_channels = 2*self.internal_channels,
			affine = self.affine)

		self.conv_block_4 = UpConv(in_channels = 2*self.internal_channels,out_channels = self.internal_channels,
			affine = self.affine)

		self.conv_block_5 = UpConv(in_channels = self.internal_channels,out_channels = self.out_channels,
			affine = self.affine)



	def forward(self , Input):

		out , skip_1 = self.conv_block_1(Input)
		out , skip_2 = self.conv_block_2(out)
		out  = self.conv_block_3(out)
		out = self.conv_block_4(out , skip_2)
		out = self.conv_block_5(out,skip_1)
		Output = out  

		return(Output)

class ResNet2(nn.Module) :
    def __init__(self , in_channels , out_channels):
        super(ResNet2 , self).__init__()
        self.conv1 = nn.Conv2d(in_channels , out_channels - in_channels, kernel_size = 3 , stride = 1 , padding = 1)
        self.n1 = nn.BatchNorm2d(out_channels - in_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels - in_channels, out_channels - in_channels, kernel_size = 3 , stride = 2 , padding = 1)
        self.n2 = nn.BatchNorm2d(out_channels - in_channels, affine=True)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self , x):

        identity = x
        x = self.conv1(x)
        x = self.n1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.n2(x)
        x = self.activation(x)
        x = torch.cat((x,self.maxpool(identity)),1)
        return(x)



class AutoPilot(nn.Module):
	def __init__(self , in_channels = 3 ,num_of_classes = 16,num_of_actions = 4):
		super(AutoPilot , self).__init__()
		self.micro_unet = MobileUNet(in_channels = in_channels,out_channels = num_of_classes)

		self.res_conv = nn.Sequential(
			nn.Conv2d(in_channels , num_of_classes , kernel_size=1 , stride=1 , padding = 0),
			nn.BatchNorm2d(num_of_classes, affine=True),
			nn.ReLU(inplace=True),
			)

		self.ResNet = nn.Sequential(
			ResNet2(num_of_classes, 2*num_of_classes), # 64*128
			ResNet2(2*num_of_classes, 4*num_of_classes), # 32*64
			ResNet2(4*num_of_classes, 8*num_of_classes), # 16*32
			ResNet2(8*num_of_classes, 16*num_of_classes), # 8*16
			ResNet2(16*num_of_classes, 32*num_of_classes), # 4*8
			)
		self.intermediam = nn.Sequential(
			nn.MaxPool2d(2),
			nn.Flatten(),
			)
		self.fc = nn.Sequential(
			nn.Dropout(p = 0.8),
			nn.Linear(256*num_of_classes,512),
			nn.ReLU(),
			nn.Dropout(p = 0.8),
			nn.Linear(512,num_of_actions),
			)
	def forward(self,Input):
		x = self.micro_unet(Input) + self.res_conv(Input)
		x = self.ResNet(x)
		x = self.intermediam(x)
		x = self.fc(x)

		return x 
