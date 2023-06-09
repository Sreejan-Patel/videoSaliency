import torch
from torch import nn


class ViNetModel(nn.Module):
    def __init__(self, model_type='vinet_conv'):
        super(ViNetModel, self).__init__()

        self.backbone = S3D()
        if model_type == 'vinet_conv':
            self.decoder = DecoderConvUp()
        else:
            self.decoder = DecoderConv()

    def forward(self, x):
        [y0, y1, y2, y3] = self.backbone(x)

        return self.decoder(y0, y1, y2, y3)


###################
# Encoder network #
###################


class S3D(nn.Module):
    def __init__(self):
        super(S3D, self).__init__()
        self.base1 = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
        )

        self.maxp2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.base2 = nn.Sequential(
            Mixed3b(),
            Mixed3c(),
        )

        self.maxp3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.base3 = nn.Sequential(
            Mixed4b(),
            Mixed4c(),
            Mixed4d(),
            Mixed4e(),
            Mixed4f(),
        )

        self.maxt4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.maxp4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), return_indices=True)

        self.base4 = nn.Sequential(
            Mixed5b(),
            Mixed5c(),
        )

    def forward(self, x):
        # print('input', x.shape)
        y3 = self.base1(x)
        # print('base1', y3.shape)

        y = self.maxp2(y3)
        # print('max_p2', y.shape)

        y2 = self.base2(y)
        # print('base2', y2.shape)

        y = self.maxp3(y2)
        # print('max_p3', y.shape)

        y1 = self.base3(y)
        # print('base3', y1.shape)

        y = self.maxt4(y1)
        y, i0 = self.maxp4(y)
        # print('max_t4_p4', y.shape)

        y0 = self.base4(y)
        # print('base4', y0.shape)
        #
        return [y0, y1, y2, y3]


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes,
                                out_planes,
                                kernel_size=(1, kernel_size, kernel_size),
                                stride=(1, stride, stride),
                                padding=(0, padding, padding),
                                bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes,
                                out_planes,
                                kernel_size=(kernel_size, 1, 1),
                                stride=(stride, 1, 1),
                                padding=(padding, 0, 0),
                                bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x


class Mixed3b(nn.Module):
    def __init__(self):
        super(Mixed3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            SepConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            SepConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed3c(nn.Module):
    def __init__(self):
        super(Mixed3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            SepConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            SepConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed4b(nn.Module):
    def __init__(self):
        super(Mixed4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            SepConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            SepConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed4c(nn.Module):
    def __init__(self):
        super(Mixed4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            SepConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed4d(nn.Module):
    def __init__(self):
        super(Mixed4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            SepConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed4e(nn.Module):
    def __init__(self):
        super(Mixed4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            SepConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            SepConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed4f(nn.Module):
    def __init__(self):
        super(Mixed4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed5b(nn.Module):
    def __init__(self):
        super(Mixed5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed5c(nn.Module):
    def __init__(self):
        super(Mixed5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            SepConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            SepConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

###################
# Decoder network #
###################

class DecoderConvUp(nn.Module):
	def __init__(self):
		super(DecoderConvUp, self).__init__()
		self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
  
		self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
  
		self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
  
		self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
  
		self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, 

			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, 

			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
			nn.ReLU(),            
			nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, y0, y1, y2, y3):
		z = self.convtsp1(y0)
		# print('conv1', z.shape)

		z = torch.cat((z,y1), 2)
		# print('cat_conv1', z.shape)
		
		z = self.convtsp2(z)
		# print('conv2', z.shape)

		z = torch.cat((z,y2), 2)
		# print('cat_conv2', z.shape)
		
		z = self.convtsp3(z)
		# print('conv3', z.shape)

		z = torch.cat((z,y3), 2)
		# print("cat_conv3", z.shape)
		
		z = self.convtsp4(z)
		# print('conv4', z.shape)
		
		z = z.view(z.size(0), z.size(3), z.size(4))
		# print('output', z.shape)

		return z

class DecoderConv(nn.Module):
    def __init__(self):
        super(DecoderConv, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

        self.conv1 = nn.Sequential(
            nn.Conv3d(1024, 832, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        
        # Original
        # nn.Conv3d(832, 480, kernel_size=(3, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1), bias=False),
        # nn.ReLU(),
        self.conv2 = nn.Sequential(
            SepConv3d(832, 480, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(480, 480, kernel_size=(1, 1, 1), stride=(3, 1, 1)),
            nn.ReLU(),
            self.upsampling
        )

        # Original
        # nn.Conv3d(480, 192, kernel_size=(5, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1), bias=False),
        # nn.ReLU(),
        self.conv3 = nn.Sequential(
            SepConv3d(480, 192, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(192, 192, kernel_size=(1, 1, 1), stride=(5, 1, 1)),
            nn.ReLU(),
            self.upsampling
        )

        # Original
        # nn.Conv3d(192, 64, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
        # nn.ReLU(),
        self.conv4 = nn.Sequential(
            SepConv3d(192, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(64, 64, kernel_size=(1, 1, 1), stride=(5, 1, 1)),
            nn.ReLU(),
            self.upsampling,

            nn.Conv3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,

            nn.Conv3d(32, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True),
            nn.Sigmoid()
        )

    def forward(self, y0, y1, y2, y3):
        z = self.conv1(y0)
        # print('conv1', z.shape)

        z = torch.cat((z, y1), 2)
        # print('cat_conv1', z.shape)

        z = self.conv2(z)
        # print('conv2', z.shape)

        z = torch.cat((z, y2), 2)
        # print('cat_conv2', z.shape)

        z = self.conv3(z)
        # print('conv3', z.shape)

        z = torch.cat((z, y3), 2)
        # print("cat_conv3", z.shape)

        z = self.conv4(z)
        # print('conv4', z.shape)

        z = z.view(z.size(0), z.size(3), z.size(4))
        # print('output', z.shape)

        return z


if __name__ == '__main__':
    model = ViNetModel()
    x = torch.randn(4, 3, 8, 224, 384)
    out = model(x)
