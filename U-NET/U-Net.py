import torch
from torch import nn
import torchvision.transforms.functional as TF


class Double_ConvNet(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super(Double_ConvNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,features=[64,128,256,512]) -> None:
        super(UNET,self).__init__()
        self.up = nn.ModuleList()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for feature in features:
            self.down.append(Double_ConvNet(in_channels,feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.up.append(
                nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2)
            )
            self.up.append(Double_ConvNet(feature*2,feature))

        self.bottle_neck = Double_ConvNet(features[-1],features[-1]*2)
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        skip_connections = []

        for layer in self.down:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottle_neck(x)

        skip_connections = skip_connections[::-1]

        for i in range(len(self.up)):
            if i % 2 == 0:
                x = self.up[i](x)
            else:
                skip_connection = skip_connections[i//2]
                # handling if image had odd width and height
                if x.shape != skip_connection.shape:
                    x = TF.resize(x, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.up[i](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 3, 256, 256))
    model = UNET(in_channels=3, out_channels=3)
    preds = model(x)
    assert preds.shape == x.shape
    print(x.shape)
    print(preds.shape)


if __name__ == "__main__":
    test()
