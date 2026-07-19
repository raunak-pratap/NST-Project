import torch.nn as nn
import torch
import torchvision.models as models

class VGGEncoder(nn.Module):
    def __init__(self, vgg_path=None, pretrained_cloud=False):
        super(VGGEncoder, self).__init__()
        
        # This structure maps exactly to the AdaIN architecture layers
        vgg_features = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1_2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2_2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3_2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3_3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3_4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU()  # relu4_1
        )
        
        if pretrained_cloud:
            # Safely pulls standard VGG19 features using PyTorch's official cloud backend
            vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            # Copy the weights cleanly into your custom reflection-padded VGG layers
            for target_layer, source_layer in zip(vgg_features, vgg19.features):
                if isinstance(target_layer, nn.Conv2d) and isinstance(source_layer, nn.Conv2d):
                    target_layer.weight.data = source_layer.weight.data.clone()
                    target_layer.bias.data = source_layer.bias.data.clone()
        else:
            # Fall back to loading your local file state dict on your Mac
            vgg_features.load_state_dict(torch.load(vgg_path))
            
        # Slicing layers as expected by your original model forward logic
        self.slice1 = nn.Sequential(*list(vgg_features.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg_features.children())[4:11])
        self.slice3 = nn.Sequential(*list(vgg_features.children())[11:18])
        self.slice4 = nn.Sequential(*list(vgg_features.children())[18:31])

        # Freeze the encoder weights entirely
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input, is_test=False):
        h1 = self.enc_1(input)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)

        if is_test:
            return h4
        return h1, h2, h3, h4
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),         

        )

    def forward(self, input):
        return self.decoder(input)

    