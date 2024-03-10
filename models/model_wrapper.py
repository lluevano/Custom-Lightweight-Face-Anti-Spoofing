import torch
from .model_tools import AntiSpoofModel
import torchvision

class model_wrapper(AntiSpoofModel):

    def __init__(self,model_loaded, **kwargs):
        super().__init__(**kwargs)
        
        self.model = model_loaded
        #self.model.fc = self.spoofer
        
        self.include_spoofer = False
    def forward(self,x):
        x = self.model.forward(x)
        if self.include_spoofer:
            x = self.spoofer(x)
        return x
        
        
        
def ShuffleNetV2_default(**kwargs):
    model_shfnet = torchvision.models.shufflenetv2.shufflenet_v2_x2_0(weights='IMAGENET1K_V1')
    model_shfnet.fc = torch.nn.Identity()
    return model_wrapper(model_shfnet, **kwargs)

def MobileNetV3_large_default(**kwargs):
    model_mbnet = torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V1')
    
    default_classifier = model_mbnet.classifier
    model_mbnet.classifier = torch.nn.Sequential(default_classifier[0])
    
    new_model = model_wrapper(model_mbnet, **kwargs)
    new_model.spoofer = torch.nn.Sequential(*default_classifier[1:])
    new_model.spoofer[-1] = torch.nn.Linear(new_model.embeding_dim, 2, bias=True)
    
    #print(new_model)
    return new_model
   
def ResNet50_default(**kwargs):
    model_resnet = torchvision.models.resnet18()
    model_resnet.fc = torch.nn.Identity()
    return model_wrapper(model_resnet, **kwargs)

