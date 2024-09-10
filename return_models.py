from models.Rolling_Unet.rolling_unet import Rolling_Unet_S,Rolling_Unet_M,Rolling_Unet_L
from models.UNeXt.UNeXt import UNext_S, UNext
#
from models.UNeXt_official.UNeXt import UNext as UNext_official
from models.UNeXt_official.UNeXt import UNext_S  as UNext_S_official



from models.EGEUNet.EGEUNet import EGEUNet
from models.VM_UNet.vmunet import VMUNet
from models.Swin_Unet.swin_unet import SwinUnet
from models.ACC_UNet.ACC_UNet import ACC_UNet
from models.ACC_UNet.ACC_UNet_lite import ACC_UNet_Lite
from models.ACC_UNet.ACC_UNet_w import ACC_UNet_W
from models.unet.unet_model import UNet
from models.unet_new_v5_graph_autoformer_mean_with_patch_further.unet_model import UNet as UNet_graph
#
from models.SSCANet.swin_unet import SwinUnet as SSCANet
from models.RAPL.swin_unet import SwinUnet as RAPL
from models.CLiSA.swin_unet import SwinUnet as CLiSA
from models.AMCDNet.swin_unet import SwinUnet as AMCDNet
#


# UNet(nn.Module):
#     def __init__(self, n_channels, n_classes
#ACC_UNet_Lite(3,2)
#ACC_UNet_W(3,2)
#ACC_UNet(3,2)


def return_models(names):
    if names == 'Rolling_Unet_S':
        model = Rolling_Unet_S(input_channels=3,num_classes=2)
    if names == 'Rolling_Unet_L':
        model = Rolling_Unet_L(input_channels=3,num_classes=2)
    if names == 'Rolling_Unet_M':
        model = Rolling_Unet_M(input_channels=3,num_classes=2)
    if names == 'UNext_S':
        model  = UNext_S(2)
    if names == 'UNext':
        model  = UNext(2)
        
    if names == 'UNext_S_official':
        model  = UNext_S_official(2)
    if names == 'UNext_official':
        model  = UNext_official(2)
        
    if names == 'EGEUNet':
        model  = EGEUNet(num_classes=2)
    if names == 'VMUNet':
        model  = VMUNet(num_classes=2)
    if names == 'SwinUnet':
        model  =  SwinUnet()
    #SSCANet RAPL CLiSA AMCDNet
    if names == 'SSCANet':
        model  =  SSCANet()
    if names == 'RAPL':
        model  =  RAPL()        
    if names == 'CLiSA':
        model  =  CLiSA()        
    if names == 'AMCDNet':
        model  =  AMCDNet()        
             
    if names == 'ACC_UNet_Lite':
        model = ACC_UNet_Lite(3,2)
    if names == 'ACC_UNet_W':
        model = ACC_UNet_W(3,2)
    if names == 'ACC_UNet':
        model = ACC_UNet(3,2)
    #=UNet
    if names == 'UNet':
        model = UNet(3,2)
    #UNet_graph
    if names == 'UNet_graph':
        model = UNet_graph(3,2)
        
    if names == 'UNet_graph2':
        model = UNet_graph(3,2)
    if names == 'UNet_graph3':
        model = UNet_graph(3,2)
    if names == 'UNet_graph4':
        model = UNet_graph(3,2)
    if names == 'UNet_graph5':
        model = UNet_graph(3,2)
    if names == 'UNet_graph6':
        model = UNet_graph(3,2)
    if names == 'UNet_graph7':
        model = UNet_graph(3,2)
    
    return model



# from torchstat import stat

# stat(return_models(UNet_graph),(3,512,512))

