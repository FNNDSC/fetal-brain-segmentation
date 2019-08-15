from models.unet.unet import *
from models.unet.unet_attention_bn import *
from models.unet.unet_attention import *
from models.unet.unet_bn import *

from models.fcn.resnet_fcn import *
from models.fcn.vgg19_fcn import *
from models.fcn.vgg19_attention import *

from models.se_models.resnet_se_fcn import *
from models.se_models.vgg19_se_fcn import *
from models.se_models.unet_se import *

from models.experimental_models.vgg19_fcn_upconv import *
from models.experimental_models.unet_resnet_se import *
from models.experimental_models.unet_upconv import *
from models.experimental_models.unet_upconv_bn import *
from models.experimental_models.unet_upconv_se import *
from models.experimental_models.unet_resnet_upconv_se import *
from models.experimental_models.unet_filter_attention import *

#function to return a particular model given a name

def getModel(name):
    print('Working with %s'%name)

    #the ones that worked the most
    if name == 'unet':
        model = getUnet()

    elif name == 'unet_bn_dice_loss':
        model = getUnetBN('dice')

    elif name == 'unet_bn_focal_loss':
        model = getUnetBN('focal')

    elif name == 'unet_bn_bce_dice_loss':
        model = getUnetBN('BCE_DICE')

    elif name == 'unet_attention_bn_dice_loss':
        model = getAttentionUnetBN('dice')

    elif name == 'unet_attention_bn_bce_dice_loss':
        model = getAttentionUnetBN('BCE_DICE')

    #experiments
    elif name == 'unet_bn':
        model = getUnetBN('BCE_DICE')

    elif name == 'unet_se':
        model = getSEUnet()

    elif name == 'unet_upconv':
        model = getUnetUpconv()

    elif name == 'unet_upconv_bn':
        model = getUnetUpconvBN()

    elif name == 'unet_upconv_se':
        model = getSEUnetUpconv()

    elif name == 'resnetFCN':
        model = getResnet50FCN()
    elif name == 'resnetSEFCN':
        model = getResnetSE50FCN()

    elif name == 'vgg19FCN':
        model = getVGG19FCN()

    elif name == 'vgg19SEFCN':
        model = getVGG19SEFCN()

    elif name == 'unet_resnet_upconv':
        model = getUnetResUpconv()

    elif name == 'unet_resnet_upconv_se':
        model = getUnetResUpconv(se_version = True)

    elif name == 'unet_attention':
        model = getAttentionUnet()

    elif name == 'vgg19FCN_attention_good':
        model = getVGG19Attention()

    elif name == 'vgg19_fcn_upconv':
        model = getVGG19FCN_upconv()

    elif name == 'unet_filter_attention':
        model = getUnetFilterAttention()

    elif name == 'unet_bn_bce_loss':
        model = getUnetBN('BCE')

    else:
        print('error')
        return -1

    return model
