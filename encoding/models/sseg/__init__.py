from .linknet import *  
from .refinenet import *
from .refined import *
from .rfunet import *
from .rfdnet import *
from .acnet import *
from .bisenet import *
from .bised import *


def get_segmentation_model(name, **kwargs):
    models = {
        'linknet': get_linknet,
        'refinenet': get_refinenet,
        'refined': get_refined, 
        'rfunet': get_rfunet,
        'rfdnet': get_rfdnet,
        'acnet': get_acnet,
        'bise': get_bise,
        'bised': get_bised,
    }
    return models[name.lower()](**kwargs)
