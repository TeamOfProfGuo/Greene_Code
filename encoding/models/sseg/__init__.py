from .linknet import *  
from .refinenet import *
from .refined import *
from .rfunet import *
from .rfdnet import *
from .acnet import *


def get_segmentation_model(name, **kwargs):
    models = {
        'linknet': get_linknet,
        'refinenet': get_refinenet,
        'refined': get_refined, 
        'rfunet': get_rfunet,
        'rfdnet': get_rfdnet,
        'acnet': get_acnet,

    }
    return models[name.lower()](**kwargs)
