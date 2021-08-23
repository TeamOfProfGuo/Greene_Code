
from .nyud_v2 import NYUD
from .sun_rgbd import SUN_RGBD
from .sunrgbd import *


datasets = {
    'nyud': NYUD,
    'sunrgbd': SUN_RGBD
}

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)