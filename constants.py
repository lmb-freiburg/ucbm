from os.path import join

BASE_PATH = '/save/dir'

RESULT_PATH = join(BASE_PATH, 'RESULTS')

MODEL_DIR = {
    'cub_rn18':         join(BASE_PATH, 'CUB_RN18/'),
    'places365_rn18':   join(BASE_PATH, 'PLACES365_RN18/'),
    }

MODELS = ['resnet50_v2', 'cub_rn18', 'places365_rn18']

DATA_DIR = {
    'imagenet':     'data/ILSVRC2012/', 
    'cub':          'data/CUB/', 
    'places365':    'data/PLACES365/',
}

DATA_SETS = ['imagenet', 'imagenette', 'imagenet10', 'imagenet20', 'imagenet100', 
             'cub', 'places365']
