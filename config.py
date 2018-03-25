
default_config = {
    'data_dir': '/tmp',
    'save_dir': '/tmp',
    'data_set': 'cifar',
    'save_interval': 5,
    'load_params': False,
    'nr_resnet': 5,
    'nr_filters': 100,
    'nr_logistic_mix': 10,
    'resnet_nonlinearity': 'concat_elu',
    'learning_rate': 0.001,
    'lr_decay': 0.999995,
    'batch_size': 16,
    'init_batch_size': 64,
    'dropout_p': 0.5,
    'max_epochs': 1000,
    'nr_gpu': 1,
    'polyak_decay': 0.9995,
    'num_samples': 1,
    'seed': 1,
    'spatial_conditional': False,
    'global_conditional': False,
    'spatial_latent_num_channel': 4,
    'global_latent_dim': 10,
    'context_conditioning': True,
    'debug': False,
}



configs = {}

configs['imagenet'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/imagenet",
    "save_dir": "/data/ziz/jxu/models/imagenet-test",
    "nr_filters": 160,
    "nr_resnet": 5,
    "data_set": "imagenet",
    "batch_size": 8,
    "init_batch_size": 8,
    "spatial_conditional": True,
    "save_interval": 10,
    "map_sampling": True,
    "nr_gpu": 2,
}

def get_config(config_name):
    config = {}
    for key, value in default_config.items():
        if key in configs[config_name]:
            config[key] = configs[config_name][key]
        else:
            config[key] = value
    return config
