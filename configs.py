import os

def get_default():
    cfg = {}
    cfg.update({
        "nonlinearity": "relu",
        "batch_size": 32,
        "learning_rate": 0.0001,
        "lam": 0.0,
        "nr_resnet": 5,
        "nr_logistic_mix": 10,
        "nr_filters": 100,
        "save_interval": 10,
        "sample_range": 3.0,
        "network_size": "medium",
        "reg": "mmd",
        "beta": 5e5,
    })

def use_dataset(cfg, dataset, size):
    cfg.update({
        "img_size": size,
        "data_set": "{0}{1}".format(dataset, size),
    })
    if "celeba" in dataset:
        cfg.update({
            "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
        })
    elif "svhn" in dataset:
        cfg.update({
            "data_dir": "/data/ziz/not-backed-up/jxu/SVHN",
        })
    return cfg

def get_save_dir(cfg, name=None, suffix="_"):
    base = "/data/ziz/jxu/save_dirs/"
    if name is None:
        name = "checkpoints_{0}_{1}_{2}_{3}_{4}".format(cfg['data_set'], cfg['reg'], cfg['beta'], cfg['nr_resnet'], cfg['phase'])
    return base + name

def get_config(config={}, name=None, suffix="", load_dir="", dataset='celeba', size=32, mode='test', phase='pvae', use_mask_for="input output"):
    # mode: train | test
    # phase: pvae | ce
    # use_mask_for: input output | input | output | none
    cfg = get_default()
    cfg = use_dataset(cfg, dataset, size)
    cfg.update(config)
    cfg.update({
        "mode": mode,
        "phase": phase,
        "use_mask_for": use_mask_for,
        "load_dir": load_dir,
    })
    cfg.update({"save_dir":get_save_dir(cfg, name=name, suffix=suffix)})
    if os.path.exists(cfg['save_dir']):
        overwrite = raw_input("overwrite?:")
        if not (overwrite=='y' or overwrite=='Y'):
            quit()
    return cfg
