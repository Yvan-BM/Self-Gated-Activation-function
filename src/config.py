
import argparse

'''_author = Yvan Tamdjo'''

arg = argparse.Namespace(
    batch_size = 128,
    num_workers = 2,
    print_freq = 100,
    save_dir = "checkpoints",
    lr = 0.1,
    momentum = 0.9,
    weight_decay = 1e-4,
    start_epoch = 0,
    epochs = 200,
    save_every = 10,
)