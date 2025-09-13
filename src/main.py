import os
import random
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import Preprocessor, load_dataset
from models import prideMMClassifier
from configs import cfg

def main(cfg):
    preprocessor = Preprocessor(cfg)

    if cfg.reproduce == False:
        train_loader, train_weights = load_dataset(cfg, 'train', preprocessor)
        val_loader,   val_weights   = load_dataset(cfg, 'val', preprocessor)
        print("Number of training examples:", len(train_loader))
        print("Number of validation examples:", len(val_loader))

    test_loader,  test_weights  = load_dataset(cfg, 'test', preprocessor)
    print("Number of test examples:", len(test_loader))
    
    seed_everything(cfg.seed, workers=True)
    model = prideMMClassifier(cfg, train_weights)

    monitor = "val/f1"
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.checkpoint_path,\
                                          filename='model', monitor=monitor,\
                                          mode='max', verbose=True,\
                                          save_weights_only=True,
                                          save_top_k=1, save_last=False)

    trainer = Trainer(accelerator='gpu', devices=cfg.gpus,\
                      max_epochs=cfg.max_epochs,\
                      callbacks=[checkpoint_callback],\
                      deterministic=True)

    if cfg.reproduce == False:
        trainer.fit(model, train_dataloaders=train_loader,\
                           val_dataloaders=val_loader)
        model = prideMMClassifier.load_from_checkpoint(
                    checkpoint_path=cfg.checkpoint_file, cfg = cfg) 
    else:
        state_dict = torch.load(cfg.state_dict, weights_only = True)
        model.load_state_dict(state_dict)
        
    trainer.test(model, dataloaders=test_loader)

if __name__ == '__main__':
      main(cfg)

