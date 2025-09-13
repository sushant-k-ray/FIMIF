import os
from yacs.config import CfgNode 

cfg = CfgNode()
cfg.root_dir = ''
cfg.data_folder = ''
cfg.state_dict = os.path.join(cfg.root_dir, 'state_dict.pth')
cfg.checkpoint_path = os.path.join(cfg.root_dir, 'checkpoints')
cfg.checkpoint_file = os.path.join(cfg.checkpoint_path,'model.ckpt')

cfg.clip_variant = "ViT-L/14@336px"
cfg.task = 'Hate'
cfg.seed = 42
cfg.reproduce = True
cfg.device = 'cuda'
cfg.gpus = [0]

if cfg.task =='Hate':
    cfg.class_names = ['Hate', 'No Hate']
elif cfg.task == 'Target':
    cfg.class_names = ['Undirected', 'Individual', 'Community', 'Organization']
elif cfg.task == 'Stance':
    cfg.class_names = ['Neutral', 'Support', 'Oppose']
elif cfg.task == 'Humour':
    cfg.class_names = ['No Humour', 'Humour']
  
cfg.batch_size = 16
cfg.image_size = 336

cfg.embed_dim = 786 * 2
cfg.hidden_dim = 8

cfg.lr = 1e-4
cfg.l1reg = 1e-4
cfg.l2reg = 1e-4
cfg.max_epochs = 15
cfg.num_classes = len(cfg.class_names)
