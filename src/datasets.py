import pandas as pd
import torch
from clip import clip

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, TensorDataset
from configs import cfg

class Preprocessor:
    def __init__(self, cfg):
        self.clip_model, self.clip_preprocess = clip.load(cfg.clip_variant,
                                                          device=cfg.device,
                                                          jit=False)
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()

    @torch.no_grad()
    def preprocess(self, image_paths, texts, cfg):
        features = torch.tensor([]).to(cfg.device)

        for i in tqdm(range(len(texts))):
            image = Image.open(image_paths[i]).convert('RGB')\
                         .resize((cfg.image_size, cfg.image_size))

            image_features = self.clip_preprocess(image)\
                                 .unsqueeze(0)\
                                 .to(cfg.device)
            image_features = self.clip_model.encode_image(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.reshape(-1)

            text_features = clip.tokenize([texts[i]], truncate=True)\
                                .to(cfg.device)
            text_features = self.clip_model.encode_text(text_features)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.reshape(-1)
            
            conc = torch.cat((image_features, text_features), dim=0)\
                        .reshape((1, -1))
            features = torch.cat((features, conc), dim=0)
            del conc, text_features, image_features
        
        return features

def load_dataset(cfg, split, preprocessor, shuffle=True, upsample=False):
    dataset_pd = pd.read_csv(f'{cfg.data_folder}/{cfg.task}/{split}.csv')
    labels = []

    image_paths = []
    texts = []
    for i in range(len(dataset_pd)):
        row = dataset_pd.iloc[i]
        image = row['index']
        text = row['text']
        label = int(row['label'])
        image_paths.append(f'{cfg.data_folder}/{cfg.task}/'\
                           f'{split}/{cfg.class_names[label]}/{image}')
        
        texts.append(text)
        labels.append(label)

    data = preprocessor.preprocess(image_paths, texts, cfg)
    label = torch.tensor(label).to(cfg.device)

    weights = []
    for i in range(cfg.num_classes):
        weights.append((labels == i).sum())

    m = max(weights)

    if upsample:
        X = torch.tensor([]).to(cfg.device)
        y = torch.tensor([]).to(cfg.device)

        for i in range(cfg.num_classes):
            idata = data[label == i]
            ilabel = label[label == i]
            n = len(ilabel)
            cx = idata
            cy = ilabel
            cur = n
            while cur + n < m:
                cx = torch.cat((cx, idata), dim = -1)
                cy = torch.cat((cy, ilabel), dim = -1)
                cur += n

            X = torch.cat((X, cx), dim = 0)
            y = torch.cat((y, cy), dim = 0)
            weights[i] = cur

        data = X
        label = y

    for i in range(cfg.num_classes):
        weights[i] = n / weights[i]

    weights = torch.tensor(weights).float().to(cfg.device)
    return DataLoader(TensorDataset(data, label), batch_size=cfg.batch_size,
                      shuffle=shuffle, num_workers=0), weights
