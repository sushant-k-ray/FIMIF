# FIMIF: Low‑Dimensional Multimodal Fusion with Multiplicative Fine‑Tuning

Pytorch implementation for the paper titled "Low Dimensional Multimodal Fusion Using Multiplicative Fine Tuning Modules" ([Ray et al., CASE 2025](https://aclanthology.org/2025.case-1.15)). Includes lightweight multimodal classifier for meme classification (Hate, Target, Stance, Humour) using frozen CLIP encoders, aggressive low‑dimensional compression, and a residual multiplicative Feature Interaction Module (FIM). Task‑specific checkpoints for PrideMM/CASE 2025 and HarMeme‑C are included.

---

### Highlights
- Tiny head: ~25k–51k trainable parameters; CLIP stays frozen.
- Low‑dimensional fusion: compresses 1536 dimensional vector to an h (h ∈ {8,16}) dimensional vector with a modified residual projection that favors linear separability.
- Residual multiplicative interactions: NALU‑inspired, log‑space multiplicative paths with identity initialization, gated by a residual bypass.
- Concatenation keeps parameters low: image and text embeddings are concatenated once (768+768=1536) and projected with a single small linear block, avoiding large cross‑modal tensors or deep fusion stacks that would inflate parameters.

---

### File structure

```
FIMIF/
├─ models/
│ ├─ PrideMM/
│ │ ├─ hate.pth # Head weights for Subtask A (Hate)
│ │ ├─ target.pth # Head weights for Subtask B (Target)
│ │ ├─ stance.pth # Head weights for Subtask C (Stance)
│ │ ├─ humour.pth # Head weights for Subtask D (Humour)
│ │ └─ parameters.txt # Hyperparameters used to train the above heads
│ |
│ ├─ HarMeme-C/
│ │ ├─ model.pth # Head weights for HarMeme‑C binary task
│ │ └─ parameters.txt # Hyperparameters for HarMeme‑C
│ │
│ └─ shared task (submission)/
│ ├─ hate.pth
│ ├─ target.pth
│ ├─ stance.pth
│ ├─ humour.pth
│ └─ parameters.txt
│
├─ src/
│ ├─ configs.py # YACS config: paths, task, h, LR, epochs, etc.
│ ├─ datasets.py # CLIP preprocessing, CSV loader, concat(IMG|TXT)
│ ├─ main.py # Train/val/test entrypoint (Lightning)
│ └─ models.py # Residual projection + FIM + classifier head
│
├─ LICENSE
└─ README.md
```


**Notes:**
- All `.pth` files are state_dicts for the lightweight fusion head (CLIP is loaded frozen at runtime)
- `parameters.txt` records the hidden size `h` and other training hyperparameters to reproduce inference
- `datasets.py` concatenates normalized image and text embeddings once (768+768=1536), then projects to a very small `h` to keep parameter count minimal

### Installation
Requirements (Python 3.10+ recommended):
- pytorch, pytorch‑lightning

---

### Data layout and CSV schema
Per task:
```
{data_folder}/{task}/
├─ train.csv
├─ val.csv
├─ test.csv
├─ train/{ClassName}/image.jpg
├─ val/{ClassName}/image.jpg
└─ test/{ClassName}/image.jpg
```

CSV columns:
- `index`: image filename inside its class subfolder
- `text`: OCR text string (CLIP tokenizer truncates to 77 tokens)
- `label`: integer class index consistent with `cfg.class_names`

---

### Configuration (src/configs.py)
Key fields (editable in the file or overridden in code):
- `root_dir`: workspace root; state_dict/checkpoint paths derive from here
- `data_folder`: dataset root
- `clip_variant`: e.g., `"ViT-L/14@336px"`
- `task`: `Hate | Target | Stance | Humour` (class_names auto‑populated)
- `device`: `"cuda"` or `"cpu"`; `gpus`: list of device indices (e.g., `[0]`)
- `batch_size`, `image_size=336`
- `embed_dim=768×2=1536` (image+text concatenation)
- `hidden_dim=h` (8 or 16 recommended)
- `lr`, `l1reg`, `l2reg`, `max_epochs`
- `seed`, `reproduce` (True = evaluation‑only with `state_dict`)

Override example:
```
from src.configs import cfg
cfg.data_folder = "/data/PrideMM"
cfg.task = "Target"
cfg.hidden_dim = 8
```

### Citation
If this repository is helpful, please cite using the following BibTeX:
```
@inproceedings{ray-etal-2025-tsr,
    title = "{TSR}@{CASE} 2025: Low Dimensional Multimodal Fusion Using Multiplicative Fine Tuning Modules",
    author = "Ray, Sushant Kr.  and
      Ali, Rafiq  and
      Mohammad, Abdullah  and
      Shabbir, Ebad  and
      Wazir, Samar",
    editor = {H{\"u}rriyeto{\u{g}}lu, Ali  and
      Tanev, Hristo  and
      Thapa, Surendrabikram},
    booktitle = "Proceedings of the 8th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Texts",
    month = sep,
    year = "2025",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2025.case-1.15/",
    pages = "123--132"
}
```

---

### License
See `LICENSE` for details. Datasets are subject to their respective licenses/terms.
