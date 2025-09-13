# FIMIF: Low‑Dimensional Multimodal Fusion with Multiplicative Fine‑Tuning

Lightweight multimodal classifier for meme understanding (Hate, Target, Stance, Humour) using frozen CLIP encoders, aggressive low‑dimensional compression, and a residual multiplicative Feature Interaction Module (FIM). Task‑specific checkpoints for PrideMM/CASE 2025 and HarMeme‑C are included.

---

### Highlights
- Tiny head: ~25k–51k trainable parameters; CLIP stays frozen.
- Low‑dimensional fusion: compress 1536→h (h∈{8,16}) with a modified residual projection that favors linear separability.
- Residual multiplicative interactions: NALU‑inspired, log‑space multiplicative paths with identity initialization, gated by a residual bypass.
- Concatenation keeps parameters low: image and text embeddings are concatenated once (768+768=1536) and projected with a single small linear block, avoiding large cross‑modal tensors or deep fusion stacks that would inflate parameters.

---

### Repository structure
FIMIF/
├─ models/
│ ├─ PrideMM/
│ │ ├─ hate.pth
│ │ ├─ target.pth
│ │ ├─ stance.pth
│ │ ├─ humour.pth
│ │ └─ parameters.txt
│ ├─ HarMeme-C/
│ │ ├─ model.pth
│ │ └─ parameters.txt
│ └─ shared task (submission)/
│ ├─ hate.pth
│ ├─ target.pth
│ ├─ stance.pth
│ ├─ humour.pth
│ └─ parameters.txt
├─ src/
│ ├─ configs.py
│ ├─ datasets.py
│ ├─ main.py
│ └─ models.py
├─ LICENSE
└─ README.md

text

---

### Installation
Requirements (Python 3.10+ recommended):
- torch, torchvision, timm, transformers
- pytorch‑lightning, torchmetrics
- yacs, pillow, pandas, tqdm, scikit‑learn

Example:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm transformers pytorch-lightning torchmetrics yacs pillow pandas tqdm scikit-learn

text

---

### Data layout and CSV schema
Per task:
{data_folder}/{task}/
├─ train.csv
├─ val.csv
├─ test.csv
├─ train/{ClassName}/image.jpg
├─ val/{ClassName}/image.jpg
└─ test/{ClassName}/image.jpg

text

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
from src.configs import cfg
cfg.data_folder = "/data/PrideMM"
cfg.task = "Target"
cfg.hidden_dim = 8

text

---

### How the model works (src/models.py)
- Concatenation for parameter efficiency  
  Frozen CLIP encodes image and text; both vectors are L2‑normalized and concatenated to a 1536‑d vector (768+768). A single projection compresses 1536→h, so parameters scale with `1536×h` (plus small biases) rather than with large fusion tensors or deep cross‑attention. With `h∈{8,16}`, the head stays extremely compact.

- Residual projection (compression)  
  `resModule` implements `A(X)+B(X)` with ReLU on `A` to map `1536→h`. This improves conditioning and tends to make decision boundaries mostly linear after compression.

- Residual multiplicative interactions  
  FIM uses two log‑space multiplicative paths with identity‑initialized middle matrices and a residual linear path:
mult_layer(x, A, W, B) = B(exp(W(log(ReLU(A(x))+ε))))
FIM(x) = residual(x) + mult_layer(...)*mult_layer(...)

text
Identity init means it starts near‑linear and only learns multiplicative behavior where it helps.

- Classifier and metrics  
`LayerNorm(h) → Linear(h→num_classes)` with CrossEntropyLoss (optional class weights). Metrics: macro F1, Accuracy, AUROC.

---

### Preprocessing (src/datasets.py)
- Loads CLIP (frozen), applies CLIP preprocess, resizes images to `336×336`.
- Tokenizes OCR text with `truncate=True` (77 tokens).
- Encodes image and text, L2‑normalizes each, concatenates to `[1, 1536]`.
- Returns a `TensorDataset` for `DataLoader`.

---

### Training (src/main.py)
Default (train + val + test):
python src/main.py

text

Behavior:
- If `cfg.reproduce == False`:
  - Builds train/val/test loaders
  - Seeds for determinism
  - Trains with AdamW, monitors `val/f1`
  - Saves best checkpoint to `{cfg.checkpoint_path}/model.ckpt`
  - Loads best `.ckpt` and runs test
- If `cfg.reproduce == True`:
  - Loads weights from `cfg.state_dict` (a `.pth` state_dict) and runs test directly

GPU setup:
- `cfg.device="cuda"`; `cfg.gpus=[0]` (or multiple indices)

Imbalance handling:
- The loader computes per‑class weights for CrossEntropy.
- Deterministic oversampling for Target can be enabled by switching `load_dataset(..., upsample=True)` in `src/datasets.py` if needed.

Recommended hidden sizes:
- Hate: `h=16`
- Target, Stance, Humour: `h=8`

---

### Inference (with shipped checkpoints)
Evaluate using a state_dict without retraining:
from src.configs import cfg
cfg.data_folder = "/data/PrideMM"
cfg.task = "Hate"
cfg.hidden_dim = 16
cfg.root_dir = "."
cfg.state_dict = "models/PrideMM/hate.pth" # or "models/shared task (submission)/hate.pth"
cfg.reproduce = True

text
undefined
python src/main.py

text

Note: Quote paths containing spaces:
python src/main.py --ckpt "models/shared task (submission)/hate.pth"

text

---

### Pretrained checkpoints
- `models/PrideMM/`: `hate.pth`, `target.pth`, `stance.pth`, `humour.pth`, `parameters.txt`
- `models/shared task (submission)/`: task‑wise `.pth` + `parameters.txt`
- `models/HarMeme‑C/`: `model.pth` + `parameters.txt`

Each `.pth` is a `state_dict` for the lightweight head; CLIP backbones load dynamically and remain frozen.

---

### Results (short summary)
- Comparable or better Accuracy/F1 than larger fusion heads on multiple subtasks with ~100× fewer parameters.
- Hidden‑size sweeps favor small `h` (e.g., 8), indicating effective compression and near‑linear separability after projection.

---

### Minimal examples

Train Hate:
from src.configs import cfg
cfg.data_folder="/data/PrideMM"
cfg.task="Hate"
cfg.hidden_dim=16
cfg.reproduce=False

text
undefined
python src/main.py

text

Evaluate Target with provided weights:
from src.configs import cfg
cfg.data_folder="/data/PrideMM"
cfg.task="Target"
cfg.hidden_dim=8
cfg.state_dict="models/PrideMM/target.pth"
cfg.reproduce=True

text
undefined
python src/main.py

text

---

### Citation
If this repository is helpful, please cite the associated paper and baselines.

---

### License
See `LICENSE` for details. Datasets are subject to their respective licenses/terms.

---

### Acknowledgements
Thanks to the dataset creators and to the open CLIP ecosystem that enabled this lightweight fusion approach.
