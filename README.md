# TSD-SR

The core code of "Purifying Interaction Sequences Topology-Aware Spectral Denoising for Side-Information Integrated Sequential Recommendation"

## Datasets

We use four preprocessed datasets:

- `Amazon_Beauty`
- `Amazon_Sports_and_Outdoors`
- `Amazon_Toys_and_Games`
- `yelp`

The datasets are too large to include in this repository. Please download them manually from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or the shared [Google Drive folder](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI) before running the code.

On Google Drive, the Amazon review datasets are located in the `Amazon_ratings` folder, and the Yelp dataset is available as `yelp/yelp.zip`.

Extract and organize the datasets under the `dataset/` directory:

```text
dataset/
├── Amazon_Beauty/
├── Amazon_Sports_and_Outdoors/
├── Amazon_Toys_and_Games/
└── yelp/
```

Ensure that the data files use the RecBole format and match the filenames and fields expected by the corresponding configuration. Each dataset directory should contain its interaction (`.inter`) and item feature (`.item`) files, named after the dataset, such as `Amazon_Beauty.inter` and `Amazon_Beauty.item`.

## Configuration

Each dataset has a corresponding YAML configuration in `configs/`:

```text
configs/
├── Amazon_Beauty_tsdsr.yaml
├── Amazon_Sports_and_Outdoors_tsdsr.yaml
├── Amazon_Toys_and_Games_tsdsr.yaml
└── yelp_tsdsr.yaml
```

These files define dataset-specific settings, including data paths, input fields, model parameters, training settings, and evaluation metrics. The `run_tsdsr.sh` script supplies additional command-line overrides for each dataset.

## Running the Experiments

From the repository root, run:

```bash
bash run_tsdsr.sh
```

The script runs TSDSR on Yelp, Beauty, Sports, and Toys in sequence. To select a GPU, set `gpu_id`:

```bash
gpu_id=0 bash run_tsdsr.sh
```

To run a single dataset, use its command from `run_tsdsr.sh`. For example, to run Yelp:

```bash
python run_model.py \
  --model=tsdsr \
  --dataset=yelp \
  --config_files=configs/yelp_tsdsr.yaml \
  --fusion_type=gate \
  --alpha=0.3 \
  --lambda=20 \
  --M=16 \
  --reg_weight=1e-3
```

## Project Structure

```text
├── configs/                 # Dataset-specific configurations
├── dataset/                 # Downloaded datasets
├── recbole/                 # Local RecBole package and registered TSDSR model
├── tsdsr.py                 # TSDSR model implementation
├── run_model.py             # Training entry point
├── run_tsdsr.sh             # Experiment commands for all four datasets
└── paper_experiments.py     # Additional evaluation and experiment utilities
```
