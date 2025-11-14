# NBX Pipeline: Extended Notes

Intent
- Provide a minimal, consistent way for heterogeneous scripts to load tabular/text/images/HF datasets without rewriting I/O each time.

Design
- DataConfig(kind, path, input_cols, target_col, batch_size, val_split, img_size, hf_name)
- DataPipeline.build() returns DataSplit(train, val, meta)
- summarize_dataset includes sample batch shape and basic metadata
- load_from_cli parses `--nbx.*` flags and exports NBX_DATA_* environment variables for consumers

Supported kinds
- csv / parquet / jsonl: pandas-backed tabular
- images: folder of images; subfolders as class labels (or single class if flat)
- text: folder with .txt files
- hf: Hugging Face hub datasets (optional dependency)

Usage from any script

from nbx.pipeline import load_from_cli
split, cfg = load_from_cli()
if split is not None:
    print(split.meta)
    for batch in split.train:
        ...

Notes
- NBX is intentionally light-touch and does not dictate model structure.
- For advanced pipelines (tokenization, augmentations), extend NBX or call domain libraries (transformers, torchvision) inside your training loop.
