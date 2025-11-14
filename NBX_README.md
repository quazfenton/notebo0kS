# NBX integration for notebo0kS Python scripts

This folder contains several standalone, notebook-style Python scripts with heterogeneous training logic.
To provide a consistent way to load datasets, log diagnostics, and run quick experiments without refactoring each script, we introduced the NBX package and optional CLI hooks.

What NBX provides

- Data pipeline with a single CLI for common dataset types:
  - csv, parquet, jsonl (tabular via pandas)
  - text (folder of .txt files)
  - images (folder; subfolders as class labels)
  - hf (Hugging Face datasets if `datasets` is installed)
- Lightweight logging helpers (TensorBoard wrapper)
- Utilities: seeding, simple splits

Usage

- Most scripts now accept optional NBX flags when invoked directly. For example:

  python meta_dynamics_extended.py --nbx.use --nbx.kind csv --nbx.path data/train.csv --nbx.input-cols x1,x2,x3 --nbx.target-col y

  python world_model_active_intervention_loop_advanced_jupyter_notebook.py --nbx.use --nbx.kind images --nbx.path ./images --nbx.img-size 128

  python hYpothesis_notebook.py --nbx.use --nbx.kind text --nbx.path ./corpus

  python metaLearn.py --nbx.use --nbx.kind parquet --nbx.path ./data/train.parquet --nbx.input-cols f1,f2,f3 --nbx.target-col label

- The NBX hook prints a dataset summary and exposes basic metadata through environment variables for scripts that want to consume them:
  - NBX_DATA_KIND, NBX_DATA_PATH, NBX_DATA_META

Notes

- We intentionally kept the hooks non-invasive: they run only when you add `--nbx.use` and do not alter the existing training logic unless you read NBX variables.
- Some files (like tErMiNAlmachineL.py) contain notebook-style fenced code that is not valid Python; for those, prefer to extract runnable cells into a clean runner script or port them into a proper module before integrating NBX calls.

Extending NBX in your scripts

- From any script, you can import and use the pipeline directly:

  from nbx.pipeline import DataConfig, DataPipeline
  cfg = DataConfig(kind="csv", path="data/train.csv", input_cols=["x1","x2"], target_col="y")
  split = DataPipeline(cfg).build()
  for batch in split.train:
      ...

- TensorBoard logging:

  from nbx.logging import TBLogger
  tb = TBLogger("outputs/logs")
  tb.scalars({"loss": 0.123}, step=1)
  tb.close()

Caveats

- The scripts here vary widely in completeness and may require cleanup to be fully runnable.
- If you’d like me to convert any specific file into a clean module with full training CLI, say which one(s) and I’ll refactor them with NBX-native training loops.
