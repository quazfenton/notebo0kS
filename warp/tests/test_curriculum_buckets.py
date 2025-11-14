from pathlib import Path
import shutil

from src.slot_program.curriculum import build_buckets

def test_build_buckets(tmp_path: Path):
    base = tmp_path / "clevr_base"
    base.mkdir(parents=True, exist_ok=True)
    buckets = build_buckets(
        base_dir=base,
        levels=2,
        per_level_n=8,
        base_min_objs=2,
        base_max_objs=3,
        base_img_size=32,
        obj_step=1,
        img_step=4,
        seed=123,
    )
    assert len(buckets) == 2
    # Each bucket directory should contain some images
    for b in buckets:
        pngs = list(b.path.glob("*.png"))
        assert len(pngs) >= 1