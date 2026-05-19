#!/usr/bin/env python3
"""
Download diseased soybean leaf images for the other_disease class.

Source: anandvermagmailcom/soybean-leaf-diseases
Classes used: Bacterial_blight  (label 0)
              Frogeye            (label 1)

Why soybean diseases for other_disease?
  The other_disease class teaches the model what a NON-ALS bean-like diseased
  leaf looks like.  We cannot use:
    - bean_rust  : too visually similar to ALS (both on bean leaves)
    - corn/grape : completely different leaf shape — misleads the model

  Soybean is the best compromise:
    • Trifoliate compound leaves — same shape family as bean leaves
    • Bacterial blight: water-soaked dark angular/irregular lesions
    • Frogeye: circular grey spots with brown margins
    • Both look distinctly different from ALS angular grey-brown lesions
      bounded by leaf veins

  This gives the model real examples of "diseased-looking legume leaf that
  is NOT ALS," so it learns the ALS-specific pattern rather than just
  "bean leaf with any lesion."

  Available images: ~450 bacterial_blight + ~450 frogeye = ~900 total
  (across train/validation/test splits).

Usage:
  python prepare_other_disease.py              # fill to 800
  python prepare_other_disease.py --count 500
  python prepare_other_disease.py --overwrite
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

SAVE_DIR   = Path('data/other_disease')
HF_REPO    = 'anandvermagmailcom/soybean-leaf-diseases'
MAX_EDGE   = 800
MIN_EDGE   = 128    # soybean images are 224×224 — must be ≤ 224
JPEG_Q     = 95
THREADS    = 8
TARGET     = 800    # ~450 bacterial_blight + ~450 frogeye available

# Label indices in the soybean dataset
DISEASED_LABELS = {0, 1}   # 0=Bacterial_blight, 1=Frogeye  (2=Healthy, 3=Soyabean_rust)


def _count() -> int:
    return len(list(SAVE_DIR.glob('*.jpg'))) if SAVE_DIR.exists() else 0


def _save(pil_img, path: Path) -> bool:
    try:
        if path.exists():
            return False
        img  = pil_img.convert('RGB')
        w, h = img.size
        if min(w, h) < MIN_EDGE:
            return False
        if max(w, h) > MAX_EDGE:
            scale = MAX_EDGE / max(w, h)
            img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img.save(path, quality=JPEG_Q)
        return True
    except Exception:
        return False


def _flush(pairs: list) -> int:
    saved = 0
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(_save, img, path): path for img, path in pairs}
        for fut in as_completed(futures):
            if fut.result():
                saved += 1
    return saved


def download(needed: int, start_idx: int) -> int:
    from datasets import load_dataset

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    collected = []

    for split in ('train', 'validation', 'test'):
        if len(collected) >= needed:
            break
        print(f'   Scanning {HF_REPO}/{split} …')
        try:
            ds = load_dataset(HF_REPO, split=split, streaming=True)
            for sample in ds:
                if sample['label'] not in DISEASED_LABELS:
                    continue   # skip healthy and soyabean_rust
                idx  = start_idx + len(collected)
                path = SAVE_DIR / f'other_disease_{idx:05d}.jpg'
                collected.append((sample['image'], path))
                if len(collected) >= needed:
                    break
        except Exception as e:
            print(f'   {split} error: {e}')

    print(f'   Collected {len(collected)} images — saving …')
    return _flush(collected)


def _summary(target: int):
    print('\n── Dataset summary ─────────────────────────────────────────────')
    classes = ['angular_leaf_spot', 'healthy', 'other_disease', 'other_leaves']
    for cls in classes:
        p  = Path('data') / cls
        n  = len(list(p.glob('*.jpg'))) if p.exists() else 0
        ok = '✅' if n > 0 else '⚠️ '
        print(f'  {ok} {cls:22s}  {n:>5}')
    print('\n▶  Run:  python train_model.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count',     type=int, default=TARGET,
                        help='Target images (default: 800; dataset cap ≈ 900)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Clear existing other_disease and re-download')
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print('❌  pip install datasets pillow')
        sys.exit(1)

    if args.overwrite and SAVE_DIR.exists():
        import shutil
        shutil.rmtree(SAVE_DIR)
        print('Cleared data/other_disease/\n')

    existing = _count()
    needed   = args.count - existing

    if needed <= 0:
        print(f'✅  other_disease already has {existing} images (target={args.count})')
        _summary(args.count)
        return

    print('🌿 other_disease — diseased soybean leaves (bacterial blight + frogeye)')
    print(f'   Source: {HF_REPO}')
    print(f'   Classes: Bacterial_blight (label 0)  |  Frogeye (label 1)')
    print(f'   {existing} existing → need {needed} more (target={args.count})')
    print('=' * 60)

    saved = download(needed, existing)
    print(f'\n✅  Done — {saved} new images  (total: {_count()})')
    _summary(args.count)


if __name__ == '__main__':
    main()
