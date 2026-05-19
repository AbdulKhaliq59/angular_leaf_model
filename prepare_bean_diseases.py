#!/usr/bin/env python3
"""
Download bean leaf classes from Hugging Face ('beans' dataset).

Classes handled:
  angular_leaf_spot  → data/angular_leaf_spot/   (~430 images available)
  healthy            → data/healthy/              (~430 images available)

Non-ALS disease training is handled by prepare_other_disease.py using
soybean bacterial-blight and frogeye images — legume leaves that look
similar to bean but with disease patterns distinct from ALS.

Usage:
  python prepare_bean_diseases.py               # download both classes
  python prepare_bean_diseases.py --cls angular_leaf_spot
  python prepare_bean_diseases.py --overwrite
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

MAX_EDGE  = 800
MIN_EDGE  = 256
JPEG_Q    = 95
THREADS   = 8
TARGET    = 1000

BEANS_LABEL = {
    'angular_leaf_spot': 0,
    'healthy':           2,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count(cls: str) -> int:
    p = Path('data') / cls
    return len(list(p.glob('*.jpg'))) if p.exists() else 0


def _save(pil_img, path: Path) -> bool:
    try:
        if path.exists():
            return False
        img = pil_img.convert('RGB')
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


def _flush(pairs: list, label: str, threads: int = THREADS) -> int:
    saved = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(_save, img, path): path for img, path in pairs}
        done = 0
        for fut in as_completed(futures):
            if fut.result():
                saved += 1
            done += 1
            if done % 100 == 0 or done == len(pairs):
                print(f'      [{label}] {done}/{len(pairs)}  saved={saved}')
    return saved


def _from_beans(cls: str, needed: int, start_idx: int) -> list:
    from datasets import load_dataset

    label_id = BEANS_LABEL[cls]
    save_dir = Path('data') / cls
    save_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    for split in ('train', 'validation', 'test'):
        if len(collected) >= needed:
            break
        try:
            ds = load_dataset('beans', split=split, streaming=True)
            for sample in ds:
                if sample['labels'] == label_id:
                    idx  = start_idx + len(collected)
                    path = save_dir / f'{cls}_{idx:05d}.jpg'
                    collected.append((sample['image'], path))
                    if len(collected) >= needed:
                        break
        except Exception as e:
            print(f'      beans/{split} error: {e}')

    return collected


def fill_class(cls: str, target: int, overwrite: bool) -> None:
    save_dir = Path('data') / cls

    if overwrite and save_dir.exists():
        import shutil
        shutil.rmtree(save_dir)

    existing = _count(cls)
    needed   = target - existing

    if needed <= 0:
        print(f'  ✅ {cls}: {existing} images — already at target')
        return

    print(f'\n  📥 {cls}: {existing} existing  →  need {needed} more  (target={target})')
    pairs = _from_beans(cls, needed, existing)
    if pairs:
        saved    = _flush(pairs, cls)
        existing += saved
        print(f'     ✔ +{saved} saved  (total={existing})')
    else:
        print(f'     beans returned 0 images for {cls}')

    final = _count(cls)
    icon  = '✅' if final > 0 else '⚠️ '
    note  = ''
    if final < target:
        note = (f'  ⚠️  {final}/{target} — beans dataset exhausted (~430/class max).')
    print(f'  {icon} {cls}: {final} images{note}')


def _summary(target: int):
    print('\n── Dataset summary ─────────────────────────────────────────────')
    for cls in ['angular_leaf_spot', 'healthy']:
        n   = _count(cls)
        bar = '█' * (n // 50)
        ok  = '✅' if n >= target else '⚠️ '
        print(f'  {ok} {cls:22s}  {n:>5}  {bar}')
    print('\n▶  Also run: python prepare_other_disease.py')
    print('▶  Also run: python prepare_other_leaves.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count',     type=int, default=TARGET)
    parser.add_argument('--cls',       default=None,
                        choices=list(BEANS_LABEL.keys()))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print('❌  pip install datasets pillow')
        sys.exit(1)

    classes = [args.cls] if args.cls else list(BEANS_LABEL.keys())

    print('🌿 Bean leaf dataset preparation (ALS + Healthy only)')
    print(f'   Target: {args.count} images per class')
    print('=' * 60)

    for cls in classes:
        fill_class(cls, args.count, args.overwrite)

    _summary(args.count)


if __name__ == '__main__':
    main()
