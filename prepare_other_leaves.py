#!/usr/bin/env python3
"""
Download non-bean leaves for the other_leaves class.

Sources:
  Cassava  → andrewkatumba/cassava_leaf_diseases_dsa_2023  (any class)
  Mango    → AfiqN/mango-leaf-disease                       (any class)

Tomato is intentionally excluded: tomato leaves are visually similar to bean
leaves (broad, compound shape), which causes the model to confuse other_leaves
with the healthy bean class.  Cassava and mango have clearly distinct leaf
shapes and textures.

Usage:
  python prepare_other_leaves.py              # 300 images per crop (600 total)
  python prepare_other_leaves.py --count 400
  python prepare_other_leaves.py --overwrite
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

SAVE_DIR = Path('data/other_leaves')
MAX_EDGE = 800
MIN_EDGE = 128
JPEG_Q   = 95
THREADS  = 8
TARGET   = 300   # per crop source


def _count_prefix(prefix: str) -> int:
    return len(list(SAVE_DIR.glob(f'{prefix}_*.jpg'))) if SAVE_DIR.exists() else 0


def _total() -> int:
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


# ── Source 1: Tomato from PlantVillage ───────────────────────────────────────

def _from_tomato(needed: int, start_idx: int) -> list:
    """
    Collect tomato leaf images from Hemg/new-plant-diseases-dataset.
    Uses ALL tomato classes (healthy + diseased) to keep the hit-rate high
    (~14 of 38 classes = ~37 %) so we only need to scan ~800 images for 300.
    """
    from datasets import load_dataset

    HF_REPO = 'Hemg/new-plant-diseases-dataset'
    print(f'   [tomato] Streaming {HF_REPO} (all tomato classes) …')

    try:
        ds   = load_dataset(HF_REPO, split='train', streaming=True)
        feat = ds.features.get('label')
        if feat is None or not hasattr(feat, 'names'):
            print('   [tomato] Cannot read label names — skipping.')
            return []

        # Accept ALL tomato classes so hit-rate is ~37 % instead of ~3 %
        tomato_ids = {
            i for i, n in enumerate(feat.names)
            if 'tomato' in n.lower()
        }
        if not tomato_ids:
            print('   [tomato] No tomato classes found — skipping.')
            return []
        print(f'   [tomato] {len(tomato_ids)} tomato classes found '
              f'(hit-rate ≈ {100*len(tomato_ids)//len(feat.names)} %)')

        collected, scanned = [], 0
        for sample in ds:
            scanned += 1
            if scanned % 500 == 0:
                print(f'   [tomato] scanned {scanned}  '
                      f'collected {len(collected)}/{needed} …')
            if sample.get('label') not in tomato_ids:
                continue
            idx  = start_idx + len(collected)
            path = SAVE_DIR / f'tomato_{idx:05d}.jpg'
            collected.append((sample['image'], path))
            if len(collected) >= needed:
                break

        return collected
    except Exception as e:
        print(f'   [tomato] Error: {e}')
        return []


# ── Source 2: Cassava leaves ─────────────────────────────────────────────────

def _from_cassava(needed: int, start_idx: int) -> list:
    """
    Collect cassava leaf images.
    Takes the first `needed` images regardless of class — cassava leaves look
    entirely different from bean leaves regardless of disease state.
    """
    from datasets import load_dataset

    HF_REPO = 'andrewkatumba/cassava_leaf_diseases_dsa_2023'
    print(f'   [cassava] Streaming {HF_REPO} …')

    try:
        ds = load_dataset(HF_REPO, split='train', streaming=True)
        collected, scanned = [], 0
        for sample in ds:
            scanned += 1
            if scanned % 500 == 0:
                print(f'   [cassava] scanned {scanned}  '
                      f'collected {len(collected)}/{needed} …')
            img_key = next(
                (k for k in ('image', 'img', 'photo') if k in sample), None
            )
            if img_key is None:
                continue
            idx  = start_idx + len(collected)
            path = SAVE_DIR / f'cassava_{idx:05d}.jpg'
            collected.append((sample[img_key], path))
            if len(collected) >= needed:
                break

        return collected
    except Exception as e:
        print(f'   [cassava] Error: {e}')
        return []


# ── Source 3: Mango leaves ───────────────────────────────────────────────────

def _from_mango(needed: int, start_idx: int) -> list:
    """
    Collect mango leaf images from MangoLeafBD.
    Uses all classes — mango leaves are visually distinct from bean regardless.
    """
    from datasets import load_dataset

    HF_REPO = 'AfiqN/mango-leaf-disease'
    print(f'   [mango] Streaming {HF_REPO} …')

    try:
        ds = load_dataset(HF_REPO, split='train', streaming=True)
        collected, scanned = [], 0
        for sample in ds:
            scanned += 1
            if scanned % 500 == 0:
                print(f'   [mango] scanned {scanned}  '
                      f'collected {len(collected)}/{needed} …')
            img_key = next(
                (k for k in ('image', 'img', 'photo') if k in sample), None
            )
            if img_key is None:
                continue
            idx  = start_idx + len(collected)
            path = SAVE_DIR / f'mango_{idx:05d}.jpg'
            collected.append((sample[img_key], path))
            if len(collected) >= needed:
                break

        return collected
    except Exception as e:
        print(f'   [mango] Error: {e}')
        return []


# ── Per-source orchestrator ───────────────────────────────────────────────────

def fill_source(prefix: str, fetch_fn, target: int) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    existing = _count_prefix(prefix)
    needed   = target - existing

    if needed <= 0:
        print(f'  ✅ {prefix}: {existing} images — already at target')
        return

    print(f'\n  📥 {prefix}: {existing} existing → need {needed} more')
    pairs = fetch_fn(needed, existing)
    if pairs:
        saved = _flush(pairs)
        print(f'     ✔ +{saved} saved  (total {existing + saved})')
    else:
        print(f'     ⚠️  No images returned for {prefix}')


def _summary(target: int):
    print('\n── Dataset summary ─────────────────────────────────────────────')
    classes = ['angular_leaf_spot', 'healthy', 'other_disease', 'other_leaves']
    for cls in classes:
        p  = Path('data') / cls
        n  = len(list(p.glob('*.jpg'))) if p.exists() else 0
        ok = '✅' if n >= target else '⚠️ '
        print(f'  {ok} {cls:22s}  {n:>5}')
    print('\n▶  Run:  python train_model.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count',     type=int, default=TARGET,
                        help='Target images per crop (default: 300)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Clear existing other_leaves and re-download')
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print('❌  pip install datasets pillow')
        sys.exit(1)

    if args.overwrite and SAVE_DIR.exists():
        import shutil
        shutil.rmtree(SAVE_DIR)
        print('Cleared data/other_leaves/\n')

    print('🌿 other_leaves — cassava / mango leaves (tomato excluded)')
    print(f'   Target: {args.count} images per crop  ({args.count * 2} total)')
    print('=' * 60)

    fill_source('cassava', lambda n, s: _from_cassava(n, s), args.count)
    fill_source('mango',   lambda n, s: _from_mango(n, s),   args.count)

    print(f'\n   Total other_leaves: {_total()}')
    _summary(args.count)


if __name__ == '__main__':
    main()
