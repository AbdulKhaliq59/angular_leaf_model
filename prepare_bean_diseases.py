#!/usr/bin/env python3
"""
Download bean disease images from the 'beans' dataset (Hugging Face, no auth needed).

Dataset: https://huggingface.co/datasets/beans
Classes available:
  0 = angular_leaf_spot  → already in data/angular_leaf_spot/, skipped
  1 = bean_rust          → saved to data/bean_rust/
  2 = healthy            → already in data/healthy/, skipped

Usage:
  python prepare_bean_diseases.py              # downloads all bean_rust images
  python prepare_bean_diseases.py --count 300
  python prepare_bean_diseases.py --overwrite
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IMG_SIZE   = (224, 224)
THREADS    = 8
SAVE_DIR   = Path('data/bean_rust')

LABEL_BEAN_RUST = 1     # label index in the 'beans' HF dataset


def download_dataset(target: int, existing_count: int):
    try:
        from datasets import load_dataset
    except ImportError:
        print('❌ Run: pip install datasets pillow')
        sys.exit(1)

    print('📦 Streaming beans dataset from Hugging Face …')
    print('   (first run fetches ~50 MB — takes ~1 min on a good connection)\n')

    # Combine train + validation + test splits for maximum images
    splits = ['train', 'validation', 'test']
    all_rust = []
    for split in splits:
        ds = load_dataset('beans', split=split, streaming=True)
        for sample in ds:
            if sample['labels'] == LABEL_BEAN_RUST:
                all_rust.append(sample['image'])
                if len(all_rust) + existing_count >= target:
                    break
        if len(all_rust) + existing_count >= target:
            break

    print(f'   Found {len(all_rust)} bean_rust images in dataset')
    return all_rust


def save_images(images, start_idx: int):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    def save_one(args):
        idx, pil_img = args
        path = SAVE_DIR / f'bean_rust_{idx:05d}.jpg'
        if path.exists():
            return False
        pil_img.convert('RGB').resize(IMG_SIZE).save(path, quality=90)
        return True

    tasks = [(start_idx + i, img) for i, img in enumerate(images)]
    saved = 0
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(save_one, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            if future.result():
                saved += 1
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f'  {done}/{len(tasks)}  saved={saved}')
    return saved


def _summary():
    print()
    classes = ['angular_leaf_spot', 'bean_rust', 'healthy', 'other_leaves']
    for cls in classes:
        p = Path('data') / cls
        n = len(list(p.glob('*'))) if p.exists() else 0
        status = f'{n:>5} images'
        print(f'   data/{cls:22s}  {status}')
    print('\n▶  Run:  python train_model.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count',     type=int, default=1000,
                        help='Max total bean_rust images to save')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    existing = list(SAVE_DIR.glob('*.jpg')) if SAVE_DIR.exists() else []

    if existing and not args.overwrite:
        if len(existing) >= args.count:
            print(f'data/bean_rust already has {len(existing)} images. Use --overwrite to re-download.')
            _summary()
            return
        print(f'data/bean_rust has {len(existing)} images, collecting up to {args.count - len(existing)} more.\n')
    elif args.overwrite and existing:
        import shutil
        shutil.rmtree(SAVE_DIR)
        existing = []
        print('Cleared data/bean_rust/\n')

    images = download_dataset(args.count, len(existing))

    if not images:
        print('\n❌ No bean_rust images retrieved. Check your internet connection.')
        sys.exit(1)

    print(f'\nSaving {len(images)} images ({THREADS} threads) …\n')
    saved = save_images(images, start_idx=len(existing))

    print(f'\n✅ Done — {saved} bean_rust images saved to data/bean_rust/')
    _summary()


if __name__ == '__main__':
    main()
