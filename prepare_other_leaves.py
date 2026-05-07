#!/usr/bin/env python3
"""
Download 'other_leaves' training images using the iNaturalist public API.

- No account, no API key, no giant archive download
- Downloads individual images in parallel (8 threads)
- Targets common non-bean leaf plants (mango, avocado, tomato, corn …)

Usage:
  python prepare_other_leaves.py            # downloads 300 images
  python prepare_other_leaves.py --count 400
  python prepare_other_leaves.py --overwrite
"""

import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
import io

SAVE_DIR     = Path('data/other_leaves')
IMG_SIZE     = (224, 224)
THREADS      = 8
PER_PAGE     = 100       # iNaturalist max per page
REQUEST_DELAY = 0.3      # polite delay between API calls (seconds)

INAT_API = 'https://api.inaturalist.org/v1/observations'

# Common non-bean leaf plants — diverse shapes and colours
TAXA = [
    'Mangifera indica',        # mango
    'Persea americana',        # avocado
    'Solanum lycopersicum',    # tomato
    'Zea mays',                # corn / maize
    'Vitis vinifera',          # grape
    'Carica papaya',           # papaya
    'Musa',                    # banana
    'Citrus sinensis',         # orange
    'Solanum tuberosum',       # potato
    'Coffea arabica',          # coffee
]


def _api_url(taxon: str, page: int) -> str:
    taxon_enc = taxon.replace(' ', '+')
    return (
        f'{INAT_API}?taxon_name={taxon_enc}'
        f'&quality_grade=research&photos=true'
        f'&per_page={PER_PAGE}&page={page}'
        f'&order=desc&order_by=id'
    )


def collect_image_urls(target: int) -> list[str]:
    """Collect image URLs from iNaturalist until we have enough."""
    import json
    urls: list[str] = []
    seen: set[str] = set()

    for taxon in TAXA:
        if len(urls) >= target * 2:   # collect 2× so we have spares after failures
            break
        try:
            api_url = _api_url(taxon, 1)
            req = urllib.request.Request(
                api_url,
                headers={'User-Agent': 'LeafLensAI-DataPrep/1.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            for obs in data.get('results', []):
                for photo in obs.get('photos', []):
                    url = photo.get('url', '')
                    if url and url not in seen:
                        # iNaturalist thumb URLs end with /square.jpg — use medium
                        url = url.replace('/square.', '/medium.')
                        seen.add(url)
                        urls.append(url)

            n = len(data.get('results', []))
            print(f'  {taxon:30s}  {n} observations  (total URLs: {len(urls)})')
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f'  {taxon}: API error — {e}')

    return urls


def download_one(args: tuple) -> bool:
    """Download and resize a single image. Returns True on success."""
    idx, url, path = args
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'LeafLensAI-DataPrep/1.0'}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert('RGB').resize(IMG_SIZE)
        img.save(path, quality=90)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count',     type=int, default=300)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    existing = list(SAVE_DIR.glob('*.jpg')) if SAVE_DIR.exists() else []
    if existing and not args.overwrite:
        if len(existing) >= args.count:
            print(f'Already have {len(existing)} images — done. Use --overwrite to re-download.')
            _summary()
            return
        args.count -= len(existing)
        print(f'Have {len(existing)} images, collecting {args.count} more …')
    elif args.overwrite and existing:
        import shutil
        shutil.rmtree(SAVE_DIR)
        existing = []

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    start_idx = len(existing)

    print(f'\n🌿 Collecting {args.count} other-leaf images via iNaturalist API …\n')
    print('Fetching URLs:')
    urls = collect_image_urls(args.count)

    if not urls:
        print('\n❌ No URLs collected — check your internet connection.')
        sys.exit(1)

    print(f'\nDownloading {min(args.count, len(urls))} images ({THREADS} threads) …\n')

    tasks = [
        (start_idx + i, url, SAVE_DIR / f'inat_{start_idx + i:05d}.jpg')
        for i, url in enumerate(urls[:args.count])
    ]

    saved = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(download_one, t): t for t in tasks}
        for future in as_completed(futures):
            if future.result():
                saved += 1
            else:
                failed += 1
            done = saved + failed
            if done % 25 == 0 or done == len(tasks):
                print(f'  {done}/{len(tasks)}  saved={saved}  failed={failed}')

    print(f'\n✅ Done — {saved} images saved to {SAVE_DIR}/')
    if failed:
        print(f'   ({failed} URLs failed — normal for iNaturalist CDN)')
    _summary()


def _summary():
    print()
    for cls in ('angular_leaf_spot', 'healthy', 'other_leaves'):
        p = Path('data') / cls
        n = len(list(p.glob('*'))) if p.exists() else 0
        print(f'   data/{cls:22s}  {n:>5} images')
    print('\n▶  Run:  python train_model.py')


if __name__ == '__main__':
    main()
