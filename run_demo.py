"""Verify public inputs and run the actual N008 model demonstration."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights/best.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/demo')
    parser.add_argument('--no-download', action='store_true', help='Fail if weights are unavailable locally')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'model_manifest.json').read_text(encoding='utf-8'))
    meta = json.loads((ROOT / 'demo/N008/metadata.json').read_text(encoding='utf-8'))
    for relative, expected in meta['frame_sha256'].items():
        if sha256(ROOT / 'demo/N008' / relative) != expected:
            raise RuntimeError(f'Frame checksum mismatch: {relative}')
    if not args.weights.exists():
        if args.no_download:
            raise FileNotFoundError(args.weights)
        args.weights.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.weights.with_suffix('.pt.download')
        print('Downloading the published paper model (about 125 MB)...', flush=True)
        urllib.request.urlretrieve(manifest['public_url'], temporary)
        if sha256(temporary) != manifest['sha256']:
            temporary.unlink()
            raise RuntimeError('Downloaded model checksum mismatch')
        temporary.replace(args.weights)
    if sha256(args.weights) != manifest['sha256']:
        raise RuntimeError('Model checksum mismatch; use the model specified in model_manifest.json')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    started = time.perf_counter()
    subprocess.run([sys.executable, str(ROOT / 'demo_pipeline.py'), '--weights', str(args.weights),
                    '--output', str(args.output)], check=True)
    summary = json.loads((args.output / 'summary.json').read_text())
    if summary['output_addresses'] != meta['rows'] * meta['cols']:
        raise RuntimeError('Unexpected grid size')
    import importlib.metadata as metadata
    report = {'python': sys.version, 'platform': platform.platform(),
              'versions': {name: metadata.version(name) for name in
                           ['numpy', 'scipy', 'matplotlib', 'ultralytics', 'torch']},
              'model_sha256': sha256(args.weights), 'input_checksums': 'all 10 passed',
              'elapsed_including_subprocess_startup_s': time.perf_counter() - started,
              'scope': 'single-sequence demonstration; not the paper timing benchmark',
              'result': summary}
    (args.output / 'run_environment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Outputs: {args.output.resolve()}')


if __name__ == '__main__':
    main()
