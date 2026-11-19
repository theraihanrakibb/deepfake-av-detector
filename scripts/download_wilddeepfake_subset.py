"""Download a bounded set of official WildDeepfake face-sequence archives."""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

REPO = 'xingjunm/WildDeepfake'
REVISION = 'f3835aaf281dd9f8d79b51c4e02f050d3f7af0b4'
ROOT = 'https://huggingface.co'


def main():
    output = Path('data/wilddeepfake_official_subset')
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split in ('real_train', 'fake_train', 'real_test', 'fake_test'):
        directory = f'deepfake_in_the_wild/{split}'
        with urlopen(f'{ROOT}/api/datasets/{REPO}/tree/{REVISION}/{directory}', timeout=30) as response:
            rows = json.load(response)
        archives = sorted((row for row in rows if row['path'].endswith('.tar.gz')),
                          key=lambda row: (row['size'], row['path']))[:3]
        if len(archives) != 3:
            raise RuntimeError(f'Expected at least three archives in {split}')
        for row in archives:
            target = output / split / Path(row['path']).name
            target.parent.mkdir(exist_ok=True)
            digest = row['lfs']['oid']
            if not target.exists():
                partial = target.with_suffix('.part')
                url = f'{ROOT}/datasets/{REPO}/resolve/{REVISION}/{row["path"]}'
                with urlopen(url, timeout=120) as response, partial.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
                with partial.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                        raise ValueError(f'Archive checksum mismatch: {row["path"]}')
                partial.replace(target)
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError(f'Existing archive checksum mismatch: {target}')
            manifest.append(dict(revision=REVISION, source=row['path'], bytes=row['size'], sha256=digest))
            print(f'{row["path"]} {row["size"]:,} bytes verified', flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
