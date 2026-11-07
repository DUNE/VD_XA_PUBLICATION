#!/usr/bin/env python3
import yaml
from pathlib import Path

fpath = Path('data/PDE_combined.yml')
if not fpath.exists():
    print('No data/PDE_combined.yml found')
    raise SystemExit(1)

doc = yaml.safe_load(fpath.read_text()) or {}

# If already in the new format, keep only the top-level pair entries.
if isinstance(doc, dict) and doc and all(isinstance(v, dict) for v in doc.values()):
    pairs = doc
else:
    # Legacy format: try to extract a pair entry from the nested layout.
    pairs = {}
    if isinstance(doc, dict):
        if 'pairs' in doc and isinstance(doc['pairs'], dict):
            pairs.update(doc['pairs'])
        for k, v in doc.items():
            if '_' in k and isinstance(v, dict):
                pairs.setdefault(k, v)

if not pairs:
    print('No pair entries found to sanitize')
    raise SystemExit(1)

fpath.write_text(yaml.dump(pairs, default_flow_style=False))
print(f'Sanitized {fpath} with {len(pairs)} pair entries')
