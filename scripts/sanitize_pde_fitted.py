#!/usr/bin/env python3
import yaml
import itertools
from pathlib import Path

fpath = Path('data/PDE_fitted.yml')
if not fpath.exists():
    print('No data/PDE_fitted.yml found')
    raise SystemExit(1)

doc = yaml.safe_load(fpath.read_text()) or {}

new_root = {}

# Lift any existing 'pairs' entries
if isinstance(doc, dict) and 'pairs' in doc and isinstance(doc['pairs'], dict):
    for k, v in doc['pairs'].items():
        new_root[k] = v

# If doc contains legacy 'data' mapping, construct per-pair entries for each pair of institutes
if isinstance(doc, dict) and 'data' in doc and isinstance(doc['data'], dict):
    data = doc['data']
    # detect institutes by *_PDE keys
    institutes = [k[:-4] for k in data.keys() if k.endswith('_PDE')]
    institutes = sorted(set(institutes))
    # build per-pair entries
    for a, b in itertools.combinations(institutes, 2):
        pair = f"{a}_{b}"
        if pair in new_root:
            continue
        entry = {'selected_institutes': [a, b], 'data': {}, 'coefficients': {}}
        # include OV
        if 'OV' in data:
            entry['data']['OV'] = list(data['OV'])
        # include columns for both institutes (PDE and err)
        for inst in (a, b):
            for suffix in ('_PDE', '_err'):
                key = f"{inst}{suffix}"
                if key in data:
                    entry['data'][key] = list(data[key])
        # include combined if present
        if 'Combined' in data:
            entry['data']['Combined'] = list(data['Combined'])
        new_root[pair] = entry

# Also lift any existing top-level pair-like keys (containing '_')
if isinstance(doc, dict):
    for k, v in doc.items():
        if '_' in k and isinstance(v, dict):
            new_root.setdefault(k, v)

# Write sanitized file with only per-pair entries
fpath.write_text(yaml.dump(new_root, default_flow_style=False))
print(f'Sanitized {fpath}, wrote {len(new_root)} pair entries')
