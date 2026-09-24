#!/usr/bin/env python3
"""Offline validation; never imports experiment code or starts a worker."""
from pathlib import Path
import ast
import hashlib
import json
import sys

root = Path(__file__).resolve().parent
errors = []
counts = {'hashed_files': 0, 'python_files': 0, 'notebooks': 0, 'notebook_code_cells': 0}
manifest = root / 'SHA256SUMS'
if not manifest.exists():
    errors.append('Missing SHA256SUMS')
else:
    for line in manifest.read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append('Hash mismatch or missing: ' + name)
        counts['hashed_files'] += 1
for path in root.rglob('*.py'):
    try:
        ast.parse(path.read_bytes(), filename=str(path.relative_to(root)))
        counts['python_files'] += 1
    except Exception as exc:
        errors.append(str(path.relative_to(root)) + ': ' + str(exc))
for path in root.rglob('*.ipynb'):
    try:
        nb = json.loads(path.read_text())
        assert nb['nbformat'] == 4 and isinstance(nb['cells'], list)
        for cell in nb['cells']:
            assert cell['cell_type'] in ('code', 'markdown', 'raw')
            source = cell['source']
            assert isinstance(source, (str, list))
            if cell['cell_type'] == 'code':
                assert isinstance(cell.get('outputs', []), list)
                counts['notebook_code_cells'] += 1
        counts['notebooks'] += 1
    except Exception as exc:
        errors.append(str(path.relative_to(root)) + ': ' + str(exc))
print(json.dumps({'pass': not errors, 'checks': counts, 'errors': errors,
                  'scope': 'Integrity, Python syntax and notebook structure only; no ML imports or scientific rerun.'}, indent=2))
sys.exit(1 if errors else 0)
