#!/usr/bin/env python3
"""
Run this before every zip build to auto-increment the patch version.
Usage: python3 bump_version.py
"""
import re, sys

path = 'src/App.tsx'
src = open(path).read()

m = re.search(r'v(\d+)\.(\d+)', src)
if not m:
    print('ERROR: version string not found'); sys.exit(1)

major, minor = int(m.group(1)), int(m.group(2))
new_minor = minor + 1
old = f'v{major}.{minor}'
new = f'v{major}.{new_minor}'

src = src.replace(old, new, 1)
open(path, 'w').write(src)
print(f'Bumped {old} → {new}')
