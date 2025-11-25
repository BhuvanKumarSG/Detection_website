import sys
from pathlib import Path

def dump(path, n=256):
    p = Path(path)
    if not p.exists():
        print('File not found:', path); return
    b = p.stat().st_size
    print('Path:', path)
    print('Size:', b, 'bytes')
    with p.open('rb') as f:
        head = f.read(n)
    print('\nFirst', n, 'bytes (hex):')
    print(head.hex()[:1000])
    print('\nFirst bytes (ascii, non-printable shown as .):')
    s = ''.join((chr(c) if 32 <= c <= 126 else '.') for c in head)
    print(s)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python dump_ckpt_header.py <path> [bytes]')
        raise SystemExit(1)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    dump(sys.argv[1], n)
