"""Compares conduit_bench --json results.

  bench_compare.py before.json after.json     one row per api and measurement, with the change between the two
  bench_compare.py results.json               one row per measurement, one column per api

Medians are compared. Rows in instr/draw are exact; timings move with machine load, so read small changes in them
as noise unless several runs agree.
"""
import json
import sys


def load(path):
    with open(path, encoding='utf-8') as f:
        return {(r['api'], r['name'], r['unit']): r for r in json.load(f)}


def fmt(value):
    return f'{value:,.1f}' if abs(value) < 100000 else f'{value:,.0f}'


def compare(before, after):
    print(f'{"api":8}{"measurement":30}{"unit":11}{"before":>14}{"after":>14}{"change":>9}{"p99 before":>14}{"p99 after":>14}')
    for key, b in before.items():
        a = after.get(key)
        if not a:
            continue
        change = (a['median'] - b['median']) / b['median'] * 100 if b['median'] else 0.0
        print(f'{key[0]:8}{key[1]:30}{key[2]:11}{fmt(b["median"]):>14}{fmt(a["median"]):>14}{change:>+8.1f}%'
              f'{fmt(b["p99"]):>14}{fmt(a["p99"]):>14}')


def matrix(rows):
    apis = list(dict.fromkeys(k[0] for k in rows))
    names = list(dict.fromkeys((k[1], k[2]) for k in rows))
    print(f'{"measurement":30}{"unit":11}' + ''.join(f'{a:>14}' for a in apis))
    for name, unit in names:
        cells = [rows.get((a, name, unit)) for a in apis]
        print(f'{name:30}{unit:11}' + ''.join(f'{fmt(c["median"]) if c else "-":>14}' for c in cells))


def main():
    if len(sys.argv) == 3:
        compare(load(sys.argv[1]), load(sys.argv[2]))
    elif len(sys.argv) == 2:
        matrix(load(sys.argv[1]))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
