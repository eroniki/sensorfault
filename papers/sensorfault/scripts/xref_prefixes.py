#!/usr/bin/env python3
"""Re-derive the xr prefixes on cross-document references.

The letter (main.tex) and the supplement (supplement.tex) are separate documents linked by xr.
In the letter a reference to a supplement label is written S-<label>; in the supplement a
reference to a letter label is written M-<label>. Run this after moving text between them.
"""
import glob, re, os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
MAIN = ['main_viewgate.tex'] + ['sections/vg_%s.tex' % n for n in 'intro related method setup results conclusion'.split()]
SUPP = ['supplement.tex'] + sorted(glob.glob('sections/supp_*.tex'))
FIGS = glob.glob('figures/**/*.tex', recursive=True)
REF = re.compile(r'\\(ref|eqref|pageref)\{([^}]+)\}')

def own_labels(text):
    s = set(re.findall(r'\\label\{([^}]+)\}', text))
    s |= {'fig:' + x for x in re.findall(r'\\todofig\{([^}]+)\}', text)}
    return s

def defs(files):
    text = ''.join(open(f).read() for f in files)
    out = own_labels(text)
    for g in FIGS:
        if re.search(r'/' + re.escape(os.path.basename(g)[:-4]) + r'(\.tex)?\}', text):
            out |= own_labels(open(g).read())
    return out

M, S = defs(MAIN), defs(SUPP)
missing = []
for files, here, there, prefix in ((MAIN, M, S, 'S-'), (SUPP, S, M, 'M-')):
    for f in files:
        t = open(f).read()
        def fix(m):
            keys = []
            for k in m.group(2).split(','):
                k = re.sub(r'^[SM]-', '', k.strip())
                if k in here: keys.append(k)
                elif k in there: keys.append(prefix + k)
                else: keys.append(k); missing.append((f, k))
            return '\\%s{%s}' % (m.group(1), ','.join(keys))
        t2 = REF.sub(fix, t)
        if t2 != t: open(f, 'w').write(t2)
for f, k in missing: print('undefined:', f, k)
