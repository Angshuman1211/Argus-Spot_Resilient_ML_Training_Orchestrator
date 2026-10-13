"""
Checks the paper source against the project's writing rules:
no em or en dashes (and no double or triple hyphens as dashes), section references written as
Section~\\ref{...} (never \\S, never a typed number), and a count of colons and semicolons in prose.

    python Paper/v2/check_style.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "argus_v2.tex"), encoding="utf-8").read()
ok = True


def report(name, found):
    global ok
    bad = bool(found)
    ok &= not bad
    print(f"{'FAIL' if bad else 'ok  '}  {name}" + (f": {found[:6]}" if bad else ""))


report("em dash", [m for m in re.findall("\u2014", src)])
report("en dash", [m for m in re.findall("\u2013", src)])
report("--- or -- used as a dash", re.findall(r"(?<![-\w])-{2,3}(?![-\w])", re.sub(r"%.*", "", src)))
report(r"\S used for sections", re.findall(r"\\S\\ref|\\S\s", src))
report("typed section number", re.findall(r"Sections?\s+\d", src))
refs = re.findall(r"Sections?~\\ref\{[^}]+\}", src)
print(f"info  {len(refs)} section references written as Section~\\ref{{...}}")

body = re.sub(r"\\caption\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", "", src, flags=re.S)
body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{[^{}]*\}", " ", body)
body = re.sub(r"%.*", "", body)
body = body.split("\\begin{document}")[-1]
print(f"info  semicolons in prose: {body.count(';')}")
colons = [m.group(0).strip() for m in re.finditer(r"[^\n]{0,45}[A-Za-z\)]:\s[^\n]{0,30}", body)]
print(f"info  colons in prose: {len(colons)}")
for c in colons[:10]:
    print("      ..", c)
sys.exit(0 if ok else 1)
