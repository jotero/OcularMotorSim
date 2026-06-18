"""One-off: extract numeric result lines from the near-response reference PDFs."""
import fitz, re, os

FOLDER = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\references\near response'

PAT = re.compile(r'(peak velocity|latency|time const|main sequence|slope|'
                 r'deg/s|°/s|/s/deg|time[- ]to[- ]peak|duration|AC/?A|CA/?C|'
                 r'adaptation|tonic|dark[- ]focus|gain|diopter|prism diopter|'
                 r'\bms\b|\bs\b)', re.I)
NUM = re.compile(r'\d')

TARGETS = {
    'Hung_1997': '1-s2.0-S004269899700271X-main.pdf',
    'Hung_1992_adapt': 'Hung_1992_VerAcc.pdf',
    'Read_Schor_2022': 'Read_Schor_2022_Accommodation.pdf',
    'Schor_Kotulak_1986': 'Schor_Kotulak_1986_VergAccomm.pdf',
    'Zee_1992': 'zee-et-al-1992-saccade-vergence-interactions-in-humans.pdf',
    'Horwood': 'emss-64607.pdf',
}

for tag, fn in TARGETS.items():
    path = os.path.join(FOLDER, fn)
    if not os.path.isfile(path):
        print(f'\n### {tag}: MISSING {fn}'); continue
    doc = fitz.open(path)
    txt = '\n'.join(p.get_text() for p in doc)
    print('\n' + '#' * 92)
    print(f'### {tag}')
    seen = set()
    for ln in txt.split('\n'):
        s = ' '.join(ln.split())
        if 8 <= len(s) <= 170 and PAT.search(s) and NUM.search(s) and s not in seen:
            # keep lines that look like quantitative claims (have a unit-ish token near a number)
            if re.search(r'\d\s*(deg|°|/s|ms|s\b|D\b|MA|prism|Δ|min|%)', s) or re.search(r'(slope|peak velocity|latency|time const|AC/?A|CA/?C).{0,40}\d', s, re.I):
                print(' |', s)
                seen.add(s)
