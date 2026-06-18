"""One-off: catalog every PDF under references/ with its title head (for renaming)."""
import fitz, glob, os

ROOT = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\references'
for path in sorted(glob.glob(os.path.join(ROOT, '**', '*.pdf'), recursive=True)):
    rel = os.path.relpath(path, ROOT)
    try:
        doc = fitz.open(path)
        head = ' '.join('\n'.join(p.get_text() for p in doc[:2]).split())[:300]
        print(f'\n{rel}  ({doc.page_count} pp)\n  {head}')
    except Exception as e:
        print(f'\n{rel}\n  ERROR {e}')
