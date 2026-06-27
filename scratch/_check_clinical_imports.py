"""Which clinical benches still import cleanly after the BrainState refactor?"""
import importlib

for m in ['bench_clinical', 'bench_clinical_vestibular', 'bench_clinical_ni_vs',
          'bench_clinical_cn_palsies', 'bench_clinical_saccades',
          'bench_clinical_vergence', 'bench_clinical_cerebellum']:
    try:
        importlib.import_module(f'oculomotor.benchmarks.{m}')
        print(f'{m:32} OK')
    except Exception as e:
        print(f'{m:32} FAIL: {type(e).__name__}: {e}')
