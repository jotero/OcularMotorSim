"""Which schema entries are stale (no matching param) and which new params lack
schema enrichment — so we can clear the 'nothing stale' ask."""
from importlib.resources import files
from oculomotor.reports import gen_parameters as gp
from oculomotor.params import BrainParams, SensoryParams, PlantParams

schema_yml = str(files("oculomotor.schema") / "parameters_schema.yaml")
schema, _ = gp._load_schema(schema_yml)

code_keys = set()
for ns, cls in (('brain', BrainParams), ('sensory', SensoryParams), ('plant', PlantParams)):
    for name, *_ in gp._extract_fields(cls):
        code_keys.add(f'{ns}.{name}')
schema_keys = set(schema.keys()) if isinstance(schema, dict) else set()

print('STALE (in schema, no current param):')
for k in sorted(schema_keys - code_keys):
    print('   ', k)
print('\nNEW PARAMS of interest:')
for k in ('brain.mn_ff_yaw', 'brain.mlf_lead'):
    print(f'    {k}: {"MISSING (TODO)" if k not in schema_keys else "has schema"}')
print('\nALL missing (TODO):')
for k in sorted(code_keys - schema_keys):
    print('   ', k)
print(f'\ntotals: code={len(code_keys)} schema={len(schema_keys)} '
      f'missing={len(code_keys - schema_keys)} stale={len(schema_keys - code_keys)}')
