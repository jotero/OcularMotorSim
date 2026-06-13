"""Bundled configuration schemas (package-data).

parameters_schema.yaml  parameter anatomy + disorder tags (read by patient_builder,
                        server, gen_parameters).
states_schema.yaml      state directory (read by gen_states).

Read via importlib.resources, e.g.::
    from importlib.resources import files
    (files('oculomotor.schema') / 'parameters_schema.yaml').open(encoding='utf-8')
"""
