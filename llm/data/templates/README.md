# Real FIR templates go here

Once real FIR templates are collected (per the project workflow: "Collect
1,000+ FIR templates" from BLAST / police records), add them here as a
JSONL file, one example per line, in exactly the shape produced by
`src.data_generation.generate_examples`:

```json
{"input_text": "<informal / sign-language-derived Bangla text>", "output_json": {"offense_type": "চুরি", "complainant": {"name": "..."}, "incident_location": "...", "narrative_bn": "...", "...": "... (see src/schema.py FIRRecord for all fields)"}}
```

Then mix real examples into the training split alongside the synthetic
bootstrap data with:

```bash
python -m src.data_generation --n 1200 --out-dir data/processed \
    --extra-jsonl data/templates/real_examples.jsonl
```

`output_json` must validate against `src.schema.FIRRecord` -- run
`python -c "from src.schema import FIRRecord; import json; FIRRecord.model_validate(json.loads(open('data/templates/real_examples.jsonl').readline())['output_json'])"`
to sanity check a file before mixing it in.
