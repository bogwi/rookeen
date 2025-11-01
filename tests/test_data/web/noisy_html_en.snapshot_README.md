# Updating the `noisy_html_en` Snapshot

These steps regenerate the stored snapshot for `noisy_html_en.txt` using the same CLI + normalization flow exercised by the tests.

## 1. Regenerate Raw CLI Output

Run the analyzer on the raw HTML fixture and capture the JSON stdout:

```bash
uv run rookeen analyze --stdin --lang en --stdout < tests/test_data/web/noisy_html_en.txt > tests/test_data/web/noisy_html_en.snapshot.json
```

## 2. Normalize Snapshot Structure

Strip volatile fields (e.g., timestamps) and round floats so the snapshot is stable between runs:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

snap_path = Path('tests/test_data/web/noisy_html_en.snapshot.json')

with snap_path.open(encoding='utf-8') as fh:
    payload = json.load(fh)


def _round(value, ndigits=6):
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, list):
        return [_round(item, ndigits) for item in value]
    if isinstance(value, dict):
        return {key: _round(val, ndigits) for key, val in value.items()}
    return value


normalized_analyzers = {}
for analyzer in payload.get('analyzers', []):
    name = analyzer.get('name')
    if not name:
        continue
    normalized_analyzers[name] = {
        'confidence': analyzer.get('confidence'),
        'metadata': _round(analyzer.get('metadata', {})),
        'results': _round(analyzer.get('results', {})),
    }

source = dict(payload.get('source', {}))
source.pop('fetched_at', None)

normalized = {
    'language': {
        'code': payload.get('language', {}).get('code'),
        'model': payload.get('language', {}).get('model'),
    },
    'content': {
        'title': payload.get('content', {}).get('title'),
        'char_count': payload.get('content', {}).get('char_count'),
        'word_count': payload.get('content', {}).get('word_count'),
    },
    'source': source,
    'analyzers': normalized_analyzers,
}

with snap_path.open('w', encoding='utf-8') as fh:
    json.dump(normalized, fh, indent=2, ensure_ascii=False, sort_keys=True)
PY
```

## 3. Verify Tests

Confirm the snapshot aligns with the expectations:

```bash
uv run pytest -q tests/e2e/test_web_local_text.py -v
```

If the test now fails due to intentional pipeline changes, review the diff of `noisy_html_en.snapshot.json`, ensure the changes are expected, and commit both the snapshot and relevant code updates.

