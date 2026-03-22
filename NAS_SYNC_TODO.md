# NAS Sync TODO

## Last 3 Artists Download (2026-03-22)

Gallery cache was cleared from the Modal volume to free space, and the last 3
artists were retried. Check if they succeeded:

```bash
modal volume ls mangaka-data images/takase_yuu/
modal volume ls mangaka-data images/yokkora/
modal volume ls mangaka-data images/yuiga_naoha/
```

If any are still empty, retry:

```python
import json, modal

fetch_fn = modal.Function.from_name('mangaka', 'fetch_artist_batch')

# Gallery URLs for the 3 artists (from the original index)
artists = {
    "takase_yuu": 20,  # galleries
    "yokkora": 20,
    "yuiga_naoha": 20,
}

# Get the index
import subprocess, json
subprocess.run(['modal', 'volume', 'get', 'mangaka-data', 'artist_index.json', '/tmp/artist_index.json'])
with open('/tmp/artist_index.json') as f:
    index = json.load(f)

for name in artists:
    r = fetch_fn.remote(name, index[name])
    print(f"{name}: {r['images']} images — {'OK' if r['error'] is None else r['error']}")
```

## Sync Modal → NAS

Once all 200 artists have images, sync from Modal volume to NAS:

```bash
# Sync new images (the 76 artists that were retried)
modal volume get mangaka-data images/ /Volumes/trigger/mangaka/data/images/ --force

# Style bank was already synced (46 .pt files)
# After new style banks are built, sync again:
modal volume get mangaka-data style_bank/ /tmp/style_bank_sync/ --force
cp /tmp/style_bank_sync/style_bank/*.pt /Volumes/trigger/mangaka/data/style_bank/
```

## Status Summary

- 197/200 artists fetched successfully (101,812 + 129,615 = ~231K images)
- 3 remaining: takase_yuu, yokkora, yuiga_naoha (retry pending — gallery_cache was cleared to free volume space)
- Style bank: 46/200 artists encoded (.pt files synced to NAS)
- Gallery cache: deleted from Modal volume to free space (was 3,029 entries)
