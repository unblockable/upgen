# Scripts

This document contains various scripts which can be used to generate input files for training the model.
We only give a few sample scripts here. 
Users can also create their own scripts to generate input files.
Please note that if you are creating your own script, it should create file in which the file should have entries in `abc/xyz` format otherwise it would not work. 
Ensure that the required tools such as curl, gzip, jq, etc. are already installed for running the required scripts.

---

## Script 1

```
curl -L 'https://data.gharchive.org/2015-01-01-15.json.gz' | gzip -dc | jq -r '.repo.name' | head -n 10000 > input_file.txt
```

## Script 2

```
#!/usr/bin/env bash
out=input_file.txt

for h in {0..23}; do
  curl -s "https://data.gharchive.org/2025-06-01-$h.json.gz"
done | gunzip 2>/dev/null \
  | jq -r 'select(.type=="WatchEvent") | .repo.name' 2>/dev/null \
  | awk '!seen[$0]++' \
  | head -10000 > "$out"

wc -l "$out"
```

---
