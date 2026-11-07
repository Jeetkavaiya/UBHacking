#!/usr/bin/env bash
# Downloads NASA SMAP/MSL data into ./data (needs a Kaggle API key in ~/.kaggle/kaggle.json).
set -euo pipefail
mkdir -p data && cd data
pip install -q kaggle
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl -p .
unzip -oq nasa-anomaly-detection-dataset-smap-msl.zip && rm nasa-anomaly-detection-dataset-smap-msl.zip
# The zip may nest folders; move train/ and test/ up to ./data
for d in train test; do
  if [ ! -d "$d" ]; then
    found=$(find . -mindepth 2 -type d -name "$d" | head -1)
    [ -n "$found" ] && mv "$found" .
  fi
done
if [ ! -f labeled_anomalies.csv ]; then
  found=$(find . -name labeled_anomalies.csv | head -1)
  if [ -n "$found" ]; then mv "$found" .; else
    curl -sL -o labeled_anomalies.csv https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
  fi
fi
echo "train: $(ls train | wc -l) files, test: $(ls test | wc -l) files"
