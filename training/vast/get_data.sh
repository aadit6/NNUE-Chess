# Download the training data into RAM (/dev/shm) on the GPU box: six binpacks of 500M
# depth-9 Stockfish-scored positions, ~7.4 GB, in parallel.
set -eo pipefail
mkdir -p /dev/shm/data
cd /dev/shm/data
python3 - > files.txt <<'PY'
import json, re, urllib.request
tree = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/jshriver/historic-binpacks/tree/main"))
for item in tree:
    if re.fullmatch(r"500M_d9_p\d+\.binpack", item["path"]):
        print(item["path"], item["size"])
PY
cat files.txt
start=$(date +%s)
while read -r name size; do
  (curl -sSL --retry 5 -o "$name" "https://huggingface.co/datasets/jshriver/historic-binpacks/resolve/main/$name" \
     && echo "done $name: $(stat -c %s "$name") of $size bytes") &
done < files.txt
wait
echo "downloaded in $(( $(date +%s) - start )) s"
ls -la /dev/shm/data
echo DATA OK
