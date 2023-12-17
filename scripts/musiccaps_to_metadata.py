# %%
import os
import pandas as pd
data = pd.read_csv("data/dataset/musicset/musiccaps-public.csv")

# %%
metas = []
base_dir = "wavs/"
for i in range(len(data)):
    entity = data.iloc[i]
    metas.append(
        dict(
            wav=os.path.join(base_dir, f"{entity['ytid']}.wav"),
            seg_label="",
            labels="",
            caption=f"{entity['caption']}",
        )
    )

# %%
metas[0]

# %%
import json
train_metas = metas[:-200]
valid_metas = metas[-200:-100]
test_metas = metas[-100:]
label_file_root = "data/dataset/metadata/musiccaps/"
with open(os.path.join(label_file_root, "train.json"), "w") as f:
    json.dump(dict(data=train_metas), f, indent=4)
with open(os.path.join(label_file_root, "valid.json"), "w") as f:
    json.dump(dict(data=valid_metas), f, indent=4)
with open(os.path.join(label_file_root, "test.json"), "w") as f:
    json.dump(dict(data=test_metas), f, indent=4)

# %%
