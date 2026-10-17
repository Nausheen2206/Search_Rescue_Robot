import glob, hashlib, os, shutil, sys

SRC = r"C:\Users\Nausheen\ML_Search_Rescue_Robot\dataset\FSP_detection_dataset_3cls\All_Data"
DST = r"C:\Users\Nausheen\ML_Search_Rescue_Robot\dataset_processed"
GO = "--go" in sys.argv

def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

train = sorted(glob.glob(SRC + r"\train\images\*.jpg"))
val = sorted(glob.glob(SRC + r"\val\images\*.jpg"))
train_hashes = {md5(p) for p in train}

keep_val, seen, leak, dup = [], set(), 0, 0
for p in val:
    h = md5(p)
    if h in train_hashes:
        leak += 1
    elif h in seen:
        dup += 1
    else:
        seen.add(h)
        keep_val.append(p)

print("train images kept:", len(train))
print("val images original:", len(val))
print("val removed (identical to train):", leak)
print("val removed (duplicate inside val):", dup)
print("val images kept:", len(keep_val))

if not GO:
    print("DRY RUN - nothing copied. Run with --go to copy.")
    sys.exit()

if os.path.exists(DST):
    sys.exit("Destination already exists, stopping: " + DST)

for split, imgs in (("train", train), ("val", keep_val)):
    os.makedirs(DST + "\\images\\" + split)
    os.makedirs(DST + "\\labels\\" + split)
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        shutil.copy2(p, DST + "\\images\\" + split)
        shutil.copy2(SRC + "\\" + split + "\\labels\\" + stem + ".txt", DST + "\\labels\\" + split)
print("Done copying.")