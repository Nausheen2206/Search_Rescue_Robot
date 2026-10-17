
import glob, os, random, sys
import cv2, numpy as np

R = r"C:\Users\Nausheen\ML_Search_Rescue_Robot\dataset_processed"
cls = int(sys.argv[1])
n = int(sys.argv[2])
random.seed(0)

files = []
for lp in glob.glob(os.path.join(R, "labels", "train", "*.txt")):
    rows = [l.split() for l in open(lp) if l.strip()]
    if any(int(r[0]) == cls for r in rows):
        files.append((lp, rows))
random.shuffle(files)

tiles = []
for i, (lp, rows) in enumerate(files[:n]):
    stem = os.path.splitext(os.path.basename(lp))[0]
    img = cv2.imread(os.path.join(R, "images", "train", stem + ".jpg"))
    h, w = img.shape[:2]
    for r in rows:
        if int(r[0]) != cls:
            continue
        x, y, bw, bh = map(float, r[1:5])
        p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
        p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 3)
    img = cv2.resize(img, (320, 240))
    cv2.putText(img, str(i), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    tiles.append(img)
    print(i, stem)

while len(tiles) % 4:
    tiles.append(np.zeros((240, 320, 3), np.uint8))
sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
out = "sheet_class" + str(cls) + ".jpg"
cv2.imwrite(out, sheet)
print("saved", out)