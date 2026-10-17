import glob, os, cv2

R = r"C:\Users\Nausheen\ML_Search_Rescue_Robot\dataset_processed"
names = {0: "Person", 1: "Fire", 2: "Smoke"}
os.makedirs("preview", exist_ok=True)
done = set()

for lp in sorted(glob.glob(os.path.join(R, "labels", "train", "*.txt"))):
    rows = [l.split() for l in open(lp) if l.strip()]
    ids = {int(r[0]) for r in rows}
    if len(ids) != 1:
        continue
    c = int(rows[0][0])
    if c in done:
        continue
    stem = os.path.splitext(os.path.basename(lp))[0]
    img = cv2.imread(os.path.join(R, "images", "train", stem + ".jpg"))
    h, w = img.shape[:2]
    for r in rows:
        x, y, bw, bh = map(float, r[1:5])
        x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
        x2, y2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "class " + str(c) + " = " + names[c], (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out = os.path.join("preview", names[c] + "_" + stem + ".jpg")
    cv2.imwrite(out, img)
    print("saved", out)
    done.add(c)
    if len(done) == 3:
        break