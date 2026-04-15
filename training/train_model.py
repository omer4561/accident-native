"""
מאמן רשת נוירונים לזיהוי ספרות בכתב יד עברי/אנגלי
======================================================
בנוי מ-0 — ללא TensorFlow/PyTorch, רק NumPy + Pillow

שימוש:
    pip install numpy pillow
    python train_model.py

פלט:
    ocr_weights.json  — משקלות הרשת
    (הדבק את התוכן כ-const OCR_WEIGHTS = ... ב-index.html
     ואז קרא ל- engine.loadWeights(JSON.stringify(OCR_WEIGHTS)))
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json, random, os, sys

# ═══════════════════════════════════════════════════
# הגדרות
# ═══════════════════════════════════════════════════
IMG_SIZE   = 32       # גודל תמונה מנורמל (px)
N_CLASSES  = 10       # ספרות 0-9
SAMPLES    = 2500     # דוגמאות לכל ספרה
EPOCHS     = 80
LR         = 0.008
BATCH      = 64
HIDDEN     = [256, 128]   # שכבות נסתרות

# ═══════════════════════════════════════════════════
# יצירת נתוני אימון סינתטיים
# ═══════════════════════════════════════════════════

def get_fonts(size):
    """מנסה להשיג גופנים שונים מהמערכת — fallback ל-default"""
    candidates = [
        'arial.ttf', 'Arial.ttf', 'arialbd.ttf', 'ArialBD.ttf',
        'verdana.ttf', 'Verdana.ttf', 'times.ttf', 'Times.ttf',
        'cour.ttf', 'Courier.ttf', 'georgia.ttf', 'Georgia.ttf',
        'impact.ttf', 'Impact.ttf', 'calibri.ttf', 'Calibri.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/System/Library/Fonts/Arial.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
    ]
    fonts = []
    for path in candidates:
        try:
            fonts.append(ImageFont.truetype(path, size))
            if len(fonts) >= 6: break
        except:
            pass
    if not fonts:
        fonts = [ImageFont.load_default()]
    return fonts

def render_digit(digit, font, size=IMG_SIZE):
    """מייצר תמונת ספרה נקייה"""
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)
    txt = str(digit)
    try:
        bbox = draw.textbbox((0,0), txt, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x = (size - tw)//2 - bbox[0]
        y = (size - th)//2 - bbox[1]
    except AttributeError:
        # PIL ישן
        tw, th = draw.textsize(txt, font=font)
        x = (size - tw)//2
        y = (size - th)//2
    draw.text((x, y), txt, fill=0, font=font)
    return img

def augment(img):
    """סימולציה של כתב יד: סיבוב, קנה-מידה, גזירה, רעש, טשטוש"""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape
    size = max(h, w)

    # סיבוב אקראי ±25°
    angle = random.uniform(-25, 25)
    img_r = img.rotate(angle, fillcolor=255, resample=Image.BILINEAR)

    # שינוי קנה-מידה 0.65–1.30
    scale = random.uniform(0.65, 1.30)
    new_size = (max(4, int(w*scale)), max(4, int(h*scale)))
    img_r = img_r.resize(new_size, Image.BILINEAR)
    nw, nh = img_r.size

    # מרכוז בחזרה על גודל מקורי
    canvas = Image.new('L', (w, h), 255)
    ox = max(0, (w - nw)//2)
    oy = max(0, (h - nh)//2)
    paste = img_r.crop((0, 0, min(nw, w), min(nh, h)))
    canvas.paste(paste, (ox, oy))
    img_r = canvas

    # גזירה אופקית (shear) ±0.30
    shear = random.uniform(-0.30, 0.30)
    data = (1, shear, -shear * h / 2, 0, 1, 0)
    img_r = img_r.transform(img_r.size, Image.AFFINE, data, fillcolor=255)

    # עיבוי/דקיקות של הקו (stroke width simulation)
    if random.random() < 0.4:
        img_r = img_r.filter(ImageFilter.MaxFilter(3))   # עיבוי
    elif random.random() < 0.3:
        img_r = img_r.filter(ImageFilter.MinFilter(3))   # דקיקות

    # רעש Gaussian
    arr = np.array(img_r).astype(np.float32)
    arr += np.random.randn(*arr.shape) * random.uniform(5, 25)
    arr = np.clip(arr, 0, 255)

    # טשטוש קל (מדמה עט)
    if random.random() < 0.35:
        img_r = Image.fromarray(arr.astype(np.uint8))
        img_r = img_r.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
        arr = np.array(img_r).astype(np.float32)

    return arr.astype(np.uint8)

def build_dataset(samples_per_class, size=IMG_SIZE):
    fonts = get_fonts(int(size * 0.85))
    print(f"  [נתונים] גופנים שנמצאו: {len(fonts)}, דוגמאות לספרה: {samples_per_class}")
    X, y = [], []
    for digit in range(10):
        for _ in range(samples_per_class):
            font = random.choice(fonts)
            img  = render_digit(digit, font, size)
            aug  = augment(img)
            # בינאריזציה + נורמליזציה ל-[0,1]
            flat = (aug < 128).astype(np.float32).flatten()
            X.append(flat)
            y.append(digit)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    # ערבוב
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

# ═══════════════════════════════════════════════════
# רשת נוירונים MLP מאפס
# ═══════════════════════════════════════════════════

class MLP:
    def __init__(self, layer_sizes):
        self.W = []
        self.b = []
        self.acts = []
        for i in range(len(layer_sizes)-1):
            fan_in = layer_sizes[i]
            # He initialization
            w = np.random.randn(fan_in, layer_sizes[i+1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros(layer_sizes[i+1])
            self.W.append(w)
            self.b.append(b)
            act = 'softmax' if i == len(layer_sizes)-2 else 'relu'
            self.acts.append(act)

    # ── forward ──
    def forward(self, X):
        self._cache = [X]
        h = X
        for w, b, act in zip(self.W, self.b, self.acts):
            z = h @ w + b
            if act == 'relu':
                h = np.maximum(0, z)
            else:  # softmax
                z -= z.max(axis=1, keepdims=True)
                e  = np.exp(z)
                h  = e / e.sum(axis=1, keepdims=True)
            self._cache.append(h)
        return self._cache[-1]

    # ── loss: cross-entropy ──
    def loss(self, probs, y_true):
        N = len(y_true)
        log_p = -np.log(probs[np.arange(N), y_true] + 1e-12)
        return log_p.mean()

    def accuracy(self, probs, y_true):
        return (probs.argmax(axis=1) == y_true).mean()

    # ── backward ──
    def backward(self, y_true, lr):
        N  = len(y_true)
        dh = self._cache[-1].copy()
        dh[np.arange(N), y_true] -= 1
        dh /= N

        for i in range(len(self.W)-1, -1, -1):
            h_prev = self._cache[i]
            dW = h_prev.T @ dh
            db = dh.sum(axis=0)
            if i > 0:
                dh = dh @ self.W[i].T
                dh *= (self._cache[i] > 0).astype(float)  # relu grad
            self.W[i] -= lr * dW
            self.b[i]  -= lr * db

    # ── train one epoch ──
    def train_epoch(self, X, y, batch_size, lr):
        idx = np.random.permutation(len(X))
        total_loss = 0.0
        for s in range(0, len(X), batch_size):
            bi = idx[s:s+batch_size]
            probs = self.forward(X[bi])
            total_loss += self.loss(probs, y[bi]) * len(bi)
            self.backward(y[bi], lr)
        return total_loss / len(X)

    # ── export as JSON for browser ──
    def to_json(self):
        layers = []
        for w, b, act in zip(self.W, self.b, self.acts):
            layers.append({
                'W':          w.tolist(),
                'b':          b.tolist(),
                'activation': act
            })
        return json.dumps({'layers': layers}, separators=(',', ':'))

# ═══════════════════════════════════════════════════
# אימון
# ═══════════════════════════════════════════════════

def main():
    print("=== בניית מנוע OCR מאפס ===\n")
    np.random.seed(42)
    random.seed(42)

    print("שלב 1/3 — יוצר נתוני אימון...")
    X, y = build_dataset(SAMPLES, IMG_SIZE)
    split = int(len(X) * 0.9)
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]
    print(f"  אימון: {len(X_train)}, ולידציה: {len(X_val)}\n")

    print("שלב 2/3 — מאמן רשת נוירונים...")
    sizes = [IMG_SIZE*IMG_SIZE] + HIDDEN + [N_CLASSES]
    print(f"  ארכיטקטורה: {' → '.join(map(str, sizes))}")
    net = MLP(sizes)

    best_val  = 0.0
    best_w    = None
    lr        = LR
    no_improve = 0

    for epoch in range(1, EPOCHS+1):
        loss = net.train_epoch(X_train, y_train, BATCH, lr)
        val_probs = net.forward(X_val)
        val_acc   = net.accuracy(val_probs, y_val)

        if val_acc > best_val:
            best_val = val_acc
            best_w   = [w.copy() for w in net.W], [b.copy() for b in net.b]
            no_improve = 0
        else:
            no_improve += 1

        # learning rate decay
        if no_improve == 12:
            lr *= 0.5
            no_improve = 0
            print(f"  ↓ lr={lr:.5f}")

        if epoch % 10 == 0 or epoch == 1:
            train_probs = net.forward(X_train[:2000])
            tr_acc = net.accuracy(train_probs, y_train[:2000])
            print(f"  epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  "
                  f"train={tr_acc*100:.1f}%  val={val_acc*100:.1f}%  "
                  f"(best={best_val*100:.1f}%)")

    # שחזור המשקלות הטובות ביותר
    if best_w:
        net.W, net.b = best_w

    # הערכה סופית
    test_probs = net.forward(X_val)
    print(f"\n✅ דיוק סופי על ולידציה: {net.accuracy(test_probs, y_val)*100:.1f}%")
    print(f"   (כתב יד קריא צפוי להגיע ל-85-92%)\n")

    # per-class accuracy
    for d in range(10):
        mask   = y_val == d
        if mask.sum():
            acc = net.accuracy(test_probs[mask], y_val[mask])
            print(f"   ספרה {d}: {acc*100:.0f}%")

    print("\nשלב 3/3 — מייצא משקלות...")
    js = net.to_json()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'www', 'ocr-weights.json')
    out_path = os.path.normpath(out_path)
    with open(out_path, 'w') as f:
        f.write(js)
    kb = len(js) // 1024
    print(f"  נשמר: {out_path} ({kb} KB)")
    print("""
┌─────────────────────────────────────────────────────────┐
│  איך לחבר לאפליקציה:                                    │
│                                                         │
│  pages.yml יטען אוטומטית את ocr-weights.json           │
│  (הוא נמצא ב-www/ ולכן מועלה ל-GitHub Pages)           │
│                                                         │
│  כל שנדרש: git add + git commit + git push              │
└─────────────────────────────────────────────────────────┘
""")

if __name__ == '__main__':
    main()
