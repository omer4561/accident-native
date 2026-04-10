/* ================================================================
   מנוע OCR מותאם — בנוי מ-0 ללא תלויות חיצוניות
   ----------------------------------------------------------------
   יכולות:
     • עיבוד תמונה: grayscale, adaptive-threshold, noise removal
     • פיצול תווים: connected-components + vertical-projection
     • זיהוי ספרות: template-matching (מובנה) + רשת נוירונים (אופציונלי)
     • חילוץ שדות: תאריך ולוחית-רישוי מטופס 33.1106
   ================================================================ */

'use strict';

class OCREngine {

  /* ══════════════════ PUBLIC API ══════════════════ */

  constructor() {
    this._S      = 32;      // גודל תבנית נורמלי
    this._tmpls  = null;    // תבניות ייחוס (נבנות ב-init)
    this._nn     = null;    // משקלות רשת נוירונים (אופציונלי)
    this._ready  = false;
  }

  /** אתחול — בונה תבניות ייחוס מ-Canvas, ללא הורדות */
  async init(onProg) {
    if (this._ready) return;
    onProg?.(5,  'בונה בסיס תבניות...');
    this._tmpls = await this._buildTemplates();
    onProg?.(100, '✅ מנוע מוכן');
    this._ready = true;
  }

  /**
   * טען משקלות רשת נוירונים שיוצרו ע"י training/train_model.py
   * קריאה אופציונלית — בלעדיה המנוע עובד עם template-matching
   */
  loadWeights(jsonStr) {
    try { this._nn = JSON.parse(jsonStr); } catch(e) { console.warn('OCR weights parse error', e); }
  }

  /**
   * נתח תמונה — מחזיר { date, license_plate, driver_name }
   * @param {File} file
   * @param {Function} onProg  (pct:0-100, msg:string)=>void
   */
  async analyzeImage(file, onProg) {
    if (!this._ready) await this.init(onProg);

    onProg?.(5,  'טוען תמונה...');
    const { px, W, H } = await this._loadPixels(file, 1800);

    onProg?.(15, 'ממיר שחור-לבן...');
    const gray  = this._toGray(px, W, H);
    const bin   = this._adaptiveThreshold(gray, W, H);
    const clean = this._removeNoise(bin, W, H);

    onProg?.(35, 'מחפש תאריך...');
    const date  = this._findDate(clean, W, H);

    onProg?.(65, 'מחפש לוחית...');
    const plate = this._findPlate(clean, W, H);

    onProg?.(100, '✅ הושלם');
    return { date: date || null, license_plate: plate || null, driver_name: null };
  }

  /* ══════════════════ IMAGE LOADING ══════════════════ */

  _loadPixels(file, maxSide) {
    return new Promise((res, rej) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        let W = img.naturalWidth, H = img.naturalHeight;
        if (Math.max(W, H) > maxSide) {
          const s = maxSide / Math.max(W, H);
          W = Math.round(W * s); H = Math.round(H * s);
        }
        const c = document.createElement('canvas');
        c.width = W; c.height = H;
        c.getContext('2d').drawImage(img, 0, 0, W, H);
        const px = c.getContext('2d').getImageData(0, 0, W, H).data;
        res({ px, W, H });
      };
      img.onerror = () => { URL.revokeObjectURL(url); rej(new Error('load failed')); };
      img.src = url;
    });
  }

  /* ══════════════════ PREPROCESSING ══════════════════ */

  _toGray(px, W, H) {
    const g = new Uint8Array(W * H);
    for (let i = 0; i < W * H; i++)
      g[i] = (0.299 * px[i*4] + 0.587 * px[i*4+1] + 0.114 * px[i*4+2] + 0.5) | 0;
    return g;
  }

  /**
   * Bradley-Roth adaptive threshold:
   * pixel is "ink" (1) if brightness < localMean * T
   */
  _adaptiveThreshold(gray, W, H) {
    const T = 0.84, R = 24;
    const bin = new Uint8Array(W * H);
    // integral image (1-indexed, padded)
    const I = new Int32Array((W+1) * (H+1));
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const p = (y+1)*(W+1)+(x+1);
        I[p] = gray[y*W+x] + I[p-1] + I[p-(W+1)] - I[p-W-2];
      }
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const x1=Math.max(0,x-R), y1=Math.max(0,y-R);
        const x2=Math.min(W-1,x+R), y2=Math.min(H-1,y+R);
        const cnt = (x2-x1+1)*(y2-y1+1);
        const s = I[(y2+1)*(W+1)+(x2+1)] - I[y1*(W+1)+(x2+1)]
                - I[(y2+1)*(W+1)+x1]     + I[y1*(W+1)+x1];
        bin[y*W+x] = (gray[y*W+x] * cnt < s * T) ? 1 : 0;
      }
    }
    return bin;
  }

  /** מסיר רכיבים קטנים (רעש) */
  _removeNoise(bin, W, H) {
    const out = bin.slice();
    const visited = new Uint8Array(W * H);
    const stk = [];
    for (let i = 0; i < W * H; i++) {
      if (!bin[i] || visited[i]) continue;
      const pxs = [];
      stk.push(i); visited[i] = 1;
      while (stk.length) {
        const ci = stk.pop(); pxs.push(ci);
        const cx = ci % W, cy = (ci / W) | 0;
        for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
          const nx=cx+dx, ny=cy+dy;
          if (nx<0||nx>=W||ny<0||ny>=H) continue;
          const ni = ny*W+nx;
          if (bin[ni] && !visited[ni]) { visited[ni]=1; stk.push(ni); }
        }
      }
      if (pxs.length < 9) pxs.forEach(p => out[p] = 0);
    }
    return out;
  }

  /* ══════════════════ TEMPLATE BUILDING ══════════════════ */

  /**
   * מייצר תבניות ייחוס לכל ספרה ע"י רינדור Canvas עם
   * גופנים שונים, זוויות שונות וסקלות שונות.
   * כך נכסה מגוון רחב של כתבי יד ללא קובץ חיצוני.
   */
  async _buildTemplates() {
    const S = this._S;
    const tmpls = {};
    // צמצום דרסטי: 3×3×2×1 = 18 תבניות לספרה במקום 350
    // OCR הוא גיבוי ל-Gemini — מהירות > כיסוי
    const FONTS  = [
      `bold ${S}px Arial`,
      `bold ${S}px Verdana`,
      `${S}px "Courier New"`,
    ];
    const ANGLES = [-0.12, 0, 0.12];
    const SCALES = [0.90, 1.05];
    const SHEAR  = 0;

    const yield_ = () => new Promise(r => setTimeout(r, 0)); // שחרור ה-thread בין אצוות

    for (const ch of '0123456789') {
      tmpls[ch] = [];
      for (const font of FONTS)
        for (const angle of ANGLES)
          for (const scale of SCALES) {
            const t = this._renderChar(ch, font, angle, scale, SHEAR, S);
            if (t) tmpls[ch].push(t);
          }
      await yield_(); // מניח ל-UI לנשום בין ספרה לספרה
    }
    for (const ch of '/.-') {
      tmpls[ch] = [];
      for (const font of FONTS) {
        const t = this._renderChar(ch, font, 0, 1.0, 0, S);
        if (t) tmpls[ch].push(t);
      }
    }
    return tmpls;
  }

  _renderChar(ch, font, angle, scale, shear, S) {
    const c = document.createElement('canvas');
    c.width = c.height = S;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, S, S);
    ctx.save();
    ctx.translate(S/2, S/2);
    ctx.transform(scale, 0, shear * scale, scale, 0, 0);
    ctx.rotate(angle);
    ctx.translate(-S/2, -S/2);
    ctx.fillStyle = '#000';
    ctx.font = font;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(ch, S/2, S/2);
    ctx.restore();
    const d = ctx.getImageData(0, 0, S, S).data;
    const t = new Uint8Array(S * S);
    let ink = false;
    for (let i = 0; i < S*S; i++) { t[i] = d[i*4] < 140 ? 1 : 0; if (t[i]) ink=true; }
    return ink ? t : null;
  }

  /* ══════════════════ FIELD EXTRACTION ══════════════════ */

  _findDate(bin, W, H) {
    // אזורים לפי פריסת טופס 33.1106
    const regions = [
      { x1:0.02, y1:0.03, x2:0.40, y2:0.20 },  // עמוד 1 — מלבן שמאל-עליון
      { x1:0.10, y1:0.50, x2:0.72, y2:0.88 },  // עמוד 2 — הצהרת נהג / חלק ג'
      { x1:0.00, y1:0.00, x2:1.00, y2:1.00 },  // כל הדף (fallback)
    ];
    for (const r of regions) {
      const sub = this._crop(bin, W, H, r);
      const res = this._searchField(sub, 'date');
      if (res) return res;
    }
    return null;
  }

  _findPlate(bin, W, H) {
    const regions = [
      { x1:0.02, y1:0.10, x2:0.40, y2:0.30 },
      { x1:0.10, y1:0.63, x2:0.75, y2:0.92 },
      { x1:0.00, y1:0.00, x2:1.00, y2:1.00 },
    ];
    for (const r of regions) {
      const sub = this._crop(bin, W, H, r);
      const res = this._searchField(sub, 'plate');
      if (res) return res;
    }
    return null;
  }

  _crop(bin, W, H, { x1, y1, x2, y2 }) {
    const rx1=Math.floor(x1*W), ry1=Math.floor(y1*H);
    const rx2=Math.floor(x2*W), ry2=Math.floor(y2*H);
    const sw=rx2-rx1, sh=ry2-ry1;
    const sub = new Uint8Array(sw * sh);
    for (let y=0; y<sh; y++)
      for (let x=0; x<sw; x++)
        sub[y*sw+x] = bin[(y+ry1)*W+(x+rx1)];
    sub._w = sw; sub._h = sh;
    return sub;
  }

  /* ══════════════════ LINE & CHAR SEGMENTATION ══════════════════ */

  _searchField(sub, type) {
    const W = sub._w, H = sub._h;
    const lines = this._findLines(sub, W, H);
    for (const { y1, y2 } of lines) {
      const chars = this._segLine(sub, W, y1, y2);
      const text  = this._recognizeLine(chars, sub, W);
      const val   = type === 'date' ? this._parseDate(text) : this._parsePlate(text);
      if (val) return val;
    }
    return null;
  }

  _findLines(bin, W, H) {
    const proj = new Array(H).fill(0);
    for (let y=0; y<H; y++)
      for (let x=0; x<W; x++)
        if (bin[y*W+x]) proj[y]++;

    const thresh = Math.max(4, W * 0.035);
    const lines = []; let start=-1, last=-1;
    for (let y=0; y<H; y++) {
      if (proj[y] >= thresh) {
        if (start === -1) start = y;
        last = y;
      } else if (start !== -1 && y - last > 4) {
        const h = last - start + 1;
        if (h >= 8 && h < H * 0.20) lines.push({ y1:start, y2:last });
        start = -1;
      }
    }
    if (start !== -1 && last-start+1 >= 8) lines.push({ y1:start, y2:last });
    return lines;
  }

  _segLine(bin, W, y1, y2) {
    // Vertical projection בתוך שורה
    const vp = new Array(W).fill(0);
    for (let y=y1; y<=y2; y++)
      for (let x=0; x<W; x++)
        if (bin[y*W+x]) vp[x]++;

    const chars = []; let start=-1;
    for (let x=0; x<W; x++) {
      if (vp[x]>0 && start===-1) start=x;
      else if (vp[x]===0 && start!==-1) { chars.push({x1:start,x2:x-1,y1,y2}); start=-1; }
    }
    if (start!==-1) chars.push({x1:start,x2:W-1,y1,y2});
    return this._mergeFragments(chars, y2-y1+1);
  }

  _mergeFragments(chars, lineH) {
    if (!chars.length) return [];
    const GAP = Math.max(2, lineH * 0.18);
    const out = [{ ...chars[0] }];
    for (let i=1; i<chars.length; i++) {
      const prev=out[out.length-1], curr=chars[i];
      const gap = curr.x1 - prev.x2;
      const tinyW = (curr.x2-curr.x1+1) < lineH * 0.35;
      if (gap <= GAP && tinyW) { prev.x2 = curr.x2; }  // מיזוג נקודות/רעש
      else out.push({ ...curr });
    }
    return out.filter(c => {
      const cw=c.x2-c.x1+1, ch=c.y2-c.y1+1;
      return cw>=3 && cw < ch*4;
    });
  }

  /* ══════════════════ CHARACTER RECOGNITION ══════════════════ */

  _recognizeLine(chars, bin, W) {
    return chars.map(c => this._classifyChar(c, bin, W)).join('');
  }

  _classifyChar(comp, bin, W) {
    const { x1, x2, y1, y2 } = comp;
    const cw=x2-x1+1, ch=y2-y1+1;
    const aspect = cw / ch;

    // חילוץ patch
    const patch = new Uint8Array(cw * ch);
    for (let y=y1; y<=y2; y++)
      for (let x=x1; x<=x2; x++)
        patch[(y-y1)*cw+(x-x1)] = bin[y*W+x];

    // אם יש רשת נוירונים — השתמש בה
    if (this._nn) return this._nnPredict(patch, cw, ch);

    const S = this._S;
    const norm = this._resize(patch, cw, ch, S);

    // צר מאוד → הפרדה (/, ., -, 1)
    if (aspect < 0.30) return this._best(norm, ['/','.','-','1']);

    // רחב מאוד → שתי ספרות שדבוקות — פצל לשניים
    if (aspect > 2.0) {
      const mid = Math.floor((x1+x2)/2);
      const L = this._classifyChar({x1, x2:mid,   y1, y2}, bin, W);
      const R = this._classifyChar({x1:mid+1, x2, y1, y2}, bin, W);
      return L + R;
    }

    return this._best(norm, '0123456789'.split(''));
  }

  _resize(patch, srcW, srcH, dstS) {
    const dst = new Uint8Array(dstS * dstS);
    for (let dy=0; dy<dstS; dy++)
      for (let dx=0; dx<dstS; dx++) {
        const sx = Math.floor(dx * srcW / dstS);
        const sy = Math.floor(dy * srcH / dstS);
        dst[dy*dstS+dx] = patch[sy*srcW+sx];
      }
    return dst;
  }

  /** מחזיר את התו הקרוב ביותר לפי מרחק Hamming */
  _best(norm, candidates) {
    let bestCh='?', bestD=Infinity;
    for (const ch of candidates) {
      const ts = this._tmpls[ch];
      if (!ts) continue;
      for (const t of ts) {
        let d=0;
        for (let i=0; i<norm.length; i++) d += (norm[i] !== t[i]) ? 1 : 0;
        if (d < bestD) { bestD=d; bestCh=ch; }
      }
    }
    return bestCh;
  }

  /* ══════════════════ NEURAL NETWORK INFERENCE ══════════════════ */
  /*
   * שכבות נתמכות: dense (relu / softmax)
   * מבנה הרשת מוגדר ב-training/train_model.py ומיוצא כ-JSON.
   * פורמט: { layers: [ {W:[[...]], b:[...], activation:'relu'}, ... ] }
   */
  _nnPredict(patch, srcW, srcH) {
    const S = this._S;
    const flat = this._resize(patch, srcW, srcH, S);
    let x = Array.from(flat);  // [0|1, ...] בגודל S*S
    for (const { W, b, activation } of this._nn.layers) {
      const out = new Array(b.length).fill(0);
      for (let j=0; j<b.length; j++) {
        out[j] = b[j];
        for (let i=0; i<x.length; i++) out[j] += x[i] * W[i][j];
        if (activation === 'relu') out[j] = Math.max(0, out[j]);
      }
      if (activation === 'softmax') {
        const mx = Math.max(...out);
        const ex = out.map(v => Math.exp(v - mx));
        const sm = ex.reduce((a,b)=>a+b, 0);
        x = ex.map(v => v/sm);
      } else x = out;
    }
    return '0123456789'[x.indexOf(Math.max(...x))] || '?';
  }

  /* ══════════════════ PARSING ══════════════════ */

  _parseDate(raw) {
    // נקה — רק ספרות ומפרידים, תקן OCR שגויים (o→0, l→1)
    const t = raw.replace(/[oO]/g,'0').replace(/[lI]/g,'1')
                 .replace(/[^0-9\/\.\-]/g,'');
    const m = t.match(/(\d{1,2})[\/\.\-](\d{1,2})(?:[\/\.\-](\d{2,4}))?/);
    if (!m) return null;
    const di=+m[1], mi=+m[2];
    if (di<1||di>31||mi<1||mi>12) return null;
    const ds=String(di).padStart(2,'0'), ms=String(mi).padStart(2,'0');
    if (!m[3]) return `${ds}/${ms}`;
    const y = +m[3];
    const ys = m[3].length===2 ? (y>50?'19':'20')+m[3] : m[3];
    if (+ys<1990||+ys>2040) return null;
    return `${ds}/${ms}/${ys}`;
  }

  _parsePlate(raw) {
    const t = raw.replace(/[oO]/g,'0').replace(/[^0-9\-]/g,'');
    // אזרחי עם מקפים
    const civil = t.match(/(\d{2,3})-(\d{2,3})-(\d{2,3})/);
    if (civil) return `${civil[1]}-${civil[2]}-${civil[3]}`;
    // צבאי/ללא מקפים — 7 או 8 ספרות
    const mil = t.match(/\d{7,8}/);
    if (mil) return mil[0];
    // ספרות בלבד מהטקסט הגולמי
    const dig = raw.replace(/\D/g,'');
    if (dig.length===7) return `${dig.slice(0,2)}-${dig.slice(2,5)}-${dig.slice(5)}`;
    if (dig.length===8) return dig;
    return null;
  }
}

window.OCREngine = OCREngine;
