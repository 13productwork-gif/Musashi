# musashi_core.py
# Goodnotes to Anki Logic Core (Headless for Colab)
# v1.2: Expanded fields & Group override max_clozes

import sys, subprocess, importlib, io, os, logging, tempfile, uuid, base64
from typing import List, Tuple, Dict
from pathlib import Path

# 必要なライブラリがなければインストール
def install_deps():
    required = ["pypdf", "pypdfium2", "genanki", "Pillow"]
    for pkg in required:
        try:
            importlib.import_module(pkg if pkg != "Pillow" else "PIL")
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install_deps()

import pypdf
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
import genanki

# --- Constants & Helpers ---
def hz(hx:str)->str: return (hx or "").lstrip("#")
def hex_to_rgb_frac(hx:str)->Tuple[float,float,float]:
    c = hz(hx); return (int(c[0:2],16)/255, int(c[2:4],16)/255, int(c[4:6],16)/255)
def almost_equal_rgb(a:Tuple[int,int,int], b:Tuple[int,int,int], tol:int=3)->bool:
    return abs(a[0]-b[0])<=tol and abs(a[1]-b[1])<=tol and abs(a[2]-b[2])<=tol

class SimpleRect:
    __slots__=("x0","y0","x1","y1")
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = float(x0), float(y0), float(x1), float(y1)
    def intersects(self, o:"SimpleRect")->bool:
        return not (self.x1 <= o.x0 or self.x0 >= o.x1 or self.y1 <= o.y0 or self.y0 >= o.y1)
    def include_rect(self, o:"SimpleRect"):
        self.x0 = min(self.x0, o.x0); self.y0 = min(self.y0, o.y0)
        self.x1 = max(self.x1, o.x1); self.y1 = max(self.y1, o.y1)

def merge_rects_simple(rects: List[SimpleRect]) -> List[SimpleRect]:
    m = list(rects)
    while True:
        changed, out = False, []
        used = [False]*len(m)
        for i, r in enumerate(m):
            if used[i]: continue
            acc = SimpleRect(r.x0, r.y0, r.x1, r.y1)
            for j in range(i+1, len(m)):
                if not used[j] and acc.intersects(m[j]):
                    acc.include_rect(m[j]); used[j]=True; changed=True
            out.append(acc)
        m = out
        if not changed: break
    return m

def resolve(obj):
    try:
        if hasattr(obj, "get_object"): return obj.get_object()
    except Exception: pass
    return obj

def rgb01_to_255(seq):
    if not seq: return (0,0,0)
    out=[]
    for c in seq[:3]:
        try: f = float(c)
        except Exception: f = 0.0
        if f>1.001: v = int(round(min(max(f, 0.0), 255.0)))
        else:       v = int(round(min(max(f, 0.0), 1.0)*255.0))
        out.append(v)
    while len(out)<3: out.append(0)
    return tuple(out[:3])

def looks_like_highlight(rect: SimpleRect, page_h: float, rotation: int, aspect_min: float = 0.10, max_height_ratio: float = 0.20) -> bool:
    w = max(1e-6, rect.x1-rect.x0); h = max(1e-6, rect.y1-rect.y0)
    if rotation in (0, 180):
        return (w/h >= aspect_min) or (h <= page_h*max_height_ratio)
    else:
        return (h/w >= aspect_min) or (w <= page_h*max_height_ratio)

def rects_from_ink(annot, default_padding=2.0) -> List[SimpleRect]:
    rects=[]
    inklist = resolve(annot.get("/InkList")) or []
    w = None
    bs = resolve(annot.get("/BS"))
    if isinstance(bs, dict) and bs.get("/W") is not None:
        try: w = float(bs.get("/W"))
        except Exception: pass
    if w is None:
        border = resolve(annot.get("/Border"))
        if isinstance(border, list) and len(border)>=3:
            try: w = float(border[2])
            except Exception: pass
    pad = (w or 0)/2.0 if w else default_padding
    for path in inklist:
        try:
            xs = [float(path[i]) for i in range(0, len(path), 2)]
            ys = [float(path[i+1]) for i in range(0, len(path), 2)]
        except Exception: xs, ys = [], []
        if not xs: continue
        rects.append(SimpleRect(min(xs)-pad, min(ys)-pad, max(xs)+pad, max(ys)+pad))
    return rects

def pdf_to_pil_box(r: SimpleRect, page_h: float, scale: float) -> Tuple[int,int,int,int]:
    return (int(round(r.x0*scale)), int(round((page_h - r.y1)*scale)), int(round(r.x1*scale)), int(round((page_h - r.y0)*scale)))

def rect_to_local_box(r: SimpleRect, seg: SimpleRect, page_h: float, scale: float, w: int, h: int) -> Tuple[int,int,int,int]:
    abs_r  = pdf_to_pil_box(r,   page_h, scale)
    abs_se = pdf_to_pil_box(seg, page_h, scale)
    box = (abs_r[0]-abs_se[0], abs_r[1]-abs_se[1], abs_r[2]-abs_se[0], abs_r[3]-abs_se[1])
    return (max(0, min(w, box[0])), max(0, min(h, box[1])), max(0, min(w, box[2])), max(0, min(h, box[3])))

def rect_center(r: SimpleRect) -> Tuple[float,float]:
    return ((r.x0 + r.x1)/2.0, (r.y0 + r.y1)/2.0)

def inflate_rect(r: SimpleRect, m: float) -> SimpleRect:
    return SimpleRect(r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m)

def point_in_rect(x: float, y: float, r: SimpleRect) -> bool:
    return (r.x0 <= x <= r.x1) and (r.y0 <= y <= r.y1)

def reading_sort_key(r: SimpleRect, page_h: float, rotation: int, row_pct: float):
    cx, cy = rect_center(r)
    bucket = max(1.0, page_h * max(0.001, row_pct))
    if rotation in (0, 180):
        row = int(round((page_h - cy) / bucket))
        return (row, cx)
    else:
        col = int(round(cx / bucket))
        return (col, -(cy))

# --- Main Analysis Logic ---
def analyze_pages(reader, s_page, l_page, color_hex_map, cut_hl_color, group_ink_color, max_clozes, color_tol):
    HL_SORT_ROW_PCT = 0.035
    prio_map = {}
    for hx, pr in color_hex_map.items():
        r,g,b = [int(v*255) for v in hex_to_rgb_frac(hx)]
        prio_map[(r,g,b)] = pr
    cut_hl_rgb = tuple(int(v*255) for v in hex_to_rgb_frac(cut_hl_color))
    group_ink_rgb = tuple(int(v*255) for v in hex_to_rgb_frac(group_ink_color))
    
    res = {}
    total_pages = len(reader.pages)
    s_page = max(1, s_page); l_page = min(l_page, total_pages)

    for pi in range(s_page, l_page+1):
        page = reader.pages[pi-1]
        ph = float(page.mediabox.height)
        pl = float(page.mediabox.left); pr = float(page.mediabox.right)
        rot = int(resolve(page.get("/Rotate")) or 0) % 360

        annots = resolve(page.get("/Annots"))
        cut_rects_hl = []
        seg_annots = []
        group_rects_ink = []
        
        if isinstance(annots, list) and annots:
            for ref in annots:
                a = resolve(ref)
                subtype = a.get("/Subtype")
                col = rgb01_to_255(resolve(a.get("/C")))

                # Highlight
                if subtype == "/Highlight":
                    qp = resolve(a.get("/QuadPoints")) or []
                    rects=[]
                    if qp:
                        for q in range(0, len(qp), 8):
                            xs = [float(qp[j]) for j in (0,2,4,6)]; ys_ = [float(qp[j]) for j in (1,3,5,7)]
                            rects.append(SimpleRect(min(xs), min(ys_), max(xs), max(ys_)))
                    else:
                        rct = resolve(a.get("/Rect"))
                        if rct and len(rct)>=4:
                            rects = [SimpleRect(float(rct[0]), float(rct[1]), float(rct[2]), float(rct[3]))]
                    if not rects: continue
                    if almost_equal_rgb(col, cut_hl_rgb, tol=color_tol):
                        cut_rects_hl.extend(merge_rects_simple(rects)); continue
                    matched = None
                    for rgb, prname in prio_map.items():
                        if almost_equal_rgb(col, rgb, tol=color_tol): matched = prname; break
                    if matched:
                        for mr in merge_rects_simple(rects):
                            if looks_like_highlight(mr, ph, rot): seg_annots.append((mr, matched))
                        continue

                # Ink
                if subtype == "/Ink":
                    rects = rects_from_ink(a)
                    if not rects: continue
                    if almost_equal_rgb(col, group_ink_rgb, tol=color_tol):
                        for mr in merge_rects_simple(rects): group_rects_ink.append(inflate_rect(mr, 3.0))
                        continue
                    if almost_equal_rgb(col, cut_hl_rgb, tol=color_tol):
                        for mr in merge_rects_simple(rects):
                            if looks_like_highlight(mr, ph, rot, aspect_min=1.2): cut_rects_hl.append(mr)
                        continue
                    matched = None
                    for rgb, prname in prio_map.items():
                        if almost_equal_rgb(col, rgb, tol=color_tol): matched = prname; break
                    if matched:
                        valid_rects = [r for r in rects if looks_like_highlight(r, ph, rot)]
                        if valid_rects:
                            for mr in merge_rects_simple(valid_rects): seg_annots.append((mr, matched))
                        continue

        ys = [ph, 0]
        if cut_rects_hl:
            cut_ys = [ (r.y1) for r in cut_rects_hl ]
            cut_ys = [min(max(y, 0.0), ph) for y in cut_ys]
            cut_ys.sort(reverse=True)
            merged_ys = []
            for y in cut_ys:
                if not merged_ys or abs(y - merged_ys[-1]) >= 6.0: merged_ys.append(y)
            ys = [ph] + merged_ys + [0]
        
        info = {"segments": [], "page_h": ph}
        for top, bot in zip(ys[:-1], ys[1:]):
            seg = SimpleRect(pl, bot, pr, top)
            in_seg = [x for x in seg_annots if x[0].intersects(seg)]
            if not in_seg: continue

            applied_groups = [g for g in group_rects_ink if g.intersects(seg)]
            if applied_groups: applied_groups.sort(key=lambda r: reading_sort_key(r, ph, rot, HL_SORT_ROW_PCT))
            group_bins = [[] for _ in applied_groups]; ungrouped = []

            for (r,p) in in_seg:
                cx, cy = rect_center(r); assigned = False
                for gi, g in enumerate(applied_groups):
                    if point_in_rect(cx, cy, g): group_bins[gi].append((r,p)); assigned = True; break
                if not assigned: ungrouped.append((r,p))

            # ★改良: force_single_card フラグを追加
            def make_chunks(items, force_single_card=False):
                if not items: return [], []
                grouped_local = {}
                for r,p in items: grouped_local.setdefault(p, []).append(r)
                for p in grouped_local:
                    grouped_local[p] = merge_rects_simple(grouped_local[p])
                    grouped_local[p].sort(key=lambda r: reading_sort_key(r, ph, rot, HL_SORT_ROW_PCT))
                full_local = [r for lst in grouped_local.values() for r in lst]
                
                chunks_with_key = []
                for p, lst in grouped_local.items():
                    # グループ化されている場合(force_single_card=True)は、分割せずに丸ごと1チャンクにする
                    step = len(lst) if force_single_card else max_clozes
                    step = max(1, step)
                    
                    for i in range(0, len(lst), step):
                        sub = lst[i:i+step]
                        if not sub: continue
                        key = reading_sort_key(sub[0], ph, rot, HL_SORT_ROW_PCT)
                        chunks_with_key.append((p, sub, key))
                        
                chunks_with_key.sort(key=lambda x: x[2])
                return full_local, [(p, sub) for (p,sub,k) in chunks_with_key]

            # 1. Groups (Gold) -> 制限無視 (force_single_card=True)
            for bin_items in group_bins:
                full_l, chunks_l = make_chunks(bin_items, force_single_card=True)
                if chunks_l: info["segments"].append((seg, full_l, chunks_l))
            
            # 2. Ungrouped -> 制限あり (force_single_card=False)
            full_u, chunks_u = make_chunks(ungrouped, force_single_card=False)
            if chunks_u: info["segments"].append((seg, full_u, chunks_u))

        if info["segments"]: res[pi] = info
    return res

def render_page_pil(pdf, idx0, dpi):
    page = pdf[idx0]
    try: return page.render(scale=dpi/72).to_pil()
    finally: page.close()

def make_overlaid_images(base_rgba, seg, full_rects, chunk_rects, page_h, scale):
    w, h = base_rgba.size
    ov_front = Image.new("RGBA", (w,h)); ov_back = Image.new("RGBA", (w,h))
    d_f = ImageDraw.Draw(ov_front); d_b = ImageDraw.Draw(ov_back)
    fill_ans = (255, 141, 142, 255)
    for r in chunk_rects:
        d_f.rectangle(rect_to_local_box(r, seg, page_h, scale, w, h), fill=fill_ans)
    ow = max(1, int(round(0.8*scale)))
    for drw in (d_f, d_b):
        for r in full_rects: drw.rectangle(rect_to_local_box(r, seg, page_h, scale, w, h), outline=(0,0,0,255), width=ow)
    return Image.alpha_composite(base_rgba, ov_front), Image.alpha_composite(base_rgba, ov_back)

def process(pdf_path, color_map, cut_col, group_col, zoom, qual, max_clozes, tol):
    try:
        reader = pypdf.PdfReader(pdf_path)
        pdf = pdfium.PdfDocument(pdf_path)
    except Exception as e: return None, str(e)
    
    analysis = analyze_pages(reader, 1, len(reader.pages), color_map, cut_col, group_col, max_clozes, tol)
    if not analysis: return None, "有効なハイライトが見つかりませんでした。"
    
    deck_name = Path(pdf_path).stem
    deck = genanki.Deck(2059408600, deck_name)
    
    # ★改良: モデルフィールドの拡張
    model_fields = [
        {'name':'ヘッダー'}, 
        {'name':'重要度'}, 
        {'name':'ページ番号'},
        {'name':'表面（質問）'}, 
        {'name':'裏面（答え）'},
        {'name':'追加画像（表面）'}, 
        {'name':'追加画像（裏面）'}, 
        {'name':'音声（表面）'}, 
        {'name':'音声（裏面）'}, 
        {'name':'コメント'}
    ]
    
    model = genanki.Model(
        1607398600, 
        'ImgCloze_Musashi_v2', 
        fields=model_fields,
        templates=[{
            'name':'Card', 
            'qfmt': '''
                <div style="text-align:center">
                    <strong>{{ヘッダー}}</strong><br>
                    {{表面（質問）}}<br>
                    {{追加画像（表面）}}<br>
                    {{音声（表面）}}
                </div>
            ''', 
            'afmt': '''
                <div style="text-align:center">
                    <strong>{{ヘッダー}}</strong><br>
                    {{裏面（答え）}}<br>
                    {{追加画像（裏面）}}<br>
                    {{音声（裏面）}}<br>
                    <hr>
                    <div style="color:#888;">p.{{ページ番号}} / {{重要度}}</div>
                    <div style="color:#2a2;">{{コメント}}</div>
                </div>
            '''
        }],
        css='img { max-width: 100%; height: auto; }'
    )
    
    dpi = int(zoom*72)
    media_files = []
    global_card_seq = 0
    
    print(f"Converting {len(analysis)} pages...")
    with tempfile.TemporaryDirectory() as temp_media_dir:
        for pi, info in sorted(analysis.items()):
            base = render_page_pil(pdf, pi-1, dpi)
            scale = base.height / info["page_h"]
            for seg, full, chunks_info in info["segments"]:
                crop = base.crop(pdf_to_pil_box(seg, info["page_h"], scale))
                base_rgba = crop.convert("RGBA")
                
                for prio, chunk in chunks_info:
                    front, back = make_overlaid_images(base_rgba, seg, full, chunk, info["page_h"], scale)
                    
                    global_card_seq += 1
                    safe_deck_name = "".join(c for c in deck_name if c.isalnum())
                    fname_base = f"{safe_deck_name}_p{pi:03d}_{global_card_seq:05d}"
                    fname_f = f"{fname_base}_Q.jpg"
                    fname_b = f"{fname_base}_A.jpg"
                    
                    pf = os.path.join(temp_media_dir, fname_f); pb = os.path.join(temp_media_dir, fname_b)
                    front.convert("RGB").save(pf, quality=qual); back.convert("RGB").save(pb, quality=qual)
                    media_files.append(pf); media_files.append(pb)
                    
                    # Note作成（拡張フィールド対応）
                    note = genanki.Note(
                        model=model,
                        fields=[
                            deck_name,          # ヘッダー
                            prio,               # 重要度
                            str(pi),            # ページ番号
                            f'<img src="{fname_f}">', # 表面画像
                            f'<img src="{fname_b}">', # 裏面画像
                            '', '', '', '', ''  # 追加フィールドは空で初期化
                        ]
                    )
                    deck.add_note(note)
        
        pdf.close()
        pkg = genanki.Package(deck); pkg.media_files = media_files
        out_path = f"/content/{deck_name}.apkg"
        pkg.write_to_file(out_path)
        return out_path, None
