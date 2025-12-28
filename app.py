import streamlit as st
import sys, subprocess, importlib
import io, os, logging, tempfile, uuid
from typing import List, Tuple, Dict
from pathlib import Path
from PIL import Image, ImageDraw

# --- 必須ライブラリのチェックとインストール ---
# Streamlit Cloudではrequirements.txtを使いますが、念のため
try:
    import pypdf
    import pypdfium2 as pdfium
    import genanki
except ImportError:
    st.error("ライブラリが見つかりません。requirements.txtを確認してください。")
    st.stop()

# ===========================================================
# 0) セキュリティ設定 (パスワード保護)
# ===========================================================
def check_password():
    """Returns `True` if the user had the correct password."""
    # 実際の運用では st.secrets を使うのがベストですが、簡易的にコードに書くか
    # 環境変数から読み込む形にします。
    # ここでは仮に 'musashi123' としています。
    password = st.text_input("パスワードを入力してください", type="password")
    if password == "musashi123": # ← 毎月ここを変えればOK
        return True
    return False

# ===========================================================
# 1) UI & Config
# ===========================================================
st.set_page_config(page_title="Goodnotes to Anki Converter", layout="wide")

if not check_password():
    st.stop()

st.title("Goodnotes × Anki 自動変換ツール 'Musashi'")
st.markdown("""
PDFをアップロードすると、ハイライト箇所を自動で穴埋めカードに変換します。
- **緑**: 重要度High
- **ピンク**: 重要度Medium
- **黄色**: 重要度Low
- **水色**: ページ分割線 (#cut)
""")

# サイドバー設定
with st.sidebar:
    st.header("設定")
    zoom_factor = st.slider("画質 (Zoom)", 1, 4, 2)
    jpeg_quality = st.slider("JPEG圧縮率", 40, 95, 70)
    max_clozes = st.slider("1カードあたりの最大穴埋め数", 1, 8, 4)
    
    with st.expander("カラー設定 (HEX)"):
        color_high = st.color_picker("High (緑)", "#31FC85")
        color_mid = st.color_picker("Mid (ピンク)", "#FF58C4")
        color_low = st.color_picker("Low (黄)", "#FFFF10")
        cut_hl_color = st.color_picker("Cut Line (水色)", "#00FFFF")
        group_ink_color = st.color_picker("Group Ink (金)", "#FFD700")

    MODEL_ID = 1607398600
    DECK_ID = 2059408600

# ===========================================================
# 2) Helpers
# ===========================================================
def hz(hx:str)->str: return (hx or "").lstrip("#")
def hex_to_rgb_frac(hx:str)->Tuple[float,float,float]:
    c = hz(hx); return (int(c[0:2],16)/255, int(c[2:4],16)/255, int(c[4:6],16)/255)
def hex_to_rgba_tuple(hx:str, alpha:int=255)->Tuple[int,int,int,int]:
    c = hz(hx); return (int(c[0:2],16), int(c[2:4],16), int(c[4:6],16), alpha)
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

def pdf_to_pil_box(r: SimpleRect, page_h: float, scale: float) -> Tuple[int,int,int,int]:
    return (int(round(r.x0*scale)),
            int(round((page_h - r.y1)*scale)),
            int(round(r.x1*scale)),
            int(round((page_h - r.y0)*scale)))

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
    pad = (w or 0)/2.0 if w else default_padding
    for path in inklist:
        try:
            xs = [float(path[i]) for i in range(0, len(path), 2)]
            ys = [float(path[i+1]) for i in range(0, len(path), 2)]
        except Exception: xs, ys = [], []
        if not xs: continue
        rects.append(SimpleRect(min(xs)-pad, min(ys)-pad, max(xs)+pad, max(ys)+pad))
    return rects

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

def reading_sort_key(r: SimpleRect, page_h: float, rotation: int, row_pct: float=0.035):
    cx, cy = rect_center(r)
    bucket = max(1.0, page_h * max(0.001, row_pct))
    if rotation in (0, 180):
        row = int(round((page_h - cy) / bucket))
        return (row, cx)
    else:
        col = int(round(cx / bucket))
        return (col, -(cy))

# ===========================================================
# 3) Analysis Logic
# ===========================================================
def analyze_pages_for_cards_pypdf(
    reader: pypdf.PdfReader,
    s_page: int, l_page: int,
    color_hex_map: Dict[str,str],
    cut_hl_color: str,
    group_ink_color: str,
    max_clozes: int,
):
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

                if subtype == "/Highlight":
                    # --- QuadPoints or Rect ---
                    qp = resolve(a.get("/QuadPoints")) or []
                    rects=[]
                    if qp:
                        for q in range(0, len(qp), 8):
                            xs = [float(qp[j]) for j in (0,2,4,6)]
                            ys_ = [float(qp[j]) for j in (1,3,5,7)]
                            rects.append(SimpleRect(min(xs), min(ys_), max(xs), max(ys_)))
                    else:
                        rct = resolve(a.get("/Rect"))
                        if rct and len(rct)>=4:
                            rects = [SimpleRect(float(rct[0]), float(rct[1]), float(rct[2]), float(rct[3]))]
                    
                    if not rects: continue

                    # Cut Highlight
                    if almost_equal_rgb(col, cut_hl_rgb, tol=10):
                        cut_rects_hl.extend(merge_rects_simple(rects))
                        continue

                    # Priority Match
                    matched = None
                    for rgb, prname in prio_map.items():
                        if almost_equal_rgb(col, rgb, tol=6):
                            matched = prname; break
                    if matched:
                        for mr in merge_rects_simple(rects):
                            # 形状チェック
                            if looks_like_highlight(mr, ph, rot):
                                seg_annots.append((mr, matched))
                        continue

                if subtype == "/Ink":
                    # Ink Grouping
                    if almost_equal_rgb(col, group_ink_rgb, tol=10):
                        rects = rects_from_ink(a)
                        for mr in merge_rects_simple(rects):
                            group_rects_ink.append(inflate_rect(mr, 3.0))
                        continue

        # Page Splitting (Cuts)
        ys = [ph, 0]
        if cut_rects_hl:
            # 座標整理
            cut_ys = [ (r.y1) for r in cut_rects_hl ]
            cut_ys = [min(max(y, 0.0), ph) for y in cut_ys]
            cut_ys.sort(reverse=True)
            # マージ
            merged_ys = []
            for y in cut_ys:
                if not merged_ys or abs(y - merged_ys[-1]) >= 6.0:
                    merged_ys.append(y)
            ys = [ph] + merged_ys + [0]
        
        info = {"segments": [], "page_h": ph}
        
        for top, bot in zip(ys[:-1], ys[1:]):
            seg = SimpleRect(pl, bot, pr, top)
            in_seg = [x for x in seg_annots if x[0].intersects(seg)]
            if not in_seg: continue

            # Grouping within segment
            applied_groups = [g for g in group_rects_ink if g.intersects(seg)]
            if applied_groups:
                applied_groups.sort(key=lambda r: reading_sort_key(r, ph, rot))
            
            group_bins = [[] for _ in applied_groups]
            ungrouped = []

            for (r,p) in in_seg:
                cx, cy = rect_center(r)
                assigned = False
                for gi, g in enumerate(applied_groups):
                    if point_in_rect(cx, cy, g):
                        group_bins[gi].append((r,p))
                        assigned = True; break
                if not assigned:
                    ungrouped.append((r,p))

            def make_chunks(items):
                if not items: return [], []
                # マージ
                grouped_local = {}
                for r,p in items: grouped_local.setdefault(p, []).append(r)
                for p in grouped_local:
                    grouped_local[p] = merge_rects_simple(grouped_local[p])
                    grouped_local[p].sort(key=lambda r: reading_sort_key(r, ph, rot))
                
                full_local = [r for lst in grouped_local.values() for r in lst]
                
                # チャンク化
                chunks_with_key = []
                for p, lst in grouped_local.items():
                    for i in range(0, len(lst), max_clozes):
                        sub = lst[i:i+max_clozes]
                        if not sub: continue
                        key = reading_sort_key(sub[0], ph, rot)
                        chunks_with_key.append((p, sub, key))
                
                chunks_with_key.sort(key=lambda x: x[2]) # 読み順ソート
                return full_local, [(p, sub) for (p,sub,k) in chunks_with_key]

            # 1. Groups
            for bin_items in group_bins:
                full_l, chunks_l = make_chunks(bin_items)
                if chunks_l: info["segments"].append((seg, full_l, chunks_l))
            
            # 2. Ungrouped
            full_u, chunks_u = make_chunks(ungrouped)
            if chunks_u: info["segments"].append((seg, full_u, chunks_u))
            
        if info["segments"]:
            res[pi] = info
    return res

def render_page_pil(pdf: pdfium.PdfDocument, idx0: int, dpi: int) -> Image.Image:
    page = pdf[idx0]
    try:
        return page.render(scale=dpi/72).to_pil()
    finally:
        page.close()

def make_overlaid_images(base_rgba: Image.Image, seg: SimpleRect, full_rects, chunk_rects,
                         page_h: float, scale: float):
    w, h = base_rgba.size
    ov_front = Image.new("RGBA", (w,h))
    ov_back  = Image.new("RGBA", (w,h))
    d_f = ImageDraw.Draw(ov_front)
    d_b = ImageDraw.Draw(ov_back)

    # All highlights (dimmed/visible)
    # ここでは code2 (枠線のみ) を想定して、塗りつぶしは無しにします
    fill_ans = (255, 141, 142, 255) # ピンク系

    for r in chunk_rects:
        box = rect_to_local_box(r, seg, page_h, scale, w, h)
        d_f.rectangle(box, fill=fill_ans) # 表面は隠す

    ow = max(1, int(round(0.8*scale)))
    for drw in (d_f, d_b):
        for r in full_rects:
            box = rect_to_local_box(r, seg, page_h, scale, w, h)
            drw.rectangle(box, outline=(0,0,0,255), width=ow)

    front = Image.alpha_composite(base_rgba, ov_front)
    back  = Image.alpha_composite(base_rgba, ov_back)
    return front, back

# ===========================================================
# 4) Main Processing (Base64排除版)
# ===========================================================
def process_pdf(
    pdf_file,
    color_hex_map, cut_hl_color, group_ink_color,
    zoom_factor, jpeg_quality, max_clozes
):
    # アップロードされたファイルを一時ファイルとして保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_file.getvalue())
        tmp_pdf_path = tmp_pdf.name
    
    try:
        reader = pypdf.PdfReader(tmp_pdf_path)
        pdf = pdfium.PdfDocument(tmp_pdf_path)
    except Exception:
        return None, "PDFを開けませんでした"

    # 解析実行
    analysis = analyze_pages_for_cards_pypdf(
        reader, 1, len(reader.pages),
        color_hex_map, cut_hl_color, group_ink_color, max_clozes
    )

    if not analysis:
        return None, "有効なハイライトが見つかりませんでした"

    # デッキ作成準備
    deck_name = Path(pdf_file.name).stem
    deck = genanki.Deck(DECK_ID, deck_name)
    
    # モデル定義 (imgタグ形式)
    model = genanki.Model(
        MODEL_ID, 'ImgCloze_Musashi',
        fields=[{'name':'QImage'}, {'name':'AImage'}, {'name':'Page'}, {'name':'Priority'}],
        templates=[{
            'name':'Card',
            'qfmt': '{{QImage}}',
            'afmt': '{{AImage}}<br><div style="text-align:center;color:#888;">p.{{Page}} / {{Priority}}</div>'
        }],
        css='img { max-width: 100%; height: auto; }'
    )

    media_files = [] # 画像ファイルパス格納用
    dpi = int(zoom_factor*72)
    
    # プログレスバー
    progress_bar = st.progress(0)
    total_segments = sum(len(v["segments"]) for v in analysis.values())
    processed_count = 0

    # 一時ディレクトリに画像生成
    with tempfile.TemporaryDirectory() as temp_media_dir:
        for pi, info in analysis.items():
            base = render_page_pil(pdf, pi-1, dpi)
            scale = base.height / info["page_h"]
            
            for seg_idx, (seg, full, chunks_info) in enumerate(info["segments"]):
                crop = base.crop(pdf_to_pil_box(seg, info["page_h"], scale))
                base_rgba = crop.convert("RGBA")
                
                for chunk_idx, (prio, chunk) in enumerate(chunks_info):
                    front, back = make_overlaid_images(base_rgba, seg, full, chunk, info["page_h"], scale)
                    
                    # 画像ファイルとして保存 (ここが軽量化のキモ)
                    unique_id = uuid.uuid4().hex[:8]
                    fname_f = f"{unique_id}_Q.jpg"
                    fname_b = f"{unique_id}_A.jpg"
                    path_f = os.path.join(temp_media_dir, fname_f)
                    path_b = os.path.join(temp_media_dir, fname_b)
                    
                    front.convert("RGB").save(path_f, quality=jpeg_quality)
                    back.convert("RGB").save(path_b, quality=jpeg_quality)
                    
                    media_files.append(path_f)
                    media_files.append(path_b)

                    # Note追加 (HTMLにはファイル名だけ記述)
                    note = genanki.Note(
                        model=model,
                        fields=[
                            f'<img src="{fname_f}">',
                            f'<img src="{fname_b}">',
                            str(pi),
                            prio
                        ]
                    )
                    deck.add_note(note)
                
                processed_count += 1
                progress_bar.progress(min(processed_count / total_segments, 1.0))
        
        pdf.close()
        
        # Package作成
        package = genanki.Package(deck)
        package.media_files = media_files # ★これで画像が同梱される
        
        out_apkg_path = os.path.join(tempfile.gettempdir(), f"{deck_name}.apkg")
        package.write_to_file(out_apkg_path)
        
        # Streamlitはファイルをバイナリで読み込んで渡す必要がある
        with open(out_apkg_path, "rb") as f:
            apkg_bytes = f.read()
            
    return apkg_bytes, None

# ===========================================================
# 5) Main Run
# ===========================================================
uploaded_file = st.file_uploader("PDFファイルをアップロード", type=["pdf"])

if uploaded_file:
    if st.button("変換開始"):
        with st.spinner("解析中..."):
            cmap = {color_high:"High", color_mid:"Mid", color_low:"Low"}
            
            apkg_data, err = process_pdf(
                uploaded_file, cmap, cut_hl_color, group_ink_color,
                zoom_factor, jpeg_quality, max_clozes
            )
            
            if err:
                st.error(err)
            else:
                st.success("変換完了！")
                st.download_button(
                    label="Ankiデッキ(.apkg)をダウンロード",
                    data=apkg_data,
                    file_name=f"{Path(uploaded_file.name).stem}_musashi.apkg",
                    mime="application/octet-stream"
                )