#!/usr/bin/env python3
"""Local row-by-row viewer for the preformatted DSV4 SpecForge datasets.

Serves a single-page browser on localhost: pick a dataset, step through rows
(arrow keys / jump / random), and see each row exactly as SpecForge trains on
it — special tokens as chips, supervised (loss=1) assistant spans highlighted,
masked prompt tokens muted, with per-row token counts computed by the real
DSV4-Flash tokenizer (offset-mapping method, verified to match SpecForge's
ThinkingParser exactly).

    .venv/bin/python scripts/view_dataset.py --port 8321
"""
import argparse
import bisect
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

TOK_DIR = os.environ["SF_TOKENIZER"]  # any dir with the target's tokenizer
DATA_DIR = os.environ.get("SF_DATA_DIR", "cache/dataset")
CANDIDATES = [
    ("32K train", "dsv4_flash_0813_32k_train.jsonl"),
    ("32K test", "dsv4_flash_0813_32k_test.jsonl"),
    ("full train (untrimmed)", "dsv4_flash_0813_train.jsonl"),
    ("full test (untrimmed)", "dsv4_flash_0813_test.jsonl"),
]

A_TOKEN = "<｜Assistant｜>"
EOS = "<｜end▁of▁sentence｜>"
SPAN_PAT = re.compile(re.escape(A_TOKEN) + r"([\s\S]*?(?:" + re.escape(EOS) + "|$))")

_tok = None
_tok_lock = threading.Lock()


def tok():
    global _tok
    with _tok_lock:
        if _tok is None:
            from transformers import AutoTokenizer
            print("loading tokenizer ...", flush=True)
            _tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)
        return _tok


class LineIndex:
    """Byte offsets of each line; cached beside the file."""

    def __init__(self, path):
        self.path = path
        self.offsets = []
        self._load()

    def _cache_path(self):
        return self.path + ".offsets"

    def _load(self):
        cp = self._cache_path()
        if os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(self.path):
            import array
            a = array.array("q")
            with open(cp, "rb") as f:
                a.frombytes(f.read())
            self.offsets = a.tolist()
            return
        print(f"indexing {os.path.basename(self.path)} ...", flush=True)
        offs, pos = [], 0
        with open(self.path, "rb") as f:
            for line in f:
                offs.append(pos)
                pos += len(line)
        self.offsets = offs
        import array
        with open(cp, "wb") as f:
            f.write(array.array("q", offs).tobytes())
        print(f"  {len(offs)} rows", flush=True)

    def row(self, i):
        with open(self.path, "rb") as f:
            f.seek(self.offsets[i])
            return f.readline().decode("utf-8", errors="replace")


_indexes = {}
_idx_lock = threading.Lock()


def index_for(path, build=True):
    with _idx_lock:
        got = _indexes.get(path)
    if got is not None or not build:
        return got
    idx = LineIndex(path)
    with _idx_lock:
        _indexes.setdefault(path, idx)
        return _indexes[path]


def prewarm(paths):
    for p in paths:
        index_for(p)
    print("all indexes ready", flush=True)


def analyze(text):
    """-> (segments [{m, t, n}], total_tokens, loss_tokens) — parser-exact."""
    t = tok()
    enc = t(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    starts = [a for a, _ in offs]
    n = len(offs)

    def ntok(char_a, char_b):
        return bisect.bisect_left(starts, char_b) - bisect.bisect_left(starts, char_a)

    spans = [(m.start(1), m.end(1)) for m in SPAN_PAT.finditer(text)]
    segs, pos, loss = [], 0, 0
    for a, b in spans:
        if a > pos:
            segs.append({"m": 0, "t": text[pos:a], "n": ntok(pos, a)})
        k = ntok(a, b)
        segs.append({"m": 1, "t": text[a:b], "n": k})
        loss += k
        pos = b
    if pos < len(text):
        segs.append({"m": 0, "t": text[pos:], "n": ntok(pos, len(text))})
    return segs, n, loss


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>DSV4 Corpus Browser</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>
:root{color-scheme:light dark;
  --ground:#f6f7f5;--surface:#fff;--surface-2:#eef0ed;--ink:#1a1c1b;--ink-2:#565b58;
  --ink-3:#8b918d;--line:#dde1dd;--sup:#0e8f63;--sup-bg:#0e8f6322;--sup-ink:#0a6b4a;
  --mask-ink:#767c78;--accent:#2a78d6;--warn:#b3261e;
  --chip-bg:#1a1c1b0d;--chip-line:#1a1c1b26;}
@media (prefers-color-scheme: dark){:root{
  --ground:#141614;--surface:#1d201e;--surface-2:#252927;--ink:#e8ebe8;--ink-2:#a6aca8;
  --ink-3:#7c827e;--line:#333834;--sup:#2cbd87;--sup-bg:#2cbd8726;--sup-ink:#5ed7a8;
  --mask-ink:#878d89;--accent:#3987e5;--warn:#ef6e66;
  --chip-bg:#e8ebe812;--chip-line:#e8ebe830;}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.55 "Instrument Sans",system-ui,sans-serif;}
header.bar{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:10px;
  align-items:center;padding:12px 20px;background:var(--surface);
  border-bottom:1px solid var(--line)}
header.bar h1{font:700 17px "Sora",sans-serif;margin:0 12px 0 0}
select,input[type=number]{font:13px "JetBrains Mono",monospace;color:var(--ink);
  background:var(--surface-2);border:1px solid var(--line);border-radius:6px;padding:5px 8px}
input[type=number]{width:90px}
button{font:600 13px "Instrument Sans",sans-serif;color:var(--ink);cursor:pointer;
  background:var(--surface-2);border:1px solid var(--line);border-radius:6px;padding:5px 12px}
button:hover{border-color:var(--accent)}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.count{color:var(--ink-3);font:12px "JetBrains Mono",monospace}
main{max-width:1200px;margin:0 auto;padding:18px 20px 60px}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px}
.badge{font:11px "JetBrains Mono",monospace;padding:3px 9px;border-radius:999px;
  border:1px solid var(--chip-line);background:var(--chip-bg);color:var(--ink-2)}
.badge.sup{background:var(--sup-bg);border-color:var(--sup);color:var(--sup-ink)}
.maskmap{width:100%;height:14px;border-radius:4px;overflow:hidden;display:block;margin:4px 0 12px}
.stream{font:12.5px/1.7 "JetBrains Mono",monospace;background:var(--surface);
  border:1px solid var(--line);border-radius:10px;padding:16px 18px;
  white-space:pre-wrap;overflow-wrap:anywhere}
.seg-sup{background:var(--sup-bg);box-shadow:inset 0 -1.5px var(--sup);border-radius:2px}
.seg-mask{color:var(--mask-ink)}
.tokchip{display:inline-block;font-size:10.5px;font-weight:700;padding:0 5px;margin:0 1px;
  border-radius:4px;border:1px solid var(--chip-line);background:var(--surface-2);
  color:var(--ink-2);vertical-align:.06em;white-space:nowrap}
.seg-sup .tokchip{border-color:var(--sup);color:var(--sup-ink);background:transparent}
.expand{display:inline-block;font:italic 11px "Instrument Sans",sans-serif;cursor:pointer;
  padding:0 8px;margin:0 2px;border-radius:999px;border:1px dashed var(--ink-3);
  color:var(--ink-3);background:var(--ground);white-space:nowrap}
.expand:hover{border-color:var(--accent);color:var(--accent)}
.loading{color:var(--ink-3);padding:40px;text-align:center;font-family:"Sora",sans-serif}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--ink-2);margin:0 0 10px}
.legend .sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:5px}
kbd{font:11px "JetBrains Mono",monospace;border:1px solid var(--line);border-bottom-width:2px;
  border-radius:4px;padding:0 5px;background:var(--surface-2)}
</style></head><body>
<header class="bar">
  <h1>DSV4 Corpus Browser</h1>
  <select id="file"></select>
  <button id="prev">← prev</button>
  <input type="number" id="idx" min="0" value="0">
  <span class="count" id="total"></span>
  <button id="next">next →</button>
  <button id="rand">random</button>
  <span class="count">keys: <kbd>←</kbd><kbd>→</kbd> navigate · <kbd>r</kbd> random</span>
</header>
<main>
  <div class="legend">
    <span><span class="sw" style="background:var(--sup-bg);box-shadow:inset 0 -2px var(--sup)"></span><b>supervised</b> (loss=1) — assistant output</span>
    <span><span class="sw" style="background:var(--surface-2);border:1px solid var(--line)"></span><b>masked</b> (loss=0) — prompt: system, tools, user, tool results</span>
  </div>
  <div class="meta" id="meta"></div>
  <svg class="maskmap" id="map"></svg>
  <div class="stream" id="stream"><div class="loading">pick a row</div></div>
</main>
<script>
const $=s=>document.querySelector(s);
const fmt=n=>n.toLocaleString("en-US");
let FILES=[],cur={f:0,i:0},busy=false;
const SPECIALS=/(<｜begin▁of▁sentence｜>|<｜end▁of▁sentence｜>|<｜User｜>|<｜Assistant｜>|<think>|<\/think>|<｜DSML｜tool_calls>|<\/｜DSML｜tool_calls>|<｜DSML｜invoke[^>]*>|<\/｜DSML｜invoke>|<｜DSML｜parameter[^>]*>|<\/｜DSML｜parameter>|<tool_result>|<\/tool_result>)/g;
const esc=t=>t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const COLLAPSE=6000, HEAD=3000, TAIL=1800;
let expandStore={};
function renderText(t,segId){
  let html="";
  if(t.length>COLLAPSE){
    const midLen=t.length-HEAD-TAIL;
    expandStore[segId]=t;
    html+=chunk(t.slice(0,HEAD));
    html+=`<span class="expand" data-seg="${segId}" role="button" tabindex="0">⋯ show ${fmt(midLen)} hidden chars ⋯</span>`;
    html+=chunk(t.slice(t.length-TAIL));
  } else html+=chunk(t);
  return html;
}
function chunk(t){
  return t.split(SPECIALS).map(p=>{
    if(!p)return "";
    if(p.startsWith("<")&&p.match(SPECIALS)){
      let label=p.replace(/<｜DSML｜invoke name="([^"]*)"[^>]*>/,"DSML invoke: $1")
                 .replace(/<｜DSML｜parameter name="([^"]*)"[^>]*>/,"param: $1");
      return `<span class="tokchip">${esc(label)}</span>`;
    }
    return esc(p);
  }).join("");
}
async function loadFiles(){
  FILES=await (await fetch("api/files")).json();
  const sel=$("#file"), prev=sel.value;
  sel.innerHTML=FILES.map((f,i)=>f.rows==null
    ?`<option value="${i}" disabled>${f.name} — indexing…</option>`
    :`<option value="${i}">${f.name} — ${fmt(f.rows)} rows</option>`).join("");
  if(prev)sel.value=prev;
  if(FILES.some(f=>f.rows==null))setTimeout(loadFiles,8000);
  return FILES.findIndex(f=>f.rows!=null);
}
async function show(f,i){
  if(busy)return; busy=true;
  const n=FILES[f].rows;
  i=Math.max(0,Math.min(n-1,i)); cur={f,i};
  $("#idx").value=i; $("#total").textContent="/ "+fmt(n);
  $("#stream").innerHTML='<div class="loading">tokenizing row '+fmt(i)+' …</div>';
  try{
    const r=await (await fetch(`api/row?file=${f}&i=${i}`)).json();
    const badges=[
      `<span class="badge">id ${r.id||"—"}</span>`,
      `<span class="badge">conv ${(r.conv_key||"—").slice(0,18)}</span>`,
      `<span class="badge">${fmt(r.tokens)} tokens</span>`,
      `<span class="badge sup">${fmt(r.loss_tokens)} supervised (${(100*r.loss_tokens/r.tokens).toFixed(1)}%)</span>`,
      `<span class="badge">${r.thinking?"thinking mode":"chat mode"}</span>`,
      `<span class="badge">finish: ${r.finish_reason||"—"}</span>`,
    ];
    if(r.trimmed!==undefined) badges.push(`<span class="badge">${r.trimmed?`trimmed: ${r.turns_kept}/${r.turns_total} turns`:"untrimmed"}</span>`);
    $("#meta").innerHTML=badges.join("");
    let x=0,rects="";
    for(const s of r.segments){
      const w=100*s.n/r.tokens;
      rects+=`<rect x="${x.toFixed(3)}%" width="${Math.max(w,.08).toFixed(3)}%" y="0" height="14" fill="${s.m?"var(--sup)":"var(--line)"}"/>`;
      x+=w;
    }
    $("#map").innerHTML=rects;
    expandStore={};
    $("#stream").innerHTML=r.segments.map((s,k)=>
      `<span class="${s.m?"seg-sup":"seg-mask"}">${renderText(s.t,k)}</span>`).join("");
  }catch(e){ $("#stream").innerHTML='<div class="loading">error: '+esc(String(e))+'</div>'; }
  busy=false;
}
$("#stream").addEventListener("click",e=>{
  const el=e.target.closest(".expand"); if(!el)return;
  const t=expandStore[el.dataset.seg];
  el.outerHTML=chunk(t.slice(HEAD,t.length-TAIL));
});
$("#file").onchange=()=>show(+$("#file").value,0);
$("#prev").onclick=()=>show(cur.f,cur.i-1);
$("#next").onclick=()=>show(cur.f,cur.i+1);
$("#rand").onclick=()=>show(cur.f,Math.floor(Math.random()*FILES[cur.f].rows));
$("#idx").onchange=()=>show(cur.f,+$("#idx").value);
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT")return;
  if(e.key==="ArrowLeft")$("#prev").click();
  if(e.key==="ArrowRight")$("#next").click();
  if(e.key==="r")$("#rand").click();
});
(async()=>{
  let first=await loadFiles();
  while(first<0){await new Promise(r=>setTimeout(r,3000));first=await loadFiles();}
  $("#file").value=first; show(first,0);
})();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    files = []  # [(name, path)]

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        blob = json.dumps(obj, ensure_ascii=False).encode("utf-8", "replace")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            blob = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
        elif u.path == "/api/files":
            out = []
            for n, p in self.files:
                idx = index_for(p, build=False)
                out.append({"name": n, "rows": len(idx.offsets) if idx else None})
            self._json(out)
        elif u.path == "/api/row":
            q = parse_qs(u.query)
            fi = int(q.get("file", ["0"])[0])
            i = int(q.get("i", ["0"])[0])
            name, path = self.files[fi]
            idx = index_for(path)
            if not (0 <= i < len(idx.offsets)):
                self._json({"error": "row out of range"}, 404)
                return
            row = json.loads(idx.row(i))
            segs, n, loss = analyze(row["text"])
            self._json({
                "id": row.get("id"), "conv_key": row.get("conv_key"),
                "thinking": row.get("thinking"),
                "finish_reason": row.get("finish_reason"),
                "trimmed": row.get("trimmed"),
                "turns_kept": row.get("turns_kept"),
                "turns_total": row.get("turns_total"),
                "tokens": n, "loss_tokens": loss, "segments": segs,
            })
        else:
            self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8321)
    args = ap.parse_args()
    Handler.files = [
        (n, os.path.join(DATA_DIR, f))
        for n, f in CANDIDATES
        if os.path.exists(os.path.join(DATA_DIR, f))
    ]
    if not Handler.files:
        raise SystemExit("no datasets found in " + DATA_DIR)
    for n, p in Handler.files:
        print(f"  {n}: {p}")
    threading.Thread(
        target=prewarm, args=([p for _, p in Handler.files],), daemon=True
    ).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"viewer at http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
