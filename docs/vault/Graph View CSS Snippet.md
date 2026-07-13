---
tags:
  - football-prediction
  - obsidian
  - css
  - graph-view
  - customization
created: 2026-07-12
---

# 🎨 Graph View CSS Snippet

> Enhanced Graph View styling for this vault — tag node colors, transparent edges, focused node highlight, and hidden labels by default.

---

## 📦 The Snippet

**File:** `.obsidian/snippets/graph-view-enhancer.css`  
**Location:** Already placed in this vault's snippets folder.

### What It Does

| Feature | How It Works |
|---------|--------------|
| 🟦 **Regular nodes** | Soft blue (#4a8dcf) |
| 🟧 **Tag nodes** (`#tagname`) | Amber (#e6a84c) |
| 🟩 **Attachment nodes** (images, PDFs) | Sage green (#6ebf7a) |
| 🟥 **Unresolved links** (broken wikilinks) | Soft red (#d46a6a) |
| 🔗 **Edges** | 50% transparent — structure visible without visual noise |
| ✨ **Active/focused node** | Bright gold (#ffd866) — acts as a "glow" effect within WebGL limits |
| 🏷️ **Labels** | Hidden by default → shown when hovering the graph area |
| 🌙 **Dark/Light mode** | Slight tint adjustments for both themes |

---

## ⚙️ Installation

The CSS file is already in the vault at `.obsidian/snippets/graph-view-enhancer.css`.  
Just enable it:

```text
1. Open Obsidian → Settings (Ctrl+,)
2. Go to Appearance → scroll to "CSS snippets"
3. Click the folder icon 📁 (opens the snippets folder)
   → Verify graph-view-enhancer.css is there
4. Back in Obsidian, click the refresh button ↻
5. Find "graph-view-enhancer" in the list → toggle it ON
```

> **💡 Tip:** Changes take effect immediately — no restart needed. Toggle off to revert.

---

## 🧪 Testing It

Open Graph View (`Ctrl/Cmd+G`) and check:

1. **Node colors** — regular notes are blue, tags (#football-prediction, etc.) are amber
2. **Edges** — should appear softer and more transparent than before
3. **Click a node** — it should turn bright gold
4. **Hover over graph area** — labels fade in; move mouse away, labels fade out

---

## ⚠ Known Limitations (Canvas/WebGL)

Obsidian's Graph View is rendered with **WebGL on a `<canvas>` element**, not HTML. CSS has limited control:

| Want | Possible? | Workaround |
|------|-----------|------------|
| **Per-tag colors** (e.g. `#ml` = red, `#data` = blue) | ❌ Via CSS | Use built-in **Groups** feature: Graph View panel → Groups → New Group → filter by tag → pick color |
| **True glow effect** (drop-shadow, blur) | ❌ CSS doesn't affect WebGL | Brighter color on focused node approximates it |
| **Per-node hover effects** | ❌ Canvas renders as one image | Only whole-graph hover works |
| **Individual label toggles** | ❌ All labels toggle together | Use the "Labels" button in Graph View toolbar for per-session control |
| **Thicker/thinner edges** | ❌ Not exposed via CSS bridge | Use Arc theme or wait for Obsidian update |

---

## 🎨 Customizing Further

Edit `docs/vault/.obsidian/snippets/graph-view-enhancer.css` directly — each section is commented. Change the hex color values to match your preference.

For **per-tag coloring** (e.g., all `#python-module` nodes in purple):

```text
1. Open Graph View (Ctrl/Cmd+G)
2. Click the "Groups" icon in the top-right of the Graph panel
3. Click "New Group"
4. Filter: tag:#python-module
5. Pick a color → click anywhere to save
```

Repeat for any tags you want to color individually.

---

## Related

- [[Quick Start Guide]] — vault setup & recommended plugins
- [[Code Link Plugin Setup]] — making `[[wikilinks]]` resolve to `.py` source files
- [[Football Prediction Codebase]] — vault home
