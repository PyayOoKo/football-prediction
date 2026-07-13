---
tags:
  - football-prediction
  - obsidian
  - css
  - theme
  - customization
created: 2026-07-12
---

# 🎨 Vault Theme CSS Snippet

> A custom colour palette inspired by football — pitch green, trophy gold, and team colours — for code blocks, headings, callouts, and more.

---

## 🏟️ Colour Palette

| Colour | Hex | Used For |
|--------|-----|----------|
| **Pitch Green** | `#2d8a4e` | H1 headings, accent, blockquotes, tags, active file |
| **Trophy Gold** | `#d4a843` | H2 headings, bold text, links on hover, search matches |
| **Team Blue** | `#2980b9` | H3, wikilinks, italic text |
| **Team Purple** | `#8e44ad` | H4, code keywords, question callouts |
| **Team Red** | `#c0392b` | Error/danger callouts |
| **Team Orange** | `#d35400` | Warning callouts, code strings (dark) |
| **Team Green** | `#27ae60` | Success callouts, external links, code numbers |

---

## ✨ What Gets Styled

| Element | Styles Applied |
|---------|----------------|
| **H1–H6 headings** | Custom colours per level — green → gold → blue → purple → teal → grey |
| **H1 underline** | Subtle green bottom border |
| **Code blocks** | Dark charcoal background (dark) / off-white (light), 8px border-radius, border |
| **Inline code** | Rounded background with padding |
| **Syntax highlighting** | Comment, keyword, string, function, number, operator, property, punctuation |
| **Callouts** | 10px border-radius, 4px left border, team colours per type (note=green, tip=gold, info=blue, warning=orange, danger=red, question=purple, quote=grey) |
| **Blockquotes** | Green left border, subtle green background tint, rounded right corners |
| **Wikilinks** | Blue → turn gold on hover, underline transition |
| **External links** | Green, dashed underline on hover |
| **Tags** | Pill-shaped with green border, filled green on hover |
| **Horizontal rules** | Green-to-gold gradient fade |
| **Tables** | Green header row, alternating row colours, hover highlight |
| **Checkboxes** | Green accent colour |
| **Bold / italic** | Gold bold, blue italic |
| **UI elements** | Green active file indicator, gold search matches, green selection colour |

---

## ⚙️ Installation

The CSS file is already in the vault at `.obsidian/snippets/vault-theme-enhancer.css`.

```text
1. Open Obsidian → Settings (Ctrl+,)
2. Go to Appearance → scroll to "CSS snippets"
3. Click the refresh button ↻
4. Find "vault-theme-enhancer" in the list → toggle it ON
```

> **💡 Tip:** If you already have the [[Graph View CSS Snippet]] enabled, both can be active simultaneously — they affect different parts of the UI.

---

## 🧪 Preview

After enabling, check these elements:

| Check | What to Look For |
|-------|-----------------|
| **Headings** | H1 = green, H2 = gold, H3 = blue, H4 = purple |
| **Code block** | Dark background with coloured syntax, rounded border |
| **Callout** | `> [!tip]` has gold left border, `> [!warning]` orange |
| **Tag** | `#football-prediction` appears as a green pill |
| **Link hover** | `[[config.py]]` turns gold when you hover |
| **Table** | Green header, alternating rows |
| **HR** | Green-to-gold gradient line |

---

## 🎨 Customising

Edit `docs/vault/.obsidian/snippets/vault-theme-enhancer.css` directly. The file is organized into 11 numbered sections:

1. **Core Palette** — change the colour variables at the top to shift the entire theme
2. **Headings** — per-level colours, sizes, weights
3. **Code Blocks** — background, text, syntax colours (separate dark/light)
4. **Callouts** — per-type border colours
5. **Blockquotes** — border + background tint
6. **Links** — internal, external, unresolved
7. **Tags** — pill shape, hover fill
8. **Horizontal Rules** — gradient
9. **Misc Markdown** — tables, checkboxes, bold/italic
10. **UI Tweaks** — file explorer, search, selection
11. **Print / Export** — print-friendly overrides

---

## ⚠️ Notes

- **Compatibility:** Works with both Dark and Light themes (the snippet adapts via `.theme-dark` / `.theme-light` blocks where needed).
- **Other themes:** If you use a community theme (e.g., Minimal, Atom), this snippet will **override** its colours for the styled elements. Disable the snippet to revert.
- **Print:** Section 11 provides print-friendly overrides so exported PDFs look good too.

---

## Related

- [[Graph View CSS Snippet]] — Graph View node/edge styling
- [[Quick Start Guide]] — vault setup & recommended plugins
- [[Football Prediction Codebase]] — vault home
