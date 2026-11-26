# Board syntax — v0

A plain-text format for whiteboard state. It has two jobs:

1. **An LLM reads it** and immediately understands what is on the board and how it is laid out — no spec needed.
2. **A renderer draws it** (our Excalidraw-based viewer), so it carries enough geometry to reproduce the board.

v0 is deliberately small. We will iterate on it against real outputs.

## Example

```
board 1280x800

"System Architecture" @40,30 330x40

rect "Web App" @60,330 180x90 [fill=blue]
rect "Backend" @320,120 520x560 [dashed]
  rect "API Gateway" @360,180 200x80 [fill=violet]
  rect "Auth Service" @600,180 200x80
  rect "Orders Service" @480,420 220x80
ellipse "Postgres" @960,300 200x110 [fill=green]

"Web App" -> "API Gateway" : "HTTPS"
"API Gateway" -> "Auth Service" : "verify JWT"
"Orders Service" -> "Postgres" : "SQL"
```

## Rules

- **Quoted text is exactly what is on the board.** Anything in `"..."` is the literal text (escapes: `\"`, `\\`, `\n`). Everything else is structure.
- **Indentation (2 spaces) means "visually inside".** A child is drawn inside its parent.
- **Order is reading order**: the screen is split into full-width bands top-to-bottom, bands into columns left-to-right (real layout columns become `group`s), then rows within each.
- **Geometry is optional and comes last on a line**: `@X,Y WxH` — top-left corner and size, origin top-left. Converted screens are normalised to a **1000-wide** board (`board 1000x645`), so numbers read as rough thousandths of the width. Omitted → the renderer auto-lays it out.
- **Text is the identifier.** Edges refer to shapes by their quoted label. An explicit id is added only when needed (no label, or two identical labels): `rect api "API" ...`, then `api -> db`.

## Statements

| Statement | Form | Notes |
|---|---|---|
| Header | `board WxH` | Board size. Optional, first line. |
| Shape | `rect\|ellipse\|diamond [id] ["label"] [@X,Y WxH] [style]` | The label is the shape's own text. If the shape has children, the label is its title (drawn top-left); otherwise it's centered. |
| Image | `image [@X,Y WxH]` | A photo/illustration. Can contain text drawn over it (indented children). |
| Group | `group [@X,Y WxH]` + indented children | An invisible layout column (e.g. left nav / feed / sidebar). Only structure: nothing is drawn. |
| Icon run | `icons N [@X,Y WxH]` | A row of N icons (toolbars, action bars). |
| Icon | `icon [@X,Y WxH]` | A small symbol. We can't tell *which* symbol without a vision model; position (e.g. just left of a label) carries the meaning. |
| Table | `table [@X,Y WxH] cols X1,X2,… [size=N]` + indented rows `"a" \| "b" \| ""` | Text aligned in rows and columns. `cols` are the column x positions; rows are spaced evenly over the height; `size` is the font size. |
| Text | `"text" [@X,Y WxH] [style]` | Free text. Font size is implied by height ÷ lines. |
| Edge | `A -> B [: "label"] [style]` | `->`, `<-`, `<->` arrows; `--` plain line. `A`/`B` are a quoted label or an id. |
| Comment | `# ...` | Ignored. |

**Style** is an optional bracket list: `[fill=blue stroke=red dashed]`.
Text flags: `[bold]` (heavier strokes than the screen's normal text, often meaning selected/active or a heading) and `[icon]` (an icon sits just left of the text, as in nav menus).
Colors are names — `black gray red pink violet blue cyan teal green yellow orange` — or hex `#rrggbb`.

## Example: a converted UI fragment

```
rect @2,52 168x572
  icon @18,75 19x18
  "Overview" @39,72 81x22
  icon @17,110 25x19
  "Reports" @37,107 64x22
rect @202,266 750x312 [stroke=gray]
  "Recent orders" @219,281 124x20
  table @213,323 554x180 cols 214,323,527,698 [size=17]
    "Order" | "Customer" | "Amount" | "Status"
    "#1042" | "Jane Cooper" | "$120.00" | "Shipped"
```

## Open questions (to iterate on)

- Should geometry be absolute, or relative to the parent? Relative may read better for nested content.
- Are coordinates useful to the LLM at all, or should we add position words (`@top-left`)?
- Container title: label vs. first child text — the vision pipeline can't always tell them apart.
- Font size / emphasis: is height enough, or do we want `[size=28]` / `[bold]`?
- Stable ids for deltas (once video arrives).
