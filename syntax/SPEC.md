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
- **Order is reading order**: top-to-bottom, then left-to-right within a row.
- **Geometry is optional and comes last on a line**: `@X,Y WxH` — top-left corner and size, in board pixels (origin top-left). Omitted → the renderer auto-lays it out.
- **Text is the identifier.** Edges refer to shapes by their quoted label. An explicit id is added only when needed (no label, or two identical labels): `rect api "API" ...`, then `api -> db`.

## Statements

| Statement | Form | Notes |
|---|---|---|
| Header | `board WxH` | Board size. Optional, first line. |
| Shape | `rect\|ellipse\|diamond [id] ["label"] [@X,Y WxH] [style]` | The label is the shape's own text. If the shape has children, the label is its title (drawn top-left); otherwise it's centered. |
| Text | `"text" [@X,Y WxH] [style]` | Free text. Font size is implied by height ÷ lines. |
| Edge | `A -> B [: "label"] [style]` | `->`, `<-`, `<->` arrows; `--` plain line. `A`/`B` are a quoted label or an id. |
| Comment | `# ...` | Ignored. |

**Style** is an optional bracket list: `[fill=blue stroke=red dashed]`.
Colors are names — `black gray red pink violet blue cyan teal green yellow orange` — or hex `#rrggbb`.

## Open questions (to iterate on)

- Should geometry be absolute, or relative to the parent? Relative may read better for nested content.
- Are coordinates useful to the LLM at all, or should we add position words (`@top-left`)?
- Container title: label vs. first child text — the vision pipeline can't always tell them apart.
- Font size / emphasis: is height enough, or do we want `[size=28]` / `[bold]`?
- Stable ids for deltas (once video arrives).
