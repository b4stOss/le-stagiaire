/** Pixel flame in the five Mistral band colors, condensed to three rows.
    Doubles as the app's loader: the same mark animates while the agent works. */

const GRID: Array<{ row: number; col: number }> = [
  { row: 0, col: 1 },
  { row: 1, col: 0 },
  { row: 1, col: 1 },
  { row: 1, col: 2 },
  { row: 2, col: 0 },
  { row: 2, col: 1 },
  { row: 2, col: 2 },
];

const ROW_COLORS = ["#ffd800", "#ff8205", "#e10500"];

export default function FlameMark({ px = 5, animate = false }: { px?: number; animate?: boolean }) {
  const gap = Math.max(1, Math.round(px * 0.3));
  return (
    <span
      aria-hidden="true"
      className="flame-mark"
      style={{
        gridTemplateColumns: `repeat(3, ${px}px)`,
        gridTemplateRows: `repeat(3, ${px}px)`,
        gap,
      }}
    >
      {GRID.map(({ row, col }, i) => (
        <span
          key={i}
          className={`flame-pixel${animate ? " lit" : ""}`}
          style={{
            gridRow: row + 1,
            gridColumn: col + 1,
            background: ROW_COLORS[row],
            animationDelay: animate ? `${col * 90 + row * 60}ms` : undefined,
          }}
        />
      ))}
    </span>
  );
}
