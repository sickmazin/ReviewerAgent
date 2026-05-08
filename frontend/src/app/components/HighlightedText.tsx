import { useState, useMemo } from "react";

interface Highlight {
  start: number;
  end: number;
  type: "error" | "improve" | "suggestion";
  message: string;
}

interface HighlightedTextProps {
  text: string;
  highlights: Highlight[];
}

export function HighlightedText({ text, highlights }: HighlightedTextProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  const fragments = useMemo(() => {
    const activeHighlights = highlights.filter(h => (h as any).is_highlight !== false);
    if (activeHighlights.length === 0) return [{ start: 0, end: text.length, highlight: null }];

    // Priority mapping: higher is better (more important)
    const priority = {
      error: 3,
      improve: 2,
      suggestion: 1,
    };

    // Create a set of all boundary points
    const points = new Set<number>([0, text.length]);
    activeHighlights.forEach((h) => {
      points.add(Math.max(0, Math.min(text.length, h.start)));
      points.add(Math.max(0, Math.min(text.length, h.end)));
    });

    // Sort boundary points
    const sortedPoints = Array.from(points).sort((a, b) => a - b);

    // Create non-overlapping fragments
    const fragments: { start: number; end: number; highlight: Highlight | null }[] = [];
    for (let i = 0; i < sortedPoints.length - 1; i++) {
      const start = sortedPoints[i];
      const end = sortedPoints[i + 1];
      if (start === end) continue;

      // Find all highlights that cover this fragment
      const covering = activeHighlights.filter((h) => h.start <= start && h.end >= end);

      if (covering.length === 0) {
        fragments.push({ start, end, highlight: null });
      } else {
        // Pick the highlight with the highest priority
        // If same priority, pick the shortest one (most specific)
        const best = covering.reduce((prev, curr) => {
          if (priority[curr.type] > priority[prev.type]) return curr;
          if (priority[curr.type] < priority[prev.type]) return prev;
          // Same priority: pick the one that covers a smaller range (more specific)
          const currLen = curr.end - curr.start;
          const prevLen = prev.end - prev.start;
          return currLen < prevLen ? curr : prev;
        });
        fragments.push({ start, end, highlight: best });
      }
    }
    return fragments;
  }, [text, highlights]);

  const handleMouseEnter = (highlight: Highlight, event: React.MouseEvent<HTMLSpanElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setTooltipPosition({
      x: rect.left + rect.width / 2,
      y: rect.top,
    });
    // Find the actual highlight object in the original array to get its index or just store the message
    setHoveredIdx(highlights.indexOf(highlight));
  };

  const handleMouseLeave = () => {
    setHoveredIdx(null);
  };

  return (
    <>
      {fragments.map((frag, idx) => {
        const fragText = text.substring(frag.start, frag.end);
        if (!frag.highlight) {
          return <span key={idx}>{fragText}</span>;
        }

        const bgColor =
          frag.highlight.type === "error"
            ? "bg-red-400/60"
            : frag.highlight.type === "improve"
            ? "bg-yellow-400/60"
            : "bg-green-400/60";

        return (
          <span
            key={idx}
            className={`${bgColor} px-0.5 rounded-sm cursor-pointer transition-colors hover:brightness-110`}
            onMouseEnter={(e) => handleMouseEnter(frag.highlight!, e)}
            onMouseLeave={handleMouseLeave}
          >
            {fragText}
          </span>
        );
      })}
      {hoveredIdx !== null && (
        <div
          className="fixed bg-black/90 text-white text-[14px] px-4 py-2 rounded-lg shadow-2xl pointer-events-none z-[99999]"
          style={{
            left: `${tooltipPosition.x}px`,
            top: `${tooltipPosition.y - 10}px`,
            transform: "translate(-50%, -100%)",
            maxWidth: "300px",
            whiteSpace: "normal"
          }}
        >
          <div className="font-bold mb-1 uppercase text-[10px] opacity-70">
            {highlights[hoveredIdx].type === 'error' ? 'Errore' : highlights[hoveredIdx].type === 'improve' ? 'Da migliorare' : 'Suggerimento'}
          </div>
          {highlights[hoveredIdx].message}
        </div>
      )}
    </>
  );
}
