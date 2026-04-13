import { useState } from "react";

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

  if (highlights.length === 0) {
    return <span>{text}</span>;
  }

  const sortedHighlights = [...highlights].sort((a, b) => a.start - b.start);
  const segments: JSX.Element[] = [];
  let currentIndex = 0;

  const handleMouseEnter = (idx: number, event: React.MouseEvent<HTMLSpanElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setTooltipPosition({
      x: rect.left + rect.width / 2,
      y: rect.top,
    });
    setHoveredIdx(idx);
  };

  const handleMouseLeave = () => {
    setHoveredIdx(null);
  };

  sortedHighlights.forEach((highlight, idx) => {
    // Add text before highlight
    if (currentIndex < highlight.start) {
      segments.push(
        <span key={`text-${idx}`}>
          {text.substring(currentIndex, highlight.start)}
        </span>
      );
    }

    // Add highlighted text
    const bgColor =
      highlight.type === "error"
        ? "bg-red-300/60"
        : highlight.type === "improve"
        ? "bg-yellow-300/60"
        : "bg-green-300/60";

    segments.push(
      <span
        key={`highlight-${idx}`}
        className={`${bgColor} px-1 rounded cursor-pointer relative inline-block`}
        onMouseEnter={(e) => handleMouseEnter(idx, e)}
        onMouseLeave={handleMouseLeave}
        title={highlight.message}
      >
        {text.substring(highlight.start, highlight.end)}
      </span>
    );

    currentIndex = highlight.end;
  });

  // Add remaining text
  if (currentIndex < text.length) {
    segments.push(
      <span key="text-end">{text.substring(currentIndex)}</span>
    );
  }

  return (
    <>
      {segments}
      {hoveredIdx !== null && (
        <div
          className="fixed bg-black text-white text-[14px] px-4 py-2 rounded-lg whitespace-nowrap shadow-xl pointer-events-none"
          style={{
            left: `${tooltipPosition.x}px`,
            top: `${tooltipPosition.y - 40}px`,
            transform: "translateX(-50%)",
            zIndex: 99999,
          }}
        >
          {sortedHighlights[hoveredIdx].message}
        </div>
      )}
    </>
  );
}
