import { Trash2 } from "lucide-react";
import { useState } from "react";

interface ChatItemProps {
  title: string;
  isActive?: boolean;
  onClick?: () => void;
  onDelete?: () => void;
}

export function ChatItem({ title, isActive = false, onClick, onDelete }: ChatItemProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className="relative content-stretch flex gap-[10px] h-[80px] items-center overflow-clip p-[10px] rounded-[16px] shrink-0 w-full cursor-pointer transition-all"
      style={{ backgroundColor: isActive ? '#2D2D2A' : '#353831' }}
      onClick={onClick}
      onMouseEnter={(e) => {
        setIsHovered(true);
        if (!isActive) e.currentTarget.style.backgroundColor = '#3d4739';
      }}
      onMouseLeave={(e) => {
        setIsHovered(false);
        if (!isActive) e.currentTarget.style.backgroundColor = '#353831';
      }}
    >
      <div
        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
        className="flex-1 flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-h-px min-w-0 overflow-hidden relative text-white text-ellipsis whitespace-nowrap"
      >
        <p className="leading-[normal] overflow-hidden text-ellipsis text-[24px]">{title}</p>
      </div>
      {isHovered && onDelete && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="shrink-0 p-2 hover:bg-red-500/30 rounded-lg transition-colors"
          aria-label="Delete chat"
        >
          <Trash2 className="size-5 text-red-400" />
        </button>
      )}
    </div>
  );
}