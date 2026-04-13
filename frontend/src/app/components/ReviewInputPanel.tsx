import clsx from "clsx";
import { useState } from "react";

type TextProps = {
  text: string;
  additionalClassNames?: string;
};

function Text({ text, additionalClassNames = "" }: TextProps) {
  return (
    <div
      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
      className={clsx(
        "flex flex-[1_0_0] flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-h-px min-w-px relative text-white",
        additionalClassNames
      )}
    >
      <p className="leading-[normal]">{text}</p>
    </div>
  );
}

function Wrapper({ children }: React.PropsWithChildren<{}>) {
  return (
    <div className="h-[80px] relative rounded-[16px] shrink-0 w-full" style={{ backgroundColor: '#353831' }}>
      <div className="flex flex-row items-center overflow-clip rounded-[inherit] size-full">
        <div className="content-stretch flex gap-[10px] items-center p-[10px] relative size-full">
          {children}
        </div>
      </div>
    </div>
  );
}

interface CheckboxProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
}

function Checkbox({ checked, onChange }: CheckboxProps) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={clsx(
        "relative rounded-[8px] shrink-0 size-[65px] transition-colors cursor-pointer",
        checked ? "bg-[#8f8f8f]" : "bg-white"
      )}
    >
      <div
        aria-hidden="true"
        className="absolute border-[0.5px] border-black border-solid inset-0 pointer-events-none rounded-[8px]"
      />
      {checked && (
        <div className="absolute inset-0 flex items-center justify-center">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none">
            <path
              d="M5 13L9 17L19 7"
              stroke="white"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      )}
    </button>
  );
}

export default function ReviewInputPanel() {
  const [reviewText, setReviewText] = useState("");

  const handleEvaluate = () => {
    // Non fare nulla se il testo è vuoto
    if (!reviewText.trim()) {
      return;
    }
    
    // Simulazione della valutazione
    const randomScore = Math.floor(Math.random() * 41) + 60; // Score tra 60 e 100
    
    // Call parent callback if provided
    if (typeof window !== 'undefined' && (window as any).onEvaluateScore) {
      (window as any).onEvaluateScore({
        reviewText,
        score: randomScore,
        followGuidelines: Math.random() > 0.3,
        hasGrammaticalErrors: Math.random() > 0.5,
      });
    }
  };

  return (
    <div
      className="content-stretch flex flex-col gap-[15px] items-center overflow-clip p-[15px] relative rounded-[24px] size-full"
      style={{ backgroundColor: '#353831' }}
      data-name="review_input"
    >
      <div
        className="flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-w-full relative shrink-0 text-[50px] text-white w-[min-content]"
        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
      >
        <p className="leading-[normal]">Put review here</p>
      </div>
      
      <div
        className="relative rounded-[16px] flex-1 w-full min-h-0"
        style={{ backgroundColor: '#38423B' }}
        data-name="review_text_input"
      >
        <div className="flex flex-col items-center justify-center overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex flex-col items-center justify-start p-[15px] relative size-full">
            <textarea
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
              placeholder="Review..."
              className="w-full h-full resize-none bg-transparent text-[24px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white placeholder:text-white/40 outline-none leading-relaxed"
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            />
          </div>
        </div>
      </div>
      
      <button
        onClick={handleEvaluate}
        disabled={!reviewText.trim()}
        className="content-stretch flex flex-col h-[80px] items-center justify-center overflow-clip p-[5px] relative rounded-[26px] shrink-0 w-[300px] transition-colors"
        style={{
          backgroundColor: !reviewText.trim() ? '#38423B' : '#8B0000',
          cursor: !reviewText.trim() ? 'not-allowed' : 'pointer',
          opacity: !reviewText.trim() ? 0.5 : 1
        }}
        onMouseEnter={(e) => {
          if (reviewText.trim()) e.currentTarget.style.backgroundColor = '#A00000';
        }}
        onMouseLeave={(e) => {
          if (reviewText.trim()) e.currentTarget.style.backgroundColor = '#8B0000';
        }}
        data-name="score_button"
      >
        <div
          style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
          className="flex flex-[1_0_0] flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-h-px min-w-px relative text-white text-[40px] text-center w-full"
        >
          <p className="leading-[normal]">Evaluate score</p>
        </div>
      </button>
    </div>
  );
}