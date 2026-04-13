import clsx from "clsx";
import { useMemo } from "react";
import { HighlightedText } from "./HighlightedText";

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
    <div className="h-[100px] relative rounded-[16px] shrink-0 w-full" style={{ backgroundColor: '#38423B' }}>
      <div className="flex flex-row items-center overflow-clip rounded-[inherit] size-full">
        <div className="content-stretch flex gap-[10px] items-center p-[15px] relative size-full">
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
      className="relative rounded-[8px] shrink-0 size-[70px] transition-colors cursor-pointer"
      style={{ backgroundColor: checked ? '#2D2D2A' : '#353831' }}
    >
      <div
        aria-hidden="true"
        className="absolute border-[0.5px] border-white border-solid inset-0 pointer-events-none rounded-[8px]"
      />
      {checked && (
        <div className="absolute inset-0 flex items-center justify-center">
          <svg width="45" height="45" viewBox="0 0 24 24" fill="none">
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

interface Highlight {
  start: number;
  end: number;
  type: "error" | "improve" | "suggestion";
  message: string;
}

interface ReviewResultPanelProps {
  reviewText: string;
  score: number;
  followGuidelines: boolean;
  hasGrammaticalErrors: boolean;
  onBack?: () => void;
}

export default function ReviewResultPanel({
  reviewText,
  score,
  followGuidelines,
  hasGrammaticalErrors,
  onBack,
}: ReviewResultPanelProps) {
  const getEmojiForScore = (score: number): string => {
    if (score <= 20) return "😢";
    if (score <= 40) return "😕";
    if (score <= 60) return "😐";
    if (score <= 80) return "😊";
    return "🤩";
  };

  // Mock highlights - in real implementation this would come from the AI model
  const highlights: Highlight[] = useMemo(() => {
    const words = reviewText.split(" ");
    const result: Highlight[] = [];
    
    // Generate some mock highlights
    if (words.length > 5) {
      const firstErrorStart = reviewText.indexOf(words[2]);
      result.push({
        start: firstErrorStart,
        end: firstErrorStart + words[2].length,
        type: "error",
        message: "Errore grammaticale: accordo soggetto-verbo",
      });
    }
    
    if (words.length > 10) {
      const improveStart = reviewText.indexOf(words[7]);
      result.push({
        start: improveStart,
        end: improveStart + words[7].length + words[8]?.length + 1 || words[7].length,
        type: "improve",
        message: "Da migliorare: espressione poco chiara",
      });
    }
    
    if (words.length > 15) {
      const suggestionStart = reviewText.indexOf(words[12]);
      result.push({
        start: suggestionStart,
        end: suggestionStart + words[12].length,
        type: "suggestion",
        message: "Suggerimento: aggiungi più dettagli qui",
      });
    }
    
    return result;
  }, [reviewText]);

  const modelComment = `Analisi generale: La recensione presenta ${highlights.length} punti che richiedono attenzione. Il testo mostra una buona struttura generale, ma ci sono alcune aree che potrebbero essere migliorate per aumentare la chiarezza e l'impatto. Si consiglia di rivedere gli errori grammaticali evidenziati e di espandere le sezioni marcate con suggerimenti per fornire maggiore profondità all'analisi.`;

  return (
    <div className="flex gap-[15px] h-full w-full">
      {/* Main Review Text Panel */}
      <div
        className="flex-1 min-w-0 h-full content-stretch flex flex-col gap-[15px] items-center p-[15px] relative rounded-[24px]"
        style={{ backgroundColor: '#353831' }}
        data-name="review_result"
      >
        <div className="flex items-center justify-between w-full">
          <div
            className="flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] relative shrink-0 text-[50px] text-white"
            style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
          >
            <p className="leading-[normal]">Review Analysis</p>
          </div>
          {onBack && (
            <button
              onClick={onBack}
              className="transition-colors px-6 py-3 rounded-[16px] text-[20px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white"
              style={{ 
                backgroundColor: '#38423B',
                fontVariationSettings: "'GRAD' 0, 'wdth' 100"
              }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#454f48'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#38423B'}
            >
              ← Back
            </button>
          )}
        </div>

        {/* Highlighted Review Text */}
        <div
          className="relative rounded-[16px] shrink-0 w-full flex-[1_0_0] min-h-0 overflow-visible"
          style={{ backgroundColor: '#38423B' }}
          data-name="highlighted_review_text"
        >
          <div className="flex flex-col items-start overflow-y-auto rounded-[inherit] size-full p-[15px] scrollbar-hide">
            <div
              className="w-full text-[22px] font-['Roboto_Serif',sans-serif] text-white leading-relaxed pb-12"
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            >
              <HighlightedText
                text={reviewText}
                highlights={highlights}
              />
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex gap-4 w-full justify-center items-center">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-red-400 rounded"></div>
            <span
              className="text-[16px] font-['Roboto_Serif',sans-serif] text-white"
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            >
              Errori
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-yellow-400 rounded"></div>
            <span
              className="text-[16px] font-['Roboto_Serif',sans-serif] text-white"
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            >
              Da migliorare
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-green-400 rounded"></div>
            <span
              className="text-[16px] font-['Roboto_Serif',sans-serif] text-white"
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            >
              Suggerimenti
            </span>
          </div>
        </div>

        {/* Score and Checks */}
        <div
          className="w-full relative rounded-[16px] h-[280px]"
          style={{ backgroundColor: '#38423B' }}
          data-name="ai_output"
        >
          <div className="overflow-clip rounded-[inherit] size-full">
            <div className="content-stretch flex gap-[15px] items-start p-[15px] relative size-full">
              <div
                className="flex-[1_0_0] h-full min-h-px min-w-px relative rounded-[24px]"
                style={{ backgroundColor: '#2D2D2A' }}
                data-name="score"
              >
                <div className="flex flex-col items-center overflow-clip rounded-[inherit] size-full">
                  <div className="content-stretch flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium gap-[10px] items-center leading-[0] p-[10px] relative size-full text-center text-white">
                    <div
                      className="flex flex-col justify-center relative shrink-0 text-[40px] w-full"
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    >
                      <p className="leading-[normal]">Insightfulness Score</p>
                    </div>
                    <div
                      className="flex flex-[1_0_0] flex-col justify-center min-h-px min-w-px relative text-[80px] w-full"
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    >
                      <p className="leading-[normal]">
                        {score}/100 {getEmojiForScore(score)}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div
                className="h-full relative rounded-[24px] shrink-0 w-[400px]"
                style={{ backgroundColor: '#2D2D2A' }}
                data-name="others"
              >
                <div className="flex flex-col items-center justify-center overflow-clip rounded-[inherit] size-full">
                  <div className="content-stretch flex flex-col gap-[20px] items-center justify-center p-[20px] relative size-full">
                    <Wrapper>
                      <Checkbox checked={followGuidelines} onChange={() => {}} />
                      <Text text="Follow guidelines" additionalClassNames="h-[90px] text-[34px]" />
                    </Wrapper>
                    <Wrapper>
                      <Checkbox checked={hasGrammaticalErrors} onChange={() => {}} />
                      <Text
                        text="Grammatical errors"
                        additionalClassNames="h-[90px] text-[34px]"
                      />
                    </Wrapper>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Comment Panel (replaces LinkSitePanel) */}
      <div
        className="h-full relative rounded-[24px] shrink-0 w-[450px]"
        style={{ backgroundColor: '#353831' }}
        data-name="model_comment"
      >
        <div className="flex flex-col items-center overflow-y-auto rounded-[inherit] size-full p-[15px] gap-[15px] scrollbar-hide">
          <div
            style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            className="flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] relative shrink-0 text-[38px] text-white text-center"
          >
            <p className="leading-[normal]">Model Comment</p>
          </div>

          <div className="rounded-[16px] p-[20px] w-full flex-1" style={{ backgroundColor: '#38423B' }}>
            <p
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
              className="text-[18px] font-['Roboto_Serif',sans-serif] text-white leading-relaxed"
            >
              {modelComment}
            </p>
          </div>

          {/* Detailed Highlights List */}
          <div className="w-full space-y-2">
            {highlights.map((highlight, idx) => (
              <div
                key={idx}
                className="p-3 rounded-[12px]"
                style={{
                  backgroundColor:
                    highlight.type === "error"
                      ? "rgba(239, 68, 68, 0.25)"
                      : highlight.type === "improve"
                      ? "rgba(251, 191, 36, 0.25)"
                      : "rgba(34, 197, 94, 0.25)",
                }}
              >
                <p
                  style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                  className="text-[16px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white mb-1"
                >
                  {highlight.type === "error"
                    ? "Errore"
                    : highlight.type === "improve"
                    ? "Da migliorare"
                    : "Suggerimento"}
                </p>
                <p
                  style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                  className="text-[14px] font-['Roboto_Serif',sans-serif] text-white/80"
                >
                  {highlight.message}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}