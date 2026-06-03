import clsx from "clsx";
import { HighlightedText } from "./HighlightedText";
import { Review } from "../api/client";

type TextProps = { text: string; additionalClassNames?: string };
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

interface CheckboxProps { checked: boolean }
function Checkbox({ checked }: CheckboxProps) {
  return (
    <div
      className="relative rounded-[8px] shrink-0 size-[70px]"
      style={{ backgroundColor: checked ? '#2D2D2A' : '#353831' }}
    >
      <div
        aria-hidden="true"
        className="absolute border-[0.5px] border-white border-solid inset-0 pointer-events-none rounded-[8px]"
      />
      {checked && (
        <div className="absolute inset-0 flex items-center justify-center">
          <svg width="45" height="45" viewBox="0 0 24 24" fill="none">
            <path d="M5 13L9 17L19 7" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      )}
    </div>
  );
}

function emojiForScore(score: number | string | null): string {
  if (score === null) return "❓";
  
  if (typeof score === 'string') {
    if (score === "BAD") return "😢";
    if (score === "GOOD") return "😊";
    if (score === "EXCELLENT") return "🤩";
  }
  
  const numScore = Number(score);
  if (isNaN(numScore)) return "❓";
  
  if (numScore <= 36) return "😢";
  if (numScore <= 70) return "😊";
  return "🤩";
}

interface ReviewResultPanelProps {
  review: Review;
  onBack?: () => void;
}

export default function ReviewResultPanel({ review, onBack }: ReviewResultPanelProps) {
  const reviewText = review.highlights?.text ?? review.text;
  const issues = review.highlights?.issues ?? [];
  const score = review.score;

  return (
    <div className="flex gap-[15px] h-full w-full">
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
            <p className="leading-[normal]">{review.title ?? "Review Analysis"}</p>
          </div>
          {onBack && (
            <button
              onClick={onBack}
              className="transition-colors px-6 py-3 rounded-[16px] text-[20px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white"
              style={{ backgroundColor: '#38423B', fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#454f48'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#38423B'}
            >
              ← Back
            </button>
          )}
        </div>

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
              <HighlightedText text={reviewText} highlights={issues} />
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex gap-4 w-full justify-center items-center">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-red-400 rounded"></div>
            <span className="text-[16px] font-['Roboto_Serif',sans-serif] text-white" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>Errors</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-yellow-400 rounded"></div>
            <span className="text-[16px] font-['Roboto_Serif',sans-serif] text-white" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>To improve</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-green-400 rounded"></div>
            <span className="text-[16px] font-['Roboto_Serif',sans-serif] text-white" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>Suggestions</span>
          </div>
        </div>

        {/* Score and checks */}
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
                    <div className="flex flex-col justify-center relative shrink-0 text-[40px] w-full" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>
                      <p className="leading-[normal]">Insightfulness Score</p>
                    </div>
                    <div className="flex flex-[1_0_0] flex-col justify-center min-h-px min-w-px relative text-[80px] w-full" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>
                      <p className="leading-[normal]">{score ?? "ERROR"} {emojiForScore(score)}</p>
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
                      <Checkbox checked={!!review.follow_guidelines} />
                      <Text text="Follow guidelines" additionalClassNames="h-[90px] text-[34px]" />
                    </Wrapper>
                    <Wrapper>
                      <Checkbox checked={review.grammar_errors} />
                      <Text text="Grammar errors" additionalClassNames="h-[90px] text-[34px]" />
                    </Wrapper>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right side: model reasoning + issues */}
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
            <p className="leading-[normal]">Model Reasoning</p>
          </div>

          {review.reasoning && (
            <div className="rounded-[16px] p-[20px] w-full" style={{ backgroundColor: '#38423B' }}>
              <p
                style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                className="text-[18px] font-['Roboto_Serif',sans-serif] text-white leading-relaxed"
              >
                {review.reasoning}
              </p>
            </div>
          )}

          {/* Details */}
          {review.details && (
            <div className="w-full flex gap-[10px]">
              {review.details.word_count !== undefined && (
                <div className="flex-1 rounded-[12px] p-3 text-center" style={{ backgroundColor: '#38423B' }}>
                  <p style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className="text-[14px] font-['Roboto_Serif',sans-serif] text-white/60">Words</p>
                  <p style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className="text-[24px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white">{review.details.word_count}</p>
                </div>
              )}
              {review.details.char_count !== undefined && (
                <div className="flex-1 rounded-[12px] p-3 text-center" style={{ backgroundColor: '#38423B' }}>
                  <p style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className="text-[14px] font-['Roboto_Serif',sans-serif] text-white/60">Chars</p>
                  <p style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className="text-[24px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white">{review.details.char_count}</p>
                </div>
              )}
              {(review.details as any).llm_model && (
                <div className="flex-1 rounded-[12px] p-3 text-center" style={{ backgroundColor: '#38423B' }}>
                  <p style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className="text-[14px] font-['Roboto_Serif',sans-serif] text-white/60">LLM</p>
                  <p style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className="text-[16px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white whitespace-nowrap overflow-hidden text-ellipsis">{(review.details as any).llm_model.toUpperCase()}</p>
                </div>
              )}
            </div>
          )}

          {/* Grammar errors flag */}
          {review.grammar_errors !== null && review.grammar_errors !== undefined && (
            <div className="w-full rounded-[12px] p-3" style={{ backgroundColor: review.grammar_errors ? "rgba(239, 68, 68, 0.25)" : "rgba(34, 197, 94, 0.25)" }}>
              <p style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className="text-[16px] font-['Roboto_Serif',sans-serif] text-white">
                {review.grammar_errors ? "Grammatical errors detected" : "No grammatical errors"}
              </p>
            </div>
          )}

          {/* Issues list */}
          {issues.length > 0 && (
            <div className="w-full space-y-2">
              <p style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className="text-[18px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white mb-2">
                Highlights
              </p>
              {Array.from(new Map(issues.map(issue => [issue.message, issue])).values()).map((issue, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-[12px]"
                  style={{
                    backgroundColor:
                      issue.type === "error" ? "rgba(239, 68, 68, 0.25)"
                      : issue.type === "improve" ? "rgba(251, 191, 36, 0.25)"
                      : "rgba(34, 197, 94, 0.25)",
                  }}
                >
                  <p style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className="text-[16px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white mb-1">
                    {issue.type === "error" ? "Error" : issue.type === "improve" ? "To improve" : "Suggestion"}
                  </p>
                  <p style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className="text-[14px] font-['Roboto_Serif',sans-serif] text-white/80">
                    {issue.message}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
