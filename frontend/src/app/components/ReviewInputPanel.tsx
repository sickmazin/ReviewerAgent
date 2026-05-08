import LoadingSpinner from "./LoadingSpinner";

interface ReviewInputPanelProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  canSubmit: boolean;
}

export default function ReviewInputPanel({
  value,
  onChange,
  onSubmit,
  isLoading,
  canSubmit,
}: ReviewInputPanelProps) {
  if (isLoading) {
    return (
      <div
        className="content-stretch flex flex-col items-center justify-center overflow-clip p-[15px] relative rounded-[24px] size-full"
        style={{ backgroundColor: "#353831" }}
        data-name="review_loading"
      >
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div
      className="content-stretch flex flex-col gap-[15px] items-center overflow-clip p-[15px] relative rounded-[24px] size-full"
      style={{ backgroundColor: "#353831" }}
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
        style={{ backgroundColor: "#38423B" }}
        data-name="review_text_input"
      >
        <div className="flex flex-col items-center justify-center overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex flex-col items-center justify-start p-[15px] relative size-full">
            <textarea
              value={value}
              onChange={(e) => onChange(e.target.value)}
              placeholder="Review..."
              className="w-full h-full resize-none bg-transparent text-[24px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white placeholder:text-white/40 outline-none leading-relaxed"
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            />
          </div>
        </div>
      </div>

      <button
        onClick={onSubmit}
        disabled={!canSubmit}
        className="content-stretch flex flex-col h-[80px] items-center justify-center overflow-clip p-[5px] relative rounded-[26px] shrink-0 w-[300px] transition-colors"
        style={{
          backgroundColor: !canSubmit ? "#38423B" : "#8B0000",
          cursor: !canSubmit ? "not-allowed" : "pointer",
          opacity: !canSubmit ? 0.5 : 1,
        }}
        onMouseEnter={(e) => {
          if (canSubmit) e.currentTarget.style.backgroundColor = "#A00000";
        }}
        onMouseLeave={(e) => {
          if (canSubmit) e.currentTarget.style.backgroundColor = "#8B0000";
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
