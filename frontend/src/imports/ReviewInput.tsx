import clsx from "clsx";
import { useState } from "react";
import { api, Review } from "../app/api/client";
import LoadingSpinner from "../app/components/LoadingSpinner";

function Wrapper({ children }: React.PropsWithChildren<{}>) {
  return (
    <div className="bg-white h-[80px] relative rounded-[16px] shrink-0 w-full">
      <div className="flex flex-row items-center overflow-clip rounded-[inherit] size-full">
        <div className="content-stretch flex gap-[10px] items-center p-[10px] relative size-full">{children}</div>
      </div>
    </div>
  );
}

function Checkbox({ checked }: { checked?: boolean }) {
  return (
    <div className={clsx("relative rounded-[8px] shrink-0 size-[65px] flex items-center justify-center", checked ? "bg-green-500" : "bg-white")}>
      <div aria-hidden="true" className="absolute border-[0.5px] border-black border-solid inset-0 pointer-events-none rounded-[8px]" />
      {checked && (
        <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path>
        </svg>
      )}
    </div>
  );
}
type TextProps = {
  text: string;
  additionalClassNames?: string;
};

function Text({ text, additionalClassNames = "", color = "text-black" }: TextProps & { color?: string }) {
  return (
    <div style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className={clsx("flex flex-[1_0_0] flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-h-px min-w-px relative", color, additionalClassNames)}>
      <p className="leading-[normal]">{text}</p>
    </div>
  );
}

export default function ReviewInput() {
  const [reviewText, setReviewText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<Review | null>(null);

  const handleEvaluate = async () => {
    const trimmedText = reviewText.trim();
    if (!trimmedText) return;
    
    setIsLoading(true);
    try {
      console.log("Calling evaluate with:", trimmedText);
      const evaluation = await api.evaluate({
        chat_id: 0,
        text: trimmedText,
        category: "amazon", // default for standalone view
        rating: 5
      });
      console.log("Evaluation result:", evaluation);
      setResult(evaluation);
    } catch (error) {
      console.error("Evaluation failed:", error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-[#797979] flex flex-col items-center justify-center overflow-clip p-[15px] relative rounded-[24px] size-full min-h-[600px]" data-name="review_input_loading">
        <LoadingSpinner />
        <p className="text-white mt-4 font-['Roboto_Serif:Medium',sans-serif]">Analisi in corso...</p>
      </div>
    );
  }

  const canEvaluate = reviewText.trim().length > 0;

  return (
    <div className="bg-[#797979] content-stretch flex flex-col gap-[15px] items-center overflow-clip p-[15px] relative rounded-[24px] size-full" data-name="review_input">
      <div className="flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-w-full relative shrink-0 text-[50px] text-black w-[min-content]" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>
        <p className="leading-[normal]">Put review here</p>
      </div>
      <div className="bg-white h-[460px] relative rounded-[16px] shrink-0 w-full" data-name="review_text_input">
        <div className="flex flex-col items-center justify-center overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex flex-col items-center justify-center p-[5px] relative size-full">
            <textarea
              className="w-full h-full p-4 text-[32px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-black placeholder:text-gray-400 outline-none resize-none bg-transparent"
              placeholder="Review..."
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            />
          </div>
        </div>
      </div>
      <button
        onClick={handleEvaluate}
        disabled={!canEvaluate}
        className={clsx(
          "bg-[#d80000] content-stretch flex flex-col h-[80px] items-center justify-center overflow-clip p-[5px] relative rounded-[26px] shrink-0 w-[300px] transition-all duration-200",
          !canEvaluate ? "opacity-50 cursor-not-allowed grayscale" : "hover:bg-[#ff0000] hover:scale-105 active:scale-95 cursor-pointer"
        )}
        data-name="score_button"
      >
        <Text text="Evaluate score" color="text-white" additionalClassNames="text-[40px] text-center w-full" />
      </button>
      <div className="bg-[#a7a6a6] flex-[1_0_0] min-h-[250px] min-w-px relative rounded-[16px] w-full" data-name="ai_output">
        <div className="overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex gap-[15px] items-start p-[15px] relative size-full">
            <div className="bg-[#747474] flex-[1_0_0] h-full min-h-px min-w-px relative rounded-[24px]" data-name="score">
              <div className="flex flex-col items-center overflow-clip rounded-[inherit] size-full">
                <div className="content-stretch flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium gap-[10px] items-center leading-[0] p-[10px] relative size-full text-center text-white">
                  <div className="flex flex-col justify-center relative shrink-0 text-[45px] w-full" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>
                    <p className="leading-[normal]">Insightfulness Score</p>
                  </div>
                  <div className="flex flex-[1_0_0] flex-col justify-center min-h-px min-w-px relative text-[100px] w-full" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>
                    <p className="leading-[normal]">
                      {result ? (
                        (() => {
                          if (result.score === null) return "--/100";
                          const num = Number(result.score);
                          if (isNaN(num)) return `${result.score}/100 ❓`;
                          let emoji = "🤩";
                          if (num <= 36) emoji = "😢";
                          else if (num <= 70) emoji = "😊";
                          return `${num}/100 ${emoji}`;
                        })()
                      ) : (
                        "--/100"
                      )}
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-[#747474] h-full relative rounded-[24px] shrink-0 w-[517.5px]" data-name="others">
              <div className="flex flex-col items-center justify-center overflow-clip rounded-[inherit] size-full">
                <div className="content-stretch flex flex-col gap-[20px] items-center justify-center p-[20px] relative size-full">
                  <Wrapper>
                    <Checkbox checked={result?.follow_guidelines ?? false} />
                    <Text text="Follow guide linees" additionalClassNames="h-[70px] text-[40px]" />
                  </Wrapper>
                  <Wrapper>
                    <Checkbox checked={!(result?.grammar_errors ?? true)} />
                    <Text text="Grammatical errors" additionalClassNames="h-[70px] text-[40px]" />
                  </Wrapper>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}