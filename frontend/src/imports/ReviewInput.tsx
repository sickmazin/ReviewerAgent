import clsx from "clsx";

function Wrapper({ children }: React.PropsWithChildren<{}>) {
  return (
    <div className="bg-white h-[80px] relative rounded-[16px] shrink-0 w-full">
      <div className="flex flex-row items-center overflow-clip rounded-[inherit] size-full">
        <div className="content-stretch flex gap-[10px] items-center p-[10px] relative size-full">{children}</div>
      </div>
    </div>
  );
}

function Checkbox() {
  return (
    <div className="relative rounded-[8px] shrink-0 size-[65px]">
      <div aria-hidden="true" className="absolute border-[0.5px] border-black border-solid inset-0 pointer-events-none rounded-[8px]" />
    </div>
  );
}
type TextProps = {
  text: string;
  additionalClassNames?: string;
};

function Text({ text, additionalClassNames = "" }: TextProps) {
  return (
    <div style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className={clsx("flex flex-[1_0_0] flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-h-px min-w-px relative text-black", additionalClassNames)}>
      <p className="leading-[normal]">{text}</p>
    </div>
  );
}

export default function ReviewInput() {
  return (
    <div className="bg-[#797979] content-stretch flex flex-col gap-[15px] items-center overflow-clip p-[15px] relative rounded-[24px] size-full" data-name="review_input">
      <div className="flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-w-full relative shrink-0 text-[50px] text-black w-[min-content]" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>
        <p className="leading-[normal]">Put review here</p>
      </div>
      <div className="bg-white h-[460px] relative rounded-[16px] shrink-0 w-full" data-name="review_text_input">
        <div className="flex flex-col items-center justify-center overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex flex-col items-center justify-center p-[5px] relative size-full">
            <Text text="Review..." additionalClassNames="text-[50px] w-full" />
          </div>
        </div>
      </div>
      <div className="bg-[#d80000] content-stretch flex flex-col h-[80px] items-center justify-center overflow-clip p-[5px] relative rounded-[26px] shrink-0 w-[300px]" data-name="score_button">
        <Text text="Evaluate score" additionalClassNames="text-[40px] text-center w-full" />
      </div>
      <div className="bg-[#a7a6a6] flex-[1_0_0] min-h-px min-w-px relative rounded-[16px] w-full" data-name="ai_output">
        <div className="overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex gap-[15px] items-start p-[15px] relative size-full">
            <div className="bg-[#747474] flex-[1_0_0] h-full min-h-px min-w-px relative rounded-[24px]" data-name="score">
              <div className="flex flex-col items-center overflow-clip rounded-[inherit] size-full">
                <div className="content-stretch flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium gap-[10px] items-center leading-[0] p-[10px] relative size-full text-center text-white">
                  <div className="flex flex-col justify-center relative shrink-0 text-[45px] w-full" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>
                    <p className="leading-[normal]">Insightfulness Score</p>
                  </div>
                  <div className="flex flex-[1_0_0] flex-col justify-center min-h-px min-w-px relative text-[100px] w-full" style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}>
                    <p className="leading-[normal]">65/100 😉</p>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-[#747474] h-full relative rounded-[24px] shrink-0 w-[517.5px]" data-name="others">
              <div className="flex flex-col items-center justify-center overflow-clip rounded-[inherit] size-full">
                <div className="content-stretch flex flex-col gap-[20px] items-center justify-center p-[20px] relative size-full">
                  <Wrapper>
                    <Checkbox />
                    <Text text="Follow guide linees" additionalClassNames="h-[70px] text-[40px]" />
                  </Wrapper>
                  <Wrapper>
                    <Checkbox />
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