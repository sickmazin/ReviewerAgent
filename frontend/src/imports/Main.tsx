import clsx from "clsx";
import imgMain from "figma:asset/ff542d4eca6034b6a943cc359b3e290e3319857e.png";
import imgIcon from "figma:asset/276fe0b420fca2b573b6b1797b4dab39a6f69afc.png";
type Text1Props = {
  text: string;
  additionalClassNames?: string;
};

function Text1({ text, additionalClassNames = "" }: Text1Props) {
  return (
    <div style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className={clsx("flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] relative shrink-0 text-black text-center", additionalClassNames)}>
      <p className="leading-[normal]">{text}</p>
    </div>
  );
}
type TextProps = {
  text: string;
  additionalClassNames?: string;
};

function Text({ text, additionalClassNames = "" }: TextProps) {
  return (
    <div style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }} className={clsx("flex flex-[1_0_0] flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-h-px min-w-px overflow-hidden relative text-black text-ellipsis whitespace-nowrap", additionalClassNames)}>
      <p className="leading-[normal] overflow-hidden text-ellipsis">{text}</p>
    </div>
  );
}

export default function Main() {
  return (
    <div className="content-stretch flex gap-[20px] items-center justify-end px-[25px] py-[20px] relative size-full" data-name="Main">
      <img alt="" className="absolute backdrop-blur-[2000px] inset-0 max-w-none object-cover opacity-20 pointer-events-none size-full" src={imgMain} />
      <div className="bg-[#8f8f8f] h-full relative rounded-[16px] shrink-0 w-[281px]" data-name="Review">
        <div className="overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex gap-[15px] items-start p-[15px] relative size-full">
            <div className="bg-[#747474] h-full relative rounded-[24px] shrink-0 w-[250px]" data-name="chats">
              <div className="flex flex-col items-center overflow-clip rounded-[inherit] size-full">
                <div className="content-stretch flex flex-col gap-[10px] items-center p-[10px] relative size-full">
                  <Text1 text="History" additionalClassNames="min-w-full text-[38px] w-[min-content]" />
                  <div className="bg-[#a7a6a6] content-stretch flex gap-[5px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-[240px]" data-name="new_chat_element">
                    <div className="relative shrink-0 size-[45px]" data-name="icon">
                      <img alt="" className="absolute inset-0 max-w-none object-contain pointer-events-none size-full" src={imgIcon} />
                    </div>
                    <Text text="New reviews" additionalClassNames="text-[27px]" />
                  </div>
                  <div className="flex-[1_0_0] min-h-px min-w-px relative w-[250px]" data-name="chat_list">
                    <div className="flex flex-col items-center size-full">
                      <div className="content-stretch flex flex-col items-center p-[10px] relative size-full">
                        <div className="bg-white content-stretch flex gap-[20px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-[230px]" data-name="chat_element">
                          <Text text="Chat title...." additionalClassNames="text-[42px]" />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="bg-[#797979] flex-[1_0_0] h-full min-h-px min-w-px relative rounded-[24px]" data-name="main_page">
        <div className="flex flex-col items-center overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex flex-col gap-[15px] items-center p-[15px] relative size-full">
            <Text1 text="Reviewer Agent" additionalClassNames="text-[50px] w-full" />
            <div className="bg-[#a7a6a6] flex-[1_0_0] min-h-px min-w-px relative rounded-[16px] w-full" data-name="ai_output">
              <div className="size-full" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}