import svgPaths from "./svg-n82fbntdca";
import imgEbay from "figma:asset/bd8be4b28f0ef4550914f3327e0679a1dba59540.png";

function Wrapper({ children }: React.PropsWithChildren<{}>) {
  return (
    <div className="relative shrink-0 size-[65px]">
      <svg className="absolute block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 65 65">
        {children}
      </svg>
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

export default function LinkSiteBar() {
  return (
    <div className="bg-[#747474] content-stretch flex flex-col gap-[20px] items-center overflow-clip p-[10px] relative rounded-[24px] size-full" data-name="link_site_bar">
      <div className="flex flex-col font-['Instrument_Serif:Regular',sans-serif] justify-center leading-[0] not-italic relative shrink-0 text-[42px] text-black text-center whitespace-nowrap">
        <p className="leading-[normal]">Where you post this review?</p>
      </div>
      <div className="bg-white content-stretch flex gap-[10px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-[400px]" data-name="website1">
        <Checkbox />
        <Wrapper>
          <g id="amazon">
            <path d={svgPaths.pd65a80} fill="var(--fill-0, black)" id="Vector" />
          </g>
        </Wrapper>
        <div className="flex flex-[1_0_0] flex-col font-['Instrument_Serif:Regular',sans-serif] h-[70px] justify-center leading-[0] min-h-px min-w-px not-italic overflow-hidden relative text-[50px] text-black text-ellipsis whitespace-nowrap">
          <p className="leading-[normal] overflow-hidden text-ellipsis">Amazon</p>
        </div>
      </div>
      <div className="bg-white content-stretch flex gap-[10px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-[400px]" data-name="website2">
        <Checkbox />
        <div className="relative shrink-0 size-[65px]" data-name="Ebay">
          <img alt="" className="absolute inset-0 max-w-none object-contain pointer-events-none size-full" src={imgEbay} />
        </div>
        <div className="flex flex-[1_0_0] flex-col font-['Instrument_Serif:Regular',sans-serif] h-[70px] justify-center leading-[0] min-h-px min-w-px not-italic relative text-[50px] text-black">
          <p className="leading-[normal]">Ebay</p>
        </div>
      </div>
      <div className="bg-white content-stretch flex gap-[10px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-[400px]" data-name="website3">
        <Checkbox />
        <Wrapper>
          <g id="Fork">
            <path d={svgPaths.p28c99e00} fill="var(--fill-0, black)" id="Vector" />
          </g>
        </Wrapper>
        <div className="flex flex-[1_0_0] flex-col font-['Instrument_Serif:Regular',sans-serif] h-[70px] justify-center leading-[0] min-h-px min-w-px not-italic relative text-[50px] text-black">
          <p className="leading-[normal]">Restaurants</p>
        </div>
      </div>
      <div className="bg-white content-stretch flex gap-[10px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-[400px]" data-name="website4">
        <Checkbox />
        <Wrapper>
          <g id="Location Pin">
            <path d={svgPaths.p1bc87600} fill="var(--fill-0, black)" id="Vector" />
          </g>
        </Wrapper>
        <div className="flex flex-[1_0_0] flex-col font-['Instrument_Serif:Regular',sans-serif] h-[70px] justify-center leading-[0] min-h-px min-w-px not-italic relative text-[50px] text-black">
          <p className="leading-[normal]">Location</p>
        </div>
      </div>
      <div className="h-[60px] shrink-0 w-[358px]" data-name="space" />
      <div className="flex flex-col font-['Instrument_Serif:Regular',sans-serif] justify-center leading-[0] not-italic relative shrink-0 text-[42px] text-black text-center whitespace-nowrap">
        <p className="leading-[normal]">Link of product in review:</p>
      </div>
      <div className="bg-white content-stretch flex h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-[400px]" data-name="link_area">
        <div className="flex flex-[1_0_0] flex-col font-['Instrument_Serif:Regular',sans-serif] h-[70px] justify-center leading-[0] min-h-px min-w-px not-italic relative text-[42px] text-black">
          <p className="leading-[normal]">{`Url (optional)... `}</p>
        </div>
      </div>
      <div className="flex flex-col font-['Instrument_Serif:Regular',sans-serif] justify-center leading-[0] not-italic relative shrink-0 text-[42px] text-black text-center whitespace-nowrap">
        <p className="leading-[normal]">(Optional)</p>
      </div>
    </div>
  );
}