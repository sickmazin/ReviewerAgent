import svgPaths from "../../imports/svg-n82fbntdca";
import imgEbay from "figma:asset/bd8be4b28f0ef4550914f3327e0679a1dba59540.png";
import { useState } from "react";

function Wrapper({ children }: React.PropsWithChildren<{}>) {
  return (
    <div className="relative shrink-0 size-[65px]">
      <svg className="absolute block size-full" fill="none" preserveAspectRatio="none" viewBox="0 0 65 65">
        {children}
      </svg>
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
      className="relative rounded-[8px] shrink-0 size-[65px] transition-colors cursor-pointer"
      style={{ backgroundColor: checked ? '#2D2D2A' : '#353831' }}
    >
      <div
        aria-hidden="true"
        className="absolute border-[0.5px] border-white border-solid inset-0 pointer-events-none rounded-[8px]"
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

interface WebsiteItemProps {
  icon: React.ReactNode;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

function WebsiteItem({ icon, label, checked, onChange }: WebsiteItemProps) {
  return (
    <div
      className="content-stretch flex gap-[10px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-[400px]"
      style={{ backgroundColor: '#38423B' }}
      data-name={`website-${label.toLowerCase()}`}
    >
      <Checkbox checked={checked} onChange={onChange} />
      {icon}
      <div
        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
        className="flex flex-[1_0_0] flex-col font-['Roboto_Serif',sans-serif] h-[70px] justify-center leading-[0] min-h-px min-w-px overflow-hidden relative text-[50px] text-white text-ellipsis whitespace-nowrap"
      >
        <p className="leading-[normal] overflow-hidden text-ellipsis">{label}</p>
      </div>
    </div>
  );
}

export default function LinkSitePanel() {
  const [amazonChecked, setAmazonChecked] = useState(false);
  const [ebayChecked, setEbayChecked] = useState(false);
  const [restaurantsChecked, setRestaurantsChecked] = useState(false);
  const [locationChecked, setLocationChecked] = useState(false);
  const [url, setUrl] = useState("");

  return (
    <div
      className="content-stretch flex flex-col gap-[20px] items-center overflow-y-auto overflow-x-hidden p-[10px] relative rounded-[24px] size-full scrollbar-hide"
      style={{ backgroundColor: '#353831' }}
      data-name="link_site_bar"
    >
      <div
        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
        className="flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] relative shrink-0 text-[42px] text-white text-center whitespace-nowrap"
      >
        <p className="leading-[normal] text-[32px]">Where you post this review?</p>
      </div>

      <WebsiteItem
        icon={
          <Wrapper>
            <g id="amazon">
              <path d={svgPaths.pd65a80} fill="white" id="Vector" />
            </g>
          </Wrapper>
        }
        label="Amazon"
        checked={amazonChecked}
        onChange={setAmazonChecked}
      />

      <WebsiteItem
        icon={
          <div className="relative shrink-0 size-[65px]" data-name="Ebay">
            <img
              alt=""
              className="absolute inset-0 max-w-none object-contain pointer-events-none size-full"
              style={{ filter: 'brightness(0) invert(1)' }}
              src={imgEbay}
            />
          </div>
        }
        label="Ebay"
        checked={ebayChecked}
        onChange={setEbayChecked}
      />

      <WebsiteItem
        icon={
          <Wrapper>
            <g id="Fork">
              <path d={svgPaths.p28c99e00} fill="white" id="Vector" />
            </g>
          </Wrapper>
        }
        label="Restaurants"
        checked={restaurantsChecked}
        onChange={setRestaurantsChecked}
      />

      <WebsiteItem
        icon={
          <Wrapper>
            <g id="Location Pin">
              <path d={svgPaths.p1bc87600} fill="white" id="Vector" />
            </g>
          </Wrapper>
        }
        label="Location"
        checked={locationChecked}
        onChange={setLocationChecked}
      />

      <div className="h-[60px] shrink-0 w-[358px]" data-name="space" />

      <div
        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
        className="flex flex-col font-['Roboto_Serif',sans-serif] justify-center leading-[0] relative shrink-0 text-[42px] text-white text-center whitespace-nowrap"
      >
        <p className="leading-[normal] text-[35px]">Link of product in review:</p>
      </div>

      <div
        className="content-stretch flex h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-[400px]"
        style={{ backgroundColor: '#38423B' }}
        data-name="link_area"
      >
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Url (optional)..."
          className="flex-1 h-[70px] bg-transparent font-['Roboto_Serif',sans-serif] text-[42px] text-white outline-none placeholder:text-white/40"
        />
      </div>

      <div
        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
        className="flex flex-col font-['Roboto_Serif',sans-serif] justify-center leading-[0] relative shrink-0 text-[42px] text-white/60 text-center whitespace-nowrap"
      >
        
      </div>
    </div>
  );
}