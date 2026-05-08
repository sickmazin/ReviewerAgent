import { useEffect, useState } from "react";
import { api, Site } from "../api/client";
import imgAirbnb from "../../assets/Airbnb.png";
import imgAmazon from "../../assets/amazon.png";
import imgBooking from "../../assets/Booking.png";
import imgEbay from "../../assets/Ebay.png";
import imgGoogle from "../../assets/Google.png";

const siteIcons: Record<string, string> = {
  "airbnb": imgAirbnb,
  "amazon": imgAmazon,
  "booking": imgBooking,
  "ebay": imgEbay,
  "google": imgGoogle,
};

interface CheckboxProps {
  checked: boolean;
  onChange: () => void;
}

function Checkbox({ checked, onChange }: CheckboxProps) {
  return (
    <button
      onClick={onChange}
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
            <path d="M5 13L9 17L19 7" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      )}
    </button>
  );
}

interface WebsiteItemProps {
  label: string;
  checked: boolean;
  onChange: () => void;
  icon?: string;
}

function WebsiteItem({ label, checked, onChange, icon }: WebsiteItemProps) {
  return (
    <div
      className="content-stretch flex gap-[10px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-[400px]"
      style={{ backgroundColor: '#38423B' }}
      data-name={`website-${label.toLowerCase()}`}
    >
      <Checkbox checked={checked} onChange={onChange} />
      {icon && (
        <div className="relative shrink-0 size-[50px] mx-[5px]">
          <img src={icon} alt={`${label} logo`} className="absolute inset-0 size-full object-contain pointer-events-none" />
        </div>
      )}
      <div
        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
        className="flex flex-[1_0_0] flex-col font-['Roboto_Serif',sans-serif] h-[70px] justify-center leading-[0] min-h-px min-w-px overflow-hidden relative text-[34px] text-white text-ellipsis whitespace-nowrap"
      >
        <p className="leading-[normal] overflow-hidden text-ellipsis">{label}</p>
      </div>
    </div>
  );
}

interface LinkSitePanelProps {
  selectedSite: string | null;
  onSiteChange: (siteId: string | null) => void;
  url: string;
  onUrlChange: (url: string) => void;
  selectedModel: string;
  onModelChange: (modelId: string) => void;
}

export default function LinkSitePanel({
  selectedSite,
  onSiteChange,
  url,
  onUrlChange,
  selectedModel,
  onModelChange,
}: LinkSitePanelProps) {
  const [sites, setSites] = useState<Site[]>([]);

  useEffect(() => {
    api.getSites().then(setSites).catch(() => setSites([]));
  }, []);

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
        <p className="leading-[normal] text-[32px]">Choose LLM Model</p>
      </div>

      <div className="flex gap-[20px] w-full justify-center">
        <div className="flex items-center gap-[10px] cursor-pointer" onClick={() => onModelChange("gemma3:27b")}>
          <Checkbox checked={selectedModel === "gemma3:27b"} onChange={() => onModelChange("gemma3:27b")} />
          <span className="text-white text-[24px] font-['Roboto_Serif:Medium',sans-serif]">Gemma 3 (27b)</span>
        </div>
        <div className="flex items-center gap-[10px] cursor-pointer" onClick={() => onModelChange("gemma4:26b")}>
          <Checkbox checked={selectedModel === "gemma4:26b"} onChange={() => onModelChange("gemma4:26b")} />
          <span className="text-white text-[24px] font-['Roboto_Serif:Medium',sans-serif]">Gemma 4 (26b)</span>
        </div>
      </div>

      <div className="w-[80%] h-[1px] bg-white/20 my-2"></div>

      <div
        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
        className="flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] relative shrink-0 text-[42px] text-white text-center whitespace-nowrap"
      >
        <p className="leading-[normal] text-[32px]">Where you post this review?</p>
      </div>

      {sites.map((s) => (
        <WebsiteItem
          key={s.id}
          label={s.label}
          checked={selectedSite === s.id}
          onChange={() => onSiteChange(selectedSite === s.id ? null : s.id)}
          icon={siteIcons[s.id.toLowerCase()]}
        />
      ))}

    </div>
  );
}
