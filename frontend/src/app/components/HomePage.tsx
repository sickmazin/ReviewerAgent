import clsx from "clsx";
import imgMain from "figma:asset/ff542d4eca6034b6a943cc359b3e290e3319857e.png";
import imgIcon from "figma:asset/276fe0b420fca2b573b6b1797b4dab39a6f69afc.png";
import { ChatItem } from "./ChatItem";
import { Sparkles, Brain, BarChart3, Info } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { api, Chat, ModelInfo } from "../api/client";

type Text1Props = {
  text: string;
  additionalClassNames?: string;
};

function Text1({ text, additionalClassNames = "" }: Text1Props) {
  return (
    <div
      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
      className={clsx(
        "flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] relative shrink-0 text-white text-center",
        additionalClassNames
      )}
    >
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
    <div
      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
      className={clsx(
        "flex flex-[1_0_0] flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-h-px min-w-px overflow-hidden relative text-white text-ellipsis whitespace-nowrap",
        additionalClassNames
      )}
    >
      <p className="leading-[normal] overflow-hidden text-ellipsis">{text}</p>
    </div>
  );
}

export default function HomePage() {
  const navigate = useNavigate();
  const [chats, setChats] = useState<Chat[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [chatToDelete, setChatToDelete] = useState<number | null>(null);

  useEffect(() => {
    api.getChats().then(setChats).catch(() => setChats([]));
    api.getModelInfo().then(setModelInfo).catch(() => setModelInfo(null));
  }, []);

  const handleDeleteChat = (id: number) => setChatToDelete(id);

  const confirmDelete = async () => {
    if (chatToDelete == null) return;
    await api.deleteChat(chatToDelete);
    setChats((prev) => prev.filter((c) => c.id !== chatToDelete));
    setChatToDelete(null);
  };

  const handleNewChat = () => navigate("/review");
  const handleChatClick = (id: number) => navigate(`/review/${id}`);

  return (
    <div
      className="content-stretch flex gap-[20px] items-center justify-end px-[25px] py-[20px] relative size-full min-w-[1920px] min-h-[1080px]"
      data-name="Main"
      style={{ backgroundColor: '#2D2D2A' }}
    >
      <img
        alt=""
        className="absolute backdrop-blur-[2000px] inset-0 max-w-none object-cover opacity-10 pointer-events-none size-full"
        src={imgMain}
      />

      {/* Sidebar */}
      <div
        className="bg-[#38423B] h-full relative rounded-[16px] shrink-0 w-[281px]"
        data-name="Review"
      >
        <div className="overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex gap-[15px] items-start p-[15px] relative size-full">
            <div
              className="bg-[#353831] h-full relative rounded-[24px] shrink-0 w-[250px]"
              data-name="chats"
            >
              <div className="flex flex-col items-center overflow-clip rounded-[inherit] size-full">
                <div className="content-stretch flex flex-col gap-[10px] items-center p-[10px] relative size-full">
                  <Text1
                    text="History"
                    additionalClassNames="min-w-full text-[38px] w-[min-content]"
                  />

                  <button
                    onClick={handleNewChat}
                    className="bg-[#2D2D2A] hover:bg-[#38423B] transition-colors content-stretch flex gap-[5px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-full"
                    data-name="new_chat_element"
                  >
                    <div className="relative shrink-0 size-[45px]" data-name="icon">
                      <img
                        alt=""
                        className="absolute inset-0 max-w-none object-contain pointer-events-none size-full"
                        src={imgIcon}
                      />
                    </div>
                    <Text text="New review" additionalClassNames="text-[27px]" />
                  </button>

                  <div
                    className="flex-[1_0_0] min-h-px min-w-px relative w-full overflow-y-auto scrollbar-hide"
                    data-name="chat_list"
                  >
                    <div className="flex flex-col items-center gap-[10px] p-[10px]">
                      {chats.map((chat) => (
                        <ChatItem
                          key={chat.id}
                          title={chat.title}
                          isActive={false}
                          onClick={() => handleChatClick(chat.id)}
                          onDelete={() => handleDeleteChat(chat.id)}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div
        className="bg-[#38423B] flex-[1_0_0] h-full min-h-px min-w-px relative rounded-[24px]"
        data-name="main_page"
      >
        <div className="flex flex-col items-center overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex flex-col gap-[20px] items-center p-[30px] relative size-full overflow-y-auto">
            <Text1
              text="Reviewer Agent"
              additionalClassNames="text-[50px] w-full"
            />

            {/* Welcome Section */}
            <div className="bg-[#353831] w-full rounded-[16px] p-[25px]">
              <div className="flex items-start gap-4">
                <div className="shrink-0 bg-[#2D2D2A] rounded-full p-3">
                  <Sparkles className="size-8 text-white" />
                </div>
                <div>
                  <h2
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[32px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white mb-3"
                  >
                    {modelInfo?.welcome_title ?? ""}
                  </h2>
                  <p
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[18px] font-['Roboto_Serif',sans-serif] text-white/90 leading-relaxed"
                  >
                    {modelInfo?.welcome_description ?? ""}
                  </p>
                </div>
              </div>
            </div>

            {/* How It Works Section */}
            <div className="bg-[#353831] w-full rounded-[16px] p-[25px]">
              <div className="flex items-start gap-4">
                <div className="shrink-0 bg-[#2D2D2A] rounded-full p-3">
                  <Brain className="size-8 text-white" />
                </div>
                <div className="flex-1">
                  <h3
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[28px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white mb-3"
                  >
                    How It Works
                  </h3>
                  <div className="space-y-3">
                    {(modelInfo?.how_it_works ?? []).map((step, idx) => (
                      <p
                        key={idx}
                        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                        className="text-[16px] font-['Roboto_Serif',sans-serif] text-white/90 leading-relaxed"
                      >
                        {step}
                      </p>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Model Info Section */}
            <div className="bg-[#353831] w-full rounded-[16px] p-[25px]">
              <div className="flex items-start gap-4">
                <div className="shrink-0 bg-[#2D2D2A] rounded-full p-3">
                  <Info className="size-8 text-white" />
                </div>
                <div className="flex-1">
                  <h3
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[28px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white mb-3"
                  >
                    The Model
                  </h3>
                  <div className="space-y-3">
                    {(modelInfo?.model_details ?? []).map((d, idx) => (
                      <p
                        key={idx}
                        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                        className="text-[16px] font-['Roboto_Serif',sans-serif] text-white/90 leading-relaxed"
                      >
                        <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">{d.label}:</span>{" "}
                        {d.value}
                      </p>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Insightfulness Score Section */}
            <div className="bg-[#353831] w-full rounded-[16px] p-[25px]">
              <div className="flex items-start gap-4">
                <div className="shrink-0 bg-[#2D2D2A] rounded-full p-3">
                  <BarChart3 className="size-8 text-white" />
                </div>
                <div className="flex-1">
                  <h3
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[28px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white mb-3"
                  >
                    Insightfulness Score
                  </h3>
                  <ul className="space-y-2 ml-4 mb-3">
                    {(modelInfo?.score_dimensions ?? []).map((d, idx) => (
                      <li
                        key={idx}
                        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                        className="text-[16px] font-['Roboto_Serif',sans-serif] text-white/90"
                      >
                        • <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">{d.label}:</span>{" "}
                        {d.description}
                      </li>
                    ))}
                  </ul>
                  <ul className="space-y-2 ml-4 mb-3">
                    {(modelInfo?.score_categories ?? []).map((c, idx) => (
                      <li
                        key={idx}
                        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                        className="text-[16px] font-['Roboto_Serif',sans-serif] text-white/90"
                      >
                        • <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">{c.label}:</span>{" "}
                        {c.description}
                      </li>
                    ))}
                  </ul>
                  <p
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[16px] font-['Roboto_Serif',sans-serif] text-white/90 leading-relaxed"
                  >
                    {modelInfo?.metrics_description ?? ""}
                  </p>
                </div>
              </div>
            </div>

            <button
              onClick={handleNewChat}
              className="bg-[#2D2D2A] hover:bg-[#38423B] transition-colors w-full rounded-[16px] p-[20px] mt-4"
            >
              <p
                style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                className="text-[28px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white"
              >
                Start a New Review →
              </p>
            </button>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Popup */}
      {chatToDelete !== null && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-[100000]">
          <div className="rounded-[24px] p-8 max-w-md w-full mx-4 shadow-2xl" style={{ backgroundColor: '#38423B' }}>
            <h2
              className="text-[32px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white mb-4 text-center"
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            >
              Confirm deletion
            </h2>
            <p
              className="text-[20px] font-['Roboto_Serif',sans-serif] text-white mb-6 text-center"
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            >
              Are you sure you want to delete this chat? This action cannot be undone.
            </p>
            <div className="flex gap-4 justify-center">
              <button
                onClick={() => setChatToDelete(null)}
                className="transition-colors px-8 py-3 rounded-[16px] text-[20px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white"
                style={{ backgroundColor: '#353831', fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="transition-colors px-8 py-3 rounded-[16px] text-[20px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white"
                style={{ backgroundColor: '#8B0000', fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
