import clsx from "clsx";
import imgMain from "figma:asset/ff542d4eca6034b6a943cc359b3e290e3319857e.png";
import imgIcon from "../../assets/write.png";
import { ChatItem } from "./ChatItem";
import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router";
import ReviewInputPanel from "./ReviewInputPanel";
import LinkSitePanel from "./LinkSitePanel";
import ReviewResultPanel from "./ReviewResultPanel";
import { Home } from "lucide-react";
import { api, Chat, Review } from "../api/client";

type Text1Props = { text: string; additionalClassNames?: string };
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

type TextProps = { text: string; additionalClassNames?: string };
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

export default function ReviewPage() {
  const navigate = useNavigate();
  const { chatId } = useParams<{ chatId: string }>();
  const activeChatId = chatId || null;

  const [chats, setChats] = useState<Chat[]>([]);
  const [currentReview, setCurrentReview] = useState<Review | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Input state (lifted from child panels)
  const [reviewText, setReviewText] = useState("");
  const [site, setSite] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const [selectedModel, setSelectedModel] = useState("gemma3:27b");

  const [chatToDelete, setChatToDelete] = useState<string | null>(null);

  const refreshChats = () => api.getChats().then(setChats).catch(() => {});

  useEffect(() => {
    refreshChats();
  }, []);

  // Load latest review for selected chat
  useEffect(() => {
    if (activeChatId == null) {
      setCurrentReview(null);
      return;
    }
    api
      .getChatReviews(activeChatId)
      .then((reviews) => setCurrentReview(reviews[reviews.length - 1] ?? null))
      .catch(() => setCurrentReview(null));
  }, [activeChatId]);

  const handleEvaluate = async () => {
    if (!reviewText.trim() || !site) return;
    setIsLoading(true);
    try {
      const review = await api.evaluate({
        chat_id: activeChatId ?? "0",
        text: reviewText,
        category: site,
        rating: 5,
        model: selectedModel,
      });
      await refreshChats();
      setCurrentReview(review);
      if (review.chat_id !== activeChatId) {
        navigate(`/review/${review.chat_id}`);
      }
    } catch (error) {
      console.error("Evaluation failed:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteChat = (id: string) => setChatToDelete(id);

  const confirmDelete = async () => {
    if (chatToDelete == null) return;
    await api.deleteChat(chatToDelete);
    setChats((prev) => prev.filter((c) => c.id !== chatToDelete));
    if (activeChatId === chatToDelete) {
      navigate("/review");
    }
    setChatToDelete(null);
  };

  const handleNewChat = () => {
    setCurrentReview(null);
    setReviewText("");
    setSite(null);
    setUrl("");
    navigate("/review");
  };

  const handleChatClick = (id: string) => {
    setReviewText("");
    setSite(null);
    setUrl("");
    navigate(`/review/${id}`);
  };

  const showResult = currentReview !== null;

  return (
    <div
      className="content-stretch flex gap-[20px] items-center justify-end px-[25px] py-[20px] relative size-full min-w-[1920px] min-h-[1080px]"
      data-name="ReviewPage"
      style={{ backgroundColor: '#2D2D2A' }}
    >
      <img
        alt=""
        className="absolute backdrop-blur-[2000px] inset-0 max-w-none object-cover opacity-10 pointer-events-none size-full"
        src={imgMain}
      />

      {/* Sidebar */}
      <div
        className="h-full relative rounded-[16px] shrink-0 w-[281px]"
        style={{ backgroundColor: '#353831' }}
        data-name="Review"
      >
        <div className="overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex gap-[15px] items-start p-[15px] relative size-full">
            <div
              className="h-full relative rounded-[24px] shrink-0 w-[250px]"
              style={{ backgroundColor: '#38423B' }}
              data-name="chats"
            >
              <div className="flex flex-col items-center overflow-clip rounded-[inherit] size-full">
                <div className="content-stretch flex flex-col gap-[10px] items-center p-[10px] relative size-full">
                  <div className="flex items-center justify-between w-full px-2">
                    <Text1 text="History" additionalClassNames="text-[38px]" />
                    <button
                      onClick={() => navigate("/")}
                      className="transition-colors rounded-[8px] p-2"
                      style={{ backgroundColor: 'transparent' }}
                      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#454f48'}
                      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                      title="Go to Home"
                    >
                      <Home size={32} color="#FFFFFF" strokeWidth={2} />
                    </button>
                  </div>

                  <button
                    onClick={handleNewChat}
                    className="content-stretch flex gap-[5px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-full transition-colors"
                    style={{ backgroundColor: '#353831' }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#3d4739'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#353831'}
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
                          isActive={chat.id === activeChatId}
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

      {!showResult ? (
        <>
          <div className="flex-[1_0_0] h-full min-h-px min-w-px relative">
            <ReviewInputPanel
              value={reviewText}
              onChange={setReviewText}
              onSubmit={handleEvaluate}
              isLoading={isLoading}
              canSubmit={!!reviewText.trim() && !!site}
            />
          </div>

          <div className="h-full relative shrink-0 w-[450px]">
            <LinkSitePanel
              selectedSite={site}
              onSiteChange={setSite}
              url={url}
              onUrlChange={setUrl}
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
            />
          </div>
        </>
      ) : (
        currentReview && (
          <ReviewResultPanel
            review={currentReview}
            onBack={() => navigate("/")}
          />
        )
      )}

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
