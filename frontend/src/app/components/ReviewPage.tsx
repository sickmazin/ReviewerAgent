import clsx from "clsx";
import imgMain from "figma:asset/ff542d4eca6034b6a943cc359b3e290e3319857e.png";
import imgIcon from "figma:asset/276fe0b420fca2b573b6b1797b4dab39a6f69afc.png";
import { ChatItem } from "./ChatItem";
import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router";
import ReviewInputPanel from "./ReviewInputPanel";
import LinkSitePanel from "./LinkSitePanel";
import ReviewResultPanel from "./ReviewResultPanel";
import { Home } from "lucide-react";

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

export default function ReviewPage() {
  const navigate = useNavigate();
  const { chatId } = useParams<{ chatId: string }>();
  const [chats, setChats] = useState([
    { id: 1, title: "Analisi Paper ML", active: true },
    { id: 2, title: "Review Articolo AI", active: false },
    { id: 3, title: "Studio Ricerca NLP", active: false },
    { id: 4, title: "Valutazione Dataset", active: false },
  ]);

  // Mock data per le chat esistenti - da sostituire con dati dal backend
  const [chatData] = useState<Record<number, {
    reviewText: string;
    score: number;
    followGuidelines: boolean;
    hasGrammaticalErrors: boolean;
  }>>({
    1: {
      reviewText: "Questo paper presenta un approccio innovativo al machine learning. La metodologia è ben strutturata e i risultati sono convincenti. Tuttavia, manca una discussione approfondita sui limiti del modello proposto. I dataset utilizzati potrebbero essere più diversificati per validare meglio le conclusioni.",
      score: 85,
      followGuidelines: true,
      hasGrammaticalErrors: false,
    },
    2: {
      reviewText: "L'articolo sull'intelligenza artificiale offre spunti interessanti ma presenta alcune lacune metodologiche. La sezione di analisi è superficiale e mancano riferimenti bibliografici recenti. Nonostante ciò, l'idea di base è valida e meriterebbe un approfondimento.",
      score: 72,
      followGuidelines: false,
      hasGrammaticalErrors: true,
    },
    3: {
      reviewText: "Ricerca eccellente nel campo del Natural Language Processing. Gli autori dimostrano una profonda comprensione del dominio. L'implementazione tecnica è solida e ben documentata. I risultati sperimentali sono statisticamente significativi e replicabili.",
      score: 94,
      followGuidelines: true,
      hasGrammaticalErrors: false,
    },
    4: {
      reviewText: "Il dataset proposto è completo ma presenta alcuni problemi di qualità dei dati. La documentazione è scarsa e rende difficile la riproducibilità. Sarebbe utile aggiungere più esempi annotati e migliorare la descrizione delle metriche utilizzate.",
      score: 68,
      followGuidelines: true,
      hasGrammaticalErrors: false,
    },
  });

  const [isEvaluated, setIsEvaluated] = useState(false);
  const [evaluationData, setEvaluationData] = useState<{
    reviewText: string;
    score: number;
    followGuidelines: boolean;
    hasGrammaticalErrors: boolean;
  } | null>(null);

  const [chatToDelete, setChatToDelete] = useState<number | null>(null);

  // Carica i dati della chat quando viene passato un chatId nella URL
  useEffect(() => {
    if (chatId) {
      const id = parseInt(chatId, 10);
      if (!isNaN(id)) {
        // Imposta la chat come attiva
        setChats(chats.map(chat => ({ ...chat, active: chat.id === id })));
        
        // Carica i dati mock (successivamente da backend)
        const data = chatData[id];
        if (data) {
          setEvaluationData(data);
          setIsEvaluated(true);
        }
      }
    }
  }, [chatId]);

  useEffect(() => {
    // Setup callback for evaluation
    (window as any).onEvaluateScore = (data: any) => {
      setEvaluationData(data);
      setIsEvaluated(true);
    };

    return () => {
      delete (window as any).onEvaluateScore;
    };
  }, []);

  const handleDeleteChat = (id: number) => {
    setChatToDelete(id);
  };

  const confirmDelete = () => {
    if (chatToDelete !== null) {
      setChats(chats.filter((chat) => chat.id !== chatToDelete));
      setChatToDelete(null);
    }
  };

  const cancelDelete = () => {
    setChatToDelete(null);
  };

  const handleNewChat = () => {
    setIsEvaluated(false);
    setEvaluationData(null);
    // Imposta tutte le chat come non attive quando si crea una nuova chat
    setChats(chats.map(chat => ({ ...chat, active: false })));
  };

  const handleChatClick = (id: number) => {
    // Imposta la chat come attiva
    setChats(
      chats.map((chat) => ({
        ...chat,
        active: chat.id === id,
      }))
    );
    
    // Carica i dati mock della chat (successivamente da backend)
    const data = chatData[id];
    if (data) {
      setEvaluationData(data);
      setIsEvaluated(true);
    }
  };

  const handleBackToInput = () => {
    setIsEvaluated(false);
  };

  return (
    <div
      className="content-stretch flex gap-[20px] items-stretch justify-end px-[25px] py-[20px] relative size-full min-w-full h-screen overflow-hidden"
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
                  {/* Header with History title and Home icon */}
                  <div className="flex items-center justify-between w-full px-2">
                    <Text1
                      text="History"
                      additionalClassNames="text-[38px]"
                    />
                    <button
                      onClick={() => navigate("/")}
                      className="transition-colors rounded-[8px] p-2"
                      style={{ 
                        backgroundColor: 'transparent',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#454f48'}
                      onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                      title="Go to Home"
                    >
                      <Home size={32} color="#FFFFFF" strokeWidth={2} />
                    </button>
                  </div>

                  {/* New Chat Button */}
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

                  {/* Chat List with Invisible Scrollbar */}
                  <div
                    className="flex-[1_0_0] min-h-px min-w-px relative w-full overflow-y-auto scrollbar-hide"
                    data-name="chat_list"
                  >
                    <div className="flex flex-col items-center gap-[10px] p-[10px]">
                      {chats.map((chat) => (
                        <ChatItem
                          key={chat.id}
                          title={chat.title}
                          isActive={chat.active}
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

      {/* Conditional Rendering based on evaluation state */}
      {!isEvaluated ? (
        <>
          {/* Review Input Panel */}
          <div className="flex-[1_0_0] h-full min-h-px min-w-px relative">
            <ReviewInputPanel />
          </div>

          {/* Link Site Panel */}
          <div className="h-full relative shrink-0 w-[450px]">
            <LinkSitePanel />
          </div>
        </>
      ) : (
        evaluationData && (
          <ReviewResultPanel
            reviewText={evaluationData.reviewText}
            score={evaluationData.score}
            followGuidelines={evaluationData.followGuidelines}
            hasGrammaticalErrors={evaluationData.hasGrammaticalErrors}
            onBack={handleBackToInput}
          />
        )
      )}
      
      {/* Delete Confirmation Popup */}
      {chatToDelete !== null && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-[100000]">
          <div className="rounded-[24px] p-8 max-w-md w-full mx-4 shadow-2xl" style={{ backgroundColor: '#38423B' }}>
            <h2
              className="text-[32px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white mb-4 text-center"
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            >
              Conferma eliminazione
            </h2>
            <p
              className="text-[20px] font-['Roboto_Serif',sans-serif] text-white mb-6 text-center"
              style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
            >
              Sei sicuro di voler eliminare questa chat? Questa azione non può essere annullata.
            </p>
            <div className="flex gap-4 justify-center">
              <button
                onClick={cancelDelete}
                className="transition-colors px-8 py-3 rounded-[16px] text-[20px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white"
                style={{ 
                  backgroundColor: '#353831',
                  fontVariationSettings: "'GRAD' 0, 'wdth' 100"
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#3d4739'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#353831'}
              >
                Annulla
              </button>
              <button
                onClick={confirmDelete}
                className="transition-colors px-8 py-3 rounded-[16px] text-[20px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white"
                style={{ 
                  backgroundColor: '#8B0000',
                  fontVariationSettings: "'GRAD' 0, 'wdth' 100"
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#A00000'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#8B0000'}
              >
                Elimina
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}