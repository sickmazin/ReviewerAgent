import clsx from "clsx";
import imgMain from "figma:asset/ff542d4eca6034b6a943cc359b3e290e3319857e.png";
import imgIcon from "figma:asset/276fe0b420fca2b573b6b1797b4dab39a6f69afc.png";
import { ChatItem } from "./ChatItem";
import { Sparkles, Brain, BarChart3, Info } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router";

type Text1Props = {
  text: string;
  additionalClassNames?: string;
};

function Text1({ text, additionalClassNames = "" }: Text1Props) {
  return (
    <div
      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
      className={clsx(
        "flex flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] relative shrink-0 text-black text-center",
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
        "flex flex-[1_0_0] flex-col font-['Roboto_Serif:Medium',sans-serif] font-medium justify-center leading-[0] min-h-px min-w-px overflow-hidden relative text-black text-ellipsis whitespace-nowrap",
        additionalClassNames
      )}
    >
      <p className="leading-[normal] overflow-hidden text-ellipsis">{text}</p>
    </div>
  );
}

export default function HomePage() {
  const navigate = useNavigate();
  const [chats, setChats] = useState([
    { id: 1, title: "Analisi Paper ML", active: false },
    { id: 2, title: "Review Articolo AI", active: false },
    { id: 3, title: "Studio Ricerca NLP", active: false },
    { id: 4, title: "Valutazione Dataset", active: false },
  ]);

  const handleDeleteChat = (id: number) => {
    setChats(chats.filter((chat) => chat.id !== id));
  };

  const handleNewChat = () => {
    navigate("/review");
  };

  const handleChatClick = (id: number) => {
    navigate(`/review/${id}`);
  };

  return (
    <div
      className="content-stretch flex gap-[20px] items-stretch justify-end px-[25px] py-[20px] relative size-full min-w-full h-screen overflow-hidden"
      data-name="Main"
    >
      <img
        alt=""
        className="absolute backdrop-blur-[2000px] inset-0 max-w-none object-cover opacity-20 pointer-events-none size-full"
        src={imgMain}
      />
      
      {/* Sidebar */}
      <div
        className="bg-[#8f8f8f] h-full relative rounded-[16px] shrink-0 w-[281px]"
        data-name="Review"
      >
        <div className="overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex gap-[15px] items-start p-[15px] relative size-full">
            <div
              className="bg-[#747474] h-full relative rounded-[24px] shrink-0 w-[250px]"
              data-name="chats"
            >
              <div className="flex flex-col items-center overflow-clip rounded-[inherit] size-full">
                <div className="content-stretch flex flex-col gap-[10px] items-center p-[10px] relative size-full">
                  <Text1
                    text="History"
                    additionalClassNames="min-w-full text-[38px] w-[min-content]"
                  />
                  
                  {/* New Chat Button */}
                  <button
                    onClick={handleNewChat}
                    className="bg-[#a7a6a6] hover:bg-[#b8b7b7] transition-colors content-stretch flex gap-[5px] h-[80px] items-center overflow-clip p-[10px] relative rounded-[16px] shrink-0 w-full"
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
                  
                  {/* Chat List */}
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

      {/* Main Content */}
      <div
        className="bg-[#797979] flex-[1_0_0] h-full min-h-px min-w-px relative rounded-[24px]"
        data-name="main_page"
      >
        <div className="flex flex-col items-center overflow-clip rounded-[inherit] size-full">
          <div className="content-stretch flex flex-col gap-[20px] items-center p-[30px] relative size-full overflow-y-auto">
            <Text1
              text="Reviewer Agent"
              additionalClassNames="text-[50px] w-full"
            />

            {/* Welcome Section */}
            <div className="bg-[#a7a6a6] w-full rounded-[16px] p-[25px]">
              <div className="flex items-start gap-4">
                <div className="shrink-0 bg-[#8f8f8f] rounded-full p-3">
                  <Sparkles className="size-8 text-white" />
                </div>
                <div>
                  <h2 
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[32px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-black mb-3"
                  >
                    Benvenuto nel Reviewer Agent
                  </h2>
                  <p 
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[18px] font-['Roboto_Serif',sans-serif] text-black/80 leading-relaxed"
                  >
                    Un sistema intelligente progettato per valutare la profondità e l'originalità dei contenuti.
                    Il nostro agente analizza testi, paper e articoli restituendo un punteggio di{" "}
                    <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">insightfulness</span> basato su metriche avanzate.
                  </p>
                </div>
              </div>
            </div>

            {/* How It Works Section */}
            <div className="bg-[#a7a6a6] w-full rounded-[16px] p-[25px]">
              <div className="flex items-start gap-4">
                <div className="shrink-0 bg-[#8f8f8f] rounded-full p-3">
                  <Brain className="size-8 text-white" />
                </div>
                <div className="flex-1">
                  <h3 
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[28px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-black mb-3"
                  >
                    Come Funziona
                  </h3>
                  <div className="space-y-3">
                    <p 
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                      className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80 leading-relaxed"
                    >
                      <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">PLACEHOLDER_FUNZIONAMENTO:</span> Il sistema analizza
                      il contenuto attraverso una pipeline multi-fase che include preprocessing semantico,
                      estrazione di feature e valutazione contestuale.
                    </p>
                    <p 
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                      className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80 leading-relaxed"
                    >
                      Ogni review viene processata considerando fattori come originalità, profondità dell'analisi,
                      coerenza argomentativa e rilevanza delle conclusioni.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Model Info Section */}
            <div className="bg-[#a7a6a6] w-full rounded-[16px] p-[25px]">
              <div className="flex items-start gap-4">
                <div className="shrink-0 bg-[#8f8f8f] rounded-full p-3">
                  <Info className="size-8 text-white" />
                </div>
                <div className="flex-1">
                  <h3 
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[28px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-black mb-3"
                  >
                    Il Modello
                  </h3>
                  <div className="space-y-3">
                    <p 
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                      className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80 leading-relaxed"
                    >
                      <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">PLACEHOLDER_MODELLO:</span> Nome del modello, architettura
                      utilizzata e specifiche tecniche.
                    </p>
                    <p 
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                      className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80 leading-relaxed"
                    >
                      <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">PLACEHOLDER_TRAINING:</span> Informazioni sul dataset di
                      training, metodologie di fine-tuning e metriche di performance.
                    </p>
                    <p 
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                      className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80 leading-relaxed"
                    >
                      <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">PLACEHOLDER_ARCHITETTURA:</span> Dettagli sull'architettura
                      interna, layer di attenzione e meccanismi di scoring.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Insightfulness Score Section */}
            <div className="bg-[#a7a6a6] w-full rounded-[16px] p-[25px]">
              <div className="flex items-start gap-4">
                <div className="shrink-0 bg-[#8f8f8f] rounded-full p-3">
                  <BarChart3 className="size-8 text-white" />
                </div>
                <div className="flex-1">
                  <h3 
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[28px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-black mb-3"
                  >
                    Insightfulness Score
                  </h3>
                  <p 
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80 leading-relaxed mb-3"
                  >
                    Il punteggio di insightfulness è una metrica composta che valuta:
                  </p>
                  <ul className="space-y-2 ml-4">
                    <li 
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                      className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80"
                    >
                      • <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">Originalità:</span> Quanto il contenuto presenta idee nuove
                    </li>
                    <li 
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                      className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80"
                    >
                      • <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">Profondità:</span> Livello di analisi e dettaglio
                    </li>
                    <li 
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                      className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80"
                    >
                      • <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">Rilevanza:</span> Pertinenza e applicabilità delle conclusioni
                    </li>
                    <li 
                      style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                      className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80"
                    >
                      • <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">Coerenza:</span> Struttura logica dell'argomentazione
                    </li>
                  </ul>
                  <p 
                    style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                    className="text-[16px] font-['Roboto_Serif',sans-serif] text-black/80 leading-relaxed mt-3"
                  >
                    <span className="font-['Roboto_Serif:Medium',sans-serif] font-medium">PLACEHOLDER_METRICHE:</span> Spiegazione dettagliata delle
                    metriche utilizzate e dei range di punteggio.
                  </p>
                </div>
              </div>
            </div>

            {/* Call to Action */}
            <button
              onClick={handleNewChat}
              className="bg-[#8f8f8f] hover:bg-[#a0a0a0] transition-colors w-full rounded-[16px] p-[20px] mt-4"
            >
              <p 
                style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
                className="text-[28px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white"
              >
                Inizia una Nuova Review →
              </p>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}