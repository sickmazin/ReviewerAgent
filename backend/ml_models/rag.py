"""
rag_guideline_checker.py
========================
Sistema RAG per la validazione di recensioni rispetto alle linee guida
delle piattaforme (Amazon, eBay, Trustpilot, Google, ecc.).

Architettura
------------
  1. RAG Retriever  → individua il documento di linee guida corretto
                      in base all'input (nome piattaforma o recensione)
  2. GuidelineStore → repository dei documenti di linee guida letti da file JSON
  3. ComplianceChecker → prima check euristico generale, poi valutazione LLM
  4. InsightBridge  → (opzionale) integra InsightReviewScorer per il punteggio
                      di insightfulness sulla recensione validata
"""

from __future__ import annotations

import os
import re
import textwrap
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional,List,Literal

import numpy as np
import torch
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer


# ============================================================
#  GuidelineStore — carica i file dalla cartella guidelines/
# ============================================================

class GuidelineStore:
    """
    Carica i file .json dalla cartella `guidelines_path`.
    Il nome del file (senza estensione) diventa il platform_id.
    """

    def __init__(self, guidelines_path: str = "guidelines"):
        self._store: dict[str, dict] = {}

        path = Path(guidelines_path)
        if not path.exists():
            raise FileNotFoundError(
                f"[GuidelineStore] Directory '{path.resolve()}' non trovata."
            )

        # Prima prova a caricare file JSON
        json_files = list(path.glob("*.json"))
        if json_files:
            for f in sorted(json_files):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    for required in ("display_name", "keywords", "text"):
                        if required not in data:
                            raise ValueError(f"Campo obbligatorio mancante: '{required}'")
                    platform_id = f.stem.lower()
                    self._store[platform_id] = data
                    print(f"[GuidelineStore] Caricata piattaforma '{platform_id}' da {f.name}")
                except Exception as e:
                    print(f"[GuidelineStore] ⚠️  Errore in {f.name}: {e}")

        # Se non ci sono JSON, carica i file TXT e li converte
        if not self._store:
            for f in sorted(path.glob("*.txt")):
                try:
                    platform_id = f.stem.lower()
                    text_content = f.read_text(encoding="utf-8")
                    # Crea la struttura JSON dal file TXT
                    self._store[platform_id] = {
                        "display_name": platform_id.capitalize(),
                        "keywords": [platform_id],  # Keyword minima
                        "text": text_content,
                    }
                    print(f"[GuidelineStore] Caricata piattaforma '{platform_id}' da {f.name}")
                except Exception as e:
                    print(f"[GuidelineStore] ⚠️  Errore in {f.name}: {e}")

        print(f"[GuidelineStore] Totale: {len(self._store)} piattaforme.")

    def reload(self):
        self._store.clear()
        self.__init__()

    def add(self, platform_id: str, keywords: list[str], text: str):
        self._store[platform_id.lower()] = {
            "display_name": platform_id.capitalize(),
            "keywords": keywords,
            "text": text,
        }

    @property
    def platforms(self) -> list[str]:
        return list(self._store.keys())

    def get(self, platform_id: str) -> Optional[dict]:
        return self._store.get(platform_id.lower())

    def all_texts(self) -> list[tuple[str, str]]:
        return [(k, v["text"]) for k, v in self._store.items()]


# ============================================================
#  Embedder leggero per il retrieval semantico
# ============================================================

class _LightEmbedder:
    _FALLBACK_MODELS = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "microsoft/deberta-v3-small",
    ]

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self._FALLBACK_MODELS[0]
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        for name in ([self.model_name] + self._FALLBACK_MODELS):
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(name)
                self._model = AutoModel.from_pretrained(name).float().eval()
                self.model_name = name
                print(f"[Embedder] Modello caricato: {name}")
                return
            except Exception as e:
                print(f"[Embedder] {name} non disponibile: {e}")
        raise RuntimeError("Nessun modello embedding disponibile.")

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        self._load()
        if not texts:
            # Ritorna un array vuoto con la forma corretta
            return np.empty((0, 384), dtype=np.float32)

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            out = self._model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)


# ============================================================
#  RAG Retriever
# ============================================================

class GuidelineRetriever:
    """
    Recupera il documento di linee guida più pertinente dato un input testuale.

    Strategia a due livelli:
      1. Keyword match  — cerca i 'keywords' del JSON nell'input (veloce, preciso)
      2. Semantic match — embedding cosine similarity (fallback)
    """

    def __init__(
            self,
            store: Optional[GuidelineStore] = None,
            embedder: Optional[_LightEmbedder] = None,
    ):
        self.store    = store or GuidelineStore(guidelines_path="../../guidelines")
        self.embedder = embedder or _LightEmbedder()
        self._index_built = False
        self._platform_ids: list[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def _keyword_match(self, query: str) -> Optional[str]:
        """
        Cerca i 'keywords' del JSON nell'input (parola intera).
        """
        query_l = query.lower()
        for pid, data in self.store._store.items():
            for kw in data.get("keywords", []):
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', query_l):
                    return pid
        return None

    def _build_index(self):
        if self._index_built:
            return
        pairs = self.store.all_texts()
        self._platform_ids = [p for p, _ in pairs]
        texts = [t for _, t in pairs]
        print("[Retriever] Building semantic index...")
        self._embeddings = self.embedder.encode(texts)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._embeddings = self._embeddings / np.maximum(norms, 1e-9)
        self._index_built = True

    def _semantic_match(self, query: str, top_k: int = 1) -> list[tuple[str, float]]:
        self._build_index()
        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        q_emb = self.embedder.encode([query])
        q_norm = np.linalg.norm(q_emb)
        q_emb = q_emb / max(q_norm, 1e-9)
        sims = (self._embeddings @ q_emb.T).squeeze()
        if sims.ndim == 0:
            sims = sims.reshape(1)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self._platform_ids[i], float(sims[i])) for i in top_idx]

    def retrieve(
            self,
            query: str,
            semantic_threshold: float = 0.50,
    ) -> tuple[Optional[str], Optional[dict], str]:
        """
        Recupera la guideline più pertinente.
        FIX: ritorna sempre 3 valori → (platform_id, guideline_dict, method)
             così ReviewRAGSystem.check può fare: pid, guideline, method = ...
        """
        platform_id = self._keyword_match(query)
        if platform_id:
            return platform_id, self.store.get(platform_id), "keyword"

        results = self._semantic_match(query)
        if results:
            platform_id, score = results[0]
            if score >= semantic_threshold:
                return platform_id, self.store.get(platform_id), f"semantic({score:.2f})"

        return None, None, "none"


# ============================================================
#  Risultato della compliance check
# ============================================================
@dataclass
class ComplianceResult:
    platform_id: str
    platform_name: str
    review_text: str
    is_generic_compliant: bool
    follow_guidelines: Optional[bool] = None
    grammar_errors: Optional[bool] = None
    title: Optional[str] = None
    reasoning: Optional[str] = None
    highlights: Optional[TextAnalysisResponse] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "platform_id": self.platform_id,
            "platform_name": self.platform_name,
            "is_generic_compliant": self.is_generic_compliant,
            "follow_guidelines": self.follow_guidelines,
            "grammar_errors": self.grammar_errors,
            "title": self.title,
            "reasoning": self.reasoning,
            "insight_score": getattr(self, "insight_score", None),
            "highlights": self.highlights.dict() if self.highlights else None,
            "details": self.details,
        }


# ============================================================
#  Schema output strutturato per LLM
# ============================================================
class Issue(BaseModel):
    """Rappresenta un singolo problema rilevato nel testo"""
    type: Literal["Violazione", "Avvertimento", "Suggerimento", "Errore", "Da migliorare"]  # Tipo di problema
    start: int  # Indice inizio nel testo
    end: int    # Indice fine nel testo
    token: str  # La parola/frase evidenziata
    message: str  # Descrizione del problema
    suggestion: Optional[str] = None
    is_highlight: bool = True # Se True, viene mostrato come background nel testo

    class Config:
        json_schema_extra = {
            "example": {
                "type": "Errore",
                "start": 8,
                "end": 9,
                "token": "e",
                "message": "Utilizzare il verbo non la congiunzione.",
                "suggestion": "è",
                "is_highlight": True
            }
        }


class TextAnalysisResponse(BaseModel):
    """Risposta completa dell'analisi LLM"""
    text: str  # Testo originale analizzato
    issues: List[Issue]  # Lista di problemi rilevati
    grammar_errors: Optional[bool] = Field(
        default=False,
        description="La recensione contiene errori grammaticali oggettivi? True (contiene errori) o False (è corretta)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "La casa e bella",
                "issues": [
                    {
                        "type": "Errore",
                        "start": 8,
                        "end": 9,
                        "token": "e",
                        "message": "Utilizzare il verbo non la congiunzione.",
                        "suggestion": "è",
                        "is_highlight": True
                    }
                ],
                "grammar_errors": True
            }
        }

class GuidelinesEvaluation(BaseModel):
    valide: bool = Field(
        description="La recensione rispetta le linee guida in input? True o False"
    )
    title: str = Field(
        description="Un titolo breve (max 3-5 parole) che identifichi il PRODOTTO o SERVIZIO recensito. NON deve essere un giudizio sulla recensione. Esempio corretto: 'Samsung Galaxy S26' o 'Ristorante Pizzeria'. Esempio sbagliato: 'Recensione valida' o 'Nessun errore'."
    )
    reasoning: str = Field(
        description="Brevissima analisi della recensione in generale su come è fatta."
    )

# ============================================================
#  Compliance Checker — check euristico + check LLM
# ============================================================

class ComplianceChecker:
    """
    Valuta la conformità di una recensione in due fasi:
      1. _check_generic            → check euristico (regex, lunghezza, maiuscolo)
      2. check_guidelines_by_Model → valutazione profonda tramite LLM locale (Ollama)
      3. check_what_to_highlights → identifica errori reali con indici precisi (non solo suggerimenti stilistici)
    Il check LLM viene eseguito DOPO quello euristico.
    I risultati sono riportati nel ComplianceResult finale.
    """

    def __init__(self, model_name: str = "gemma3:27b"):
        self.model_name = model_name
        self.retriever: Optional[GuidelineRetriever] = None
        self.llm = ChatOllama(
            model=self.model_name, temperature=0.3
        )


    # Regex patterns condivisi
    _URL_RE   = re.compile(r'https?://\S+|www\.\S+', re.I)
    _EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.I)
    _PHONE_RE = re.compile(r'(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}')
    _CAPS_RE  = re.compile(r'[A-ZÀÈÉÌÒÙ]{5,}')

    # ------------------------------------------------------------------
    #  Fase 1 — check euristico
    # ------------------------------------------------------------------
    def _check_generic(self, review: str, guideline: dict) -> List[Issue]:
        """
        Esegue check euristico e restituisce una lista di Issue.
        """
        issues = []

        # Check URL
        for match in self._URL_RE.finditer(review):
            issues.append(Issue(
                type="Violazione",
                start=match.start(),
                end=match.end(),
                token=match.group(),
                message="Presenza di URL/link non consentiti",
                suggestion=None,
                is_highlight=True
            ))

        # Check Email
        for match in self._EMAIL_RE.finditer(review):
            issues.append(Issue(
                type="Violazione",
                start=match.start(),
                end=match.end(),
                token=match.group(),
                message="Presenza di indirizzo email — viola la privacy policy",
                suggestion=None,
                is_highlight=True
            ))

        # Check Phone
        for match in self._PHONE_RE.finditer(review):
            issues.append(Issue(
                type="Violazione",
                start=match.start(),
                end=match.end(),
                token=match.group(),
                message="Presenza di numero di telefono — viola la privacy policy",
                suggestion=None,
                is_highlight=True
            ))

        words = review.split()
        wc = len(words)

        # Heuristic checks (SIDEBAR ONLY)
        if wc < 10 and words:
            issues.append(Issue(
                type="Avvertimento",
                start=0,
                end=len(words[0]),
                token=words[0],
                message=f"Recensione molto breve ({wc} parole). Aggiungi dettagli.",
                suggestion=None,
                is_highlight=False
            ))
        elif wc < 30 and words:
            issues.append(Issue(
                type="Suggerimento",
                start=0,
                end=len(words[0]),
                token=words[0],
                message="Aggiungi dettagli specifici sull'esperienza per aumentare l'utilità",
                suggestion=None,
                is_highlight=False
            ))

        # Caps check (SIDEBAR ONLY)
        for match in self._CAPS_RE.finditer(review):
            issues.append(Issue(
                type="Avvertimento",
                start=match.start(),
                end=match.end(),
                token=match.group(),
                message="Uso eccessivo di MAIUSCOLO. Ridurre per un tono più professionale",
                suggestion=None,
                is_highlight=True
            ))

        if not any(c.isalpha() for c in review):
            issues.append(Issue(
                type="Violazione",
                start=0,
                end=len(review),
                token=review,
                message="La recensione non contiene testo alfabetico",
                suggestion=None,
                is_highlight=True
            ))

        return issues

    # ------------------------------------------------------------------
    #  Fase 2 — valutazione LLM con le linee guida recuperate
    # ------------------------------------------------------------------
    def check_guidelines_by_model(self, review: str, guideline: dict) -> tuple[bool, str, str]:
        """
        Usa l'LLM locale (Ollama) per verificare se la recensione rispetta
        le linee guida.
        """
        guidelines_text = guideline.get("text", "")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
                """
                You're an expert in text analysis and semantics. Your job is to evaluate the validity of a review.
                Given the guidelines, you'll need to verify whether the input review is valid, where valid means it doesn't violate any of the site's rules or guidelines. If even one of the rules or guidelines is violated, the review is considered invalid.
                
                You also need to generate a `title` for this review. 
                - The title MUST be a short, catchy phrase (max 3-5 words) representing the core subject of the review. 
                - Identify the product name, service, or category discussed. 
                - NEVER output generic statuses as titles (e.g., DO NOT output "Recensione valida", "Nessun problema", "Valida").
                
                Therefore, based on the following guidelines:
                {guidelines_text}
                Does the following review comply with the guidelines? Does it comply with all of them?
                Don't use ** or * in the output text and return it in Italian.
                """
            ),
            ("human", "Review to analyze:\n\n{review_text}")
        ])
        llm = self.llm.with_structured_output(GuidelinesEvaluation)
        chain = prompt | llm
        result = chain.invoke({
            "review_text": review,
            "guidelines_text": guidelines_text,
        })

        return result.valide, result.reasoning, result.title

    def check_what_to_highlights(self, review: str) -> TextAnalysisResponse:
        """
        Analizza il testo per identificare SOLO errori reali nel testo.
        Restituisce gli indici precisi (start, end) di ogni problema nel testo originale.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system",
                """
                You are an expert Italian linguist. Your task is to identify OBJECTIVE GRAMMATICAL ERRORS and set grammar_errors to True if any exist.
                
                PAY EXTREME ATTENTION TO:
                1. Missing accents on verbs: "e" (conjunction) vs "è" (verb). 
                   - ONLY flag if the text contains the UNACCENTED "e" where a verb is required.
                   - IF THE TEXT ALREADY USES "è", IT IS CORRECT. DO NOT FLAG IT.
                   - Example of ERROR: "Il pacco e arrivato" -> flag "e".
                   - Example of CORRECT: "Il pacco è arrivato" -> DO NOT FLAG.
                2. Spelling mistakes and typos.
                3. Wrong verb conjugations or subject-verb agreement.
                
                CRITICAL: 
                - Be surgical with indices. Index 0 is the first character.
                - Start index is inclusive, End index is exclusive.
                - The token at text[start:end] must match the problematic word EXACTLY.
                - IF THE SAME ERROR APPEARS MULTIPLE TIMES, CREATE A SEPARATE ISSUE OBJECT FOR EVERY SINGLE OCCURRENCE. DO NOT GROUP THEM.
                - DO NOT HALLUCINATE ERRORS. If the word is correct, ignore it.
                
                Return results in JSON format.
                Don't use ** or * in the output text.
                """
            ),
            ("human", "Analyze this text for objective grammar errors:\n\n{review_text}")
        ])

        llm = self.llm.with_structured_output(TextAnalysisResponse)
        chain = prompt | llm

        try:
            result = chain.invoke({
                "review_text": review,
            })
        except Exception as e:
            print(f"[ComplianceChecker] WARNING: Fallback attivato in check_what_to_highlights a causa di un errore LLM: {e}")
            # Se l'LLM fallisce (es. allucinazioni, output tagliato, JSON invalido)
            # ritorniamo una risposta vuota per non far crashare l'intera pipeline.
            return TextAnalysisResponse(text=review, issues=[], grammar_errors=False)

        # Post-processing: verifica e correggi allucinazioni e indici
        filtered_issues = []
        for issue in result.issues:
            # 1. Filtra suggerimenti poco utili
            if issue.type == "Suggerimento":
                similarity = self._calculate_similarity(issue.token, issue.suggestion or "")
                if similarity >= 0.85:
                    continue  # Scarta se quasi identico

            # 2. Evita l'allucinazione della "è" corretta segnata come errore
            if issue.token.lower() == "è" and "accent" in issue.message.lower():
                continue

            # 3. Verifica/aggiusta gli indici
            actual_substring = review[issue.start:issue.end]
            if actual_substring != issue.token:
                # L'LLM ha sbagliato gli indici. Proviamo a cercare il token nelle vicinanze.
                # Cerchiamo prima in tutta la stringa per trovare le occorrenze
                matches = list(re.finditer(re.escape(issue.token), review))
                if not matches:
                    continue # Token non esiste nel testo originale (allucinazione grave)
                
                # Trova il match più vicino agli indici proposti dall'LLM
                best_match = min(matches, key=lambda m: abs(m.start() - issue.start))
                issue.start = best_match.start()
                issue.end = best_match.end()

            # 4. Aggiungi alla lista finale
            filtered_issues.append(issue)

        result.issues = filtered_issues
        # Ricalcola grammar_errors nel caso avessimo rimosso tutti i presunti errori
        result.grammar_errors = any(i.type in ["Errore", "Violazione"] for i in filtered_issues)
        
        return result

    @staticmethod
    def _calculate_similarity(str1: str, str2: str) -> float:
        """Calcola la similarità tra due stringhe (0 = completamente diverse, 1 = identiche)"""
        str1 = str1.lower().strip()
        str2 = str2.lower().strip()

        if str1 == str2:
            return 1.0

        # Conta caratteri comuni in ordine
        common = sum(1 for c in str2 if c in str1)
        max_len = max(len(str1), len(str2))

        return common / max_len if max_len > 0 else 0.0


    # ------------------------------------------------------------------
    #  CHECK TOTALE
    # ------------------------------------------------------------------
    def check(self, review: str, platform_id: str, guideline: dict) -> ComplianceResult:
        """
        Esegue prima _check_generic, poi check_guidelines_by_Model e poi highlights.
        Combina tutti i risultati in un unico TextAnalysisResponse con Issue unificati.
        """
        # Fase 1 — check euristico (restituisce lista di Issue)
        generic_issues = self._check_generic(review, guideline)

        # Verifica se ci sono violazioni (non solo avvertimenti/suggerimenti)
        has_violations = any(issue.type == "Violazione" for issue in generic_issues)
        is_generic_compliant = not has_violations

        # Fase 2 — analisi errori/suggerimenti con LLM
        highlights_issues = []
        grammar_errors = None
        
        try:
            highlights_response = self.check_what_to_highlights(review)
            highlights_issues = highlights_response.issues
            grammar_errors = highlights_response.grammar_errors
        except Exception as e:
            print(f"[ComplianceChecker] Errore in check_what_to_highlights: {e}")

        # Combina tutti gli Issue in un unico TextAnalysisResponse
        all_issues = generic_issues + highlights_issues

        highlights = TextAnalysisResponse(
            text=review,
            issues=all_issues,
            grammar_errors=grammar_errors if grammar_errors is not None else False
        ) if all_issues else None

        # Fase 3 — check LLM per conformità alle linee guida
        follow_guidelines, reasoning, title= None, None, None
        try:
            follow_guidelines, reasoning, title = self.check_guidelines_by_model(review,guideline)
        except Exception as e:
            reasoning = f"[LLM non disponibile: {e}]"

        return ComplianceResult(
            platform_id=platform_id,
            platform_name=guideline.get("display_name", platform_id),
            review_text=review,
            is_generic_compliant=is_generic_compliant,
            follow_guidelines=follow_guidelines,
            grammar_errors=grammar_errors,
            reasoning=reasoning,
            title=title,
            highlights=highlights,
            details={"word_count": len(review.split()), "char_count": len(review), "llm_model": self.model_name},
        )



# ============================================================
#  Sistema principale (facade)
# ============================================================
class ReviewRAGSystem:
    """
    Facade principale del sistema RAG.
    """

    def __init__(
            self,
            embedder_model: Optional[str] = None,
            semantic_threshold: float = 0.30,
            llm_model_name: str = "gemma3:27b",
    ):
        self.store     = GuidelineStore(guidelines_path="../guidelines")
        self.embedder  = _LightEmbedder(embedder_model)
        self.retriever = GuidelineRetriever(self.store, self.embedder)
        self.checker   = ComplianceChecker(model_name=llm_model_name)
        self.checker.retriever = self.retriever
        self.semantic_threshold = semantic_threshold

    def check(
            self,
            platform: str,
            review: Optional[str] = None,
    ) -> ComplianceResult:
        """
        Controlla la conformità di una recensione alle linee guida.

        Parametri
        ----------
        platform : nome della piattaforma (es. "amazon") OPPURE
                             testo libero contenente il nome della piattaforma.
        review             : testo della recensione.
        use_insight_score  : se True calcola lo score di insightfulness.
        """
        if review is None:
            query  = platform
            review = platform
        else:
            query = platform

        pid, guideline, method = self.retriever.retrieve(
            query,
            semantic_threshold=self.semantic_threshold,
        )

        if pid is None:
            # Crea un ComplianceResult con errore di piattaforma non identificata
            error_issue = Issue(
                type="Violazione",
                start=0,
                end=len(review),
                token=review,
                message="Impossibile identificare la piattaforma di destinazione. "
                        f"Piattaforme disponibili: {', '.join(self.store.platforms)}",
                is_highlight=False
            )
            return ComplianceResult(
                platform_id="unknown",
                platform_name="Piattaforma non identificata",
                review_text=review,
                is_generic_compliant=False,
                highlights=TextAnalysisResponse(text=review, issues=[error_issue], grammar_errors=False)
            )

        result = self.checker.check(review, pid, guideline)

        return result

    def available_platforms(self) -> list[str]:
        return self.store.platforms

    def get_guideline_text(self, platform_id: str) -> Optional[str]:
        g = self.store.get(platform_id)
        return g["text"] if g else None
