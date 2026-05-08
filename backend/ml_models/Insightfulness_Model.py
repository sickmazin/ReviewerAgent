from typing import Optional
import torch
from transformers import AutoTokenizer
from .rag import ReviewRAGSystem
from .insight_nn_v2 import InsightReviewScorer

class Insightfulness():
    """
        Full model of Insightfulness network.
        Is formed by InsightReviewScorer and Rag-Checker Model.
         - InsightReviewScorer: is the main model that gives the insightfulness score to the review.
         - Rag-Checker: is a model that checks if the review is compliant with the guidelines.
    """
    def __init__(
            self,
            model_path: Optional[str] = None,
            model_name: str = "microsoft/deberta-v3-small",
            tokenizer: AutoTokenizer = None,
            llm_model_name: str = "gemma3:27b"
    ):
        self._model = None
        self._tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.scorer = InsightReviewScorer(
            model_name=model_name,
            MODEL_PATH=model_path
        )
        self.checker = ReviewRAGSystem(llm_model_name=llm_model_name)



    def score(self, text: str) -> Optional[str]:
        score = self.scorer.inference_pipeline(text, self._tokenizer)
        print(f"[INFO] Insightfulness score: {score}")
        match score:
            case _ if 0 < score <= 36:
                return "BAD"
            case _ if 36 < score <= 70:
                return "GOOD"
            case _ if score>70:
                return "EXCELLENT"
            case _:
                return "ERROR"


    def execute_output(self, review, category, llm_model_name="gemma3:27b"):
        if self.checker.checker.model_name != llm_model_name:
            print(f"[INFO] Switching LLM model from {self.checker.checker.model_name} to {llm_model_name}")
            self.checker = ReviewRAGSystem(llm_model_name=llm_model_name)

        check_result = self.checker.check(
            platform=category,
            review=review,
        )
        # Calcolo dello score di insightfulness
        insightful_score = self.score(review)

        return {
            "score": insightful_score,
            "review": review,
            "is_generic_compliant": check_result.is_generic_compliant,
            "follow_guidelines": check_result.follow_guidelines,
            "grammar_errors": check_result.grammar_errors,
            "reasoning": check_result.reasoning,
            "title": check_result.title,
            "highlights": check_result.highlights.dict() if check_result.highlights else None,
            "details": check_result.details,
        }


# ============================================================
#  Main di test
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  Insightfulness Model — Test")
    print("=" * 72)
    MODEL_PATH = "../.models/v3/best.pt"

    # Istanzia il modello
    model = Insightfulness(model_path=MODEL_PATH)

    # Test 1: Recensione Amazon buona
    print("\n[TEST 1] Recensione Amazon — buona qualità\n")
    review1 = (
        "Queste cuffie sono davvero eccellenti. Il suono è cristallino, "
        "i bassi profondi ma non invadenti. La costruzione in alluminio "
        "trasmette robustezza. La batteria dura facilmente 25 ore. "
        "Consiglio vivamente a chi cerca qualità audio senza spendere una fortuna."
    )

    result1 = model.execute_output(review1, "amazon")
    print(f"Score Insightfulness: {result1['score']}")
    print(f"Conforme euristicamente: {result1['is_generic_compliant']}")
    print(f"Conforme alle linee guida (LLM): {result1['follow_guidelines']}")
    print(f"Errori grammaticali: {result1['grammar_errors']}")
    print(f"Titolo suggerito: '{result1['title']}'")
    if result1['highlights']:
        print(f"Highlights trovati: {len(result1['highlights']['issues'])} problemi")
        violations = [i for i in result1['highlights']['issues'] if i['type'] == 'Violazione']
        if violations:
            print(f"Violazioni: {len(violations)}")
        for issue in result1['highlights']['issues'][:3]:  # Mostra i primi 3
            print(f"  - [{issue['type']}]: '{issue['token']}' (pos {issue['start']}-{issue['end']})")

    # Test 2: Recensione con errori
    print("\n\n[TEST 2] Recensione con errori grammaticali\n")
    review2 = (
        "PRODOTTO OTTIMO!!! La qualita e buona, pero la spedizione "
        "duro troppe giorni. Comunque lo raccomando altamenti consigliato."
    )

    result2 = model.execute_output(review2, "amazon")
    print(f"Score Insightfulness: {result2['score']}")
    print(f"Conforme euristicamente: {result2['is_generic_compliant']}")
    print(f"Conforme alle linee guida (LLM): {result2['follow_guidelines']}")
    print(f"Errori grammaticali: {result2['grammar_errors']}")
    if result2['highlights']:
        print(f"Highlights trovati: {len(result2['highlights']['issues'])} problemi")
        violations = [i for i in result2['highlights']['issues'] if i['type'] == 'Violazione']
        if violations:
            print(f"Violazioni rilevate: {len(violations)}")
        for issue in result2['highlights']['issues'][:5]:
            print(f"  - [{issue['type']}]: '{issue['token']}'")

    # Test 3: Recensione breve
    print("\n\n[TEST 3] Recensione breve\n")
    review3 = "Prodotto buono, arrivato veloce."

    result3 = model.execute_output(review3, "ebay")
    print(f"Score Insightfulness: {result3['score']}")
    print(f"Conforme euristicamente: {result3['is_generic_compliant']}")
    if result3['highlights']:
        print(f"Highlights trovati: {len(result3['highlights']['issues'])} problemi")
        warnings = [i for i in result3['highlights']['issues'] if i['type'] == 'Avvertimento']
        if warnings:
            print(f"Avvertimenti: {len(warnings)}")

    print("\n" + "=" * 72)
    print("  Test completato!")
    print("=" * 72)
