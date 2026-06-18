from typing import Optional
import torch
from transformers import AutoTokenizer

from ml_models.rag_and_compliance.rag_system import ReviewRAGSystem
from .insightfulness_nn import InsightReviewScorer

class Insightfulness:
    """
        Full model of Insightfulness network.
        Is formed by InsightReviewScorer and Rag-Checker Model.
         - InsightReviewScorer: is the main model that gives the insightfulness score to the review.
         - Rag-Checker: is a model that checks if the review is compliant with the guidelines.
    """
    def __init__(
            self,
            model_path: Optional[str] = None,
            model_name: str = "microsoft/deberta-v3-large",
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



    def score(self, text: str) -> Optional[int]:
        score_val = self.scorer.inference_pipeline(text, self._tokenizer)
        print(f"[INFO] Insightfulness score: {score_val}")
        try:
            return int(round(score_val))
        except (ValueError, TypeError):
            return None


    def execute_output(self, review, category, llm_model_name="gemma3:27b"):
        if self.checker.checker.model_name != llm_model_name:
            print(f"[INFO] Switching LLM model from {self.checker.checker.model_name} to {llm_model_name}")
            self.checker.update_llm_model(llm_model_name)

        check_result = self.checker.check(
            platform=category,
            review=review,
        )
        # Calculation of the insightfulness score
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
#  Test Main
# ============================================================

if __name__ == "__main__":
    import os
    print("=" * 72)
    print("  Insightfulness Model — Test")
    print("=" * 72)
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".weights", "v7_frozen", "best.pt"))

    # Instantiate the model
    model = Insightfulness(model_path=MODEL_PATH)

    # Test 1: Good Amazon review
    print("\n[TEST 1] Amazon review — good quality\n")
    review1 = (
        "These headphones are truly excellent. The sound is crystal clear, "
        "the bass is deep but not intrusive. The aluminum construction "
        "conveys robustness. The battery easily lasts 25 hours. "
        "Highly recommend to anyone looking for audio quality without spending a fortune."
    )

    result1 = model.execute_output(review1, "amazon")
    print(f"Insightfulness Score: {result1['score']}")
    print(f"Heuristically Compliant: {result1['is_generic_compliant']}")
    print(f"Guidelines Compliant (LLM): {result1['follow_guidelines']}")
    print(f"Grammar Errors: {result1['grammar_errors']}")
    print(f"Suggested Title: '{result1['title']}'")
    if result1['highlights']:
        print(f"Highlights found: {len(result1['highlights']['issues'])} issues")
        violations = [i for i in result1['highlights']['issues'] if i['type'] == 'Violazione']
        if violations:
            print(f"Violations: {len(violations)}")
        for issue in result1['highlights']['issues'][:3]:  # Show first 3
            print(f"  - [{issue['type']}]: '{issue['token']}' (pos {issue['start']}-{issue['end']})")

    # Test 2: Review with errors
    print("\n\n[TEST 2] Review with grammar errors\n")
    review2 = (
        "EXCELLENT PRODUCT!!! The quality is good, but the shipping "
        "lasted too many days. However, I highly recommend it highly recommended."
    )

    result2 = model.execute_output(review2, "amazon")
    print(f"Insightfulness Score: {result2['score']}")
    print(f"Heuristically Compliant: {result2['is_generic_compliant']}")
    print(f"Guidelines Compliant (LLM): {result2['follow_guidelines']}")
    print(f"Grammar Errors: {result2['grammar_errors']}")
    if result2['highlights']:
        print(f"Highlights found: {len(result2['highlights']['issues'])} issues")
        violations = [i for i in result2['highlights']['issues'] if i['type'] == 'Violazione']
        if violations:
            print(f"Violations detected: {len(violations)}")
        for issue in result2['highlights']['issues'][:5]:
            print(f"  - [{issue['type']}]: '{issue['token']}'")

    # Test 3: Short review
    print("\n\n[TEST 3] Short review\n")
    review3 = "Good product, arrived fast."

    result3 = model.execute_output(review3, "ebay")
    print(f"Insightfulness Score: {result3['score']}")
    print(f"Heuristically Compliant: {result3['is_generic_compliant']}")
    if result3['highlights']:
        print(f"Highlights found: {len(result3['highlights']['issues'])} issues")
        warnings = [i for i in result3['highlights']['issues'] if i['type'] == 'Avvertimento']
        if warnings:
            print(f"Warnings: {len(warnings)}")

    print("\n" + "=" * 72)
    print("  Test complete!")
    print("=" * 72)
