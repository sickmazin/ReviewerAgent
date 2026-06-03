# ============================================================
#  Main system (facade)
# ============================================================

from __future__ import annotations
from typing import Optional
from backend.ml_models.rag_and_compliance.compliance_checker import ComplianceChecker
from backend.ml_models.rag_and_compliance.config import GUIDELINES_PATH
from backend.ml_models.rag_and_compliance.guideline_retriever import GuidelineStore,LightEmbedder,GuidelineRetriever
from backend.ml_models.rag_and_compliance.output_schema_llm import TextAnalysisResponse,Issue,ComplianceResult


class ReviewRAGSystem:
    """
    Main facade of the RAG system.
    """

    def __init__(
            self,
            embedder_model: Optional[str] = None,
            semantic_threshold: float = 0.30,
            llm_model_name: str = "gemma3:27b",
    ):
        self.store     = GuidelineStore(guidelines_path=GUIDELINES_PATH)
        self.embedder  = LightEmbedder(embedder_model)
        self.retriever = GuidelineRetriever(self.store, self.embedder)
        self.checker   = ComplianceChecker(model_name=llm_model_name)
        self.checker.retriever = self.retriever
        self.semantic_threshold = semantic_threshold

    def update_llm_model(self, llm_model_name: str):
        """
        Updates the LLM model used by the checker without recreating the entire system.
        """
        if self.checker.model_name != llm_model_name:
            print(f"[ReviewRAGSystem] Updating LLM model to: {llm_model_name}")
            self.checker = ComplianceChecker(model_name=llm_model_name)
            self.checker.retriever = self.retriever

    def check(
            self,
            platform: str,
            review: Optional[str] = None,
    ) -> ComplianceResult:
        """
        Checks a review's compliance with the guidelines.

        Parameters
        ----------
        platform : platform name (e.g., "amazon") OR
                             free text containing the platform name.
        review             : review text.
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
            # Creates a ComplianceResult with unidentified platform error
            error_issue = Issue(
                type="Violation",
                start=0,
                end=len(review),
                token=review,
                message="Unable to identify the target platform. "
                        f"Available platforms: {', '.join(self.store.platforms)}",
                is_highlight=False
            )
            return ComplianceResult(
                platform_id="unknown",
                platform_name="Unidentified platform",
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
