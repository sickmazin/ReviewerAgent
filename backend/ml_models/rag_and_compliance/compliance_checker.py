from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional,List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from backend.ml_models.rag_and_compliance.config import MODEL_NAME,PHONE_RE,EMAIL_RE,URL_RE,GRAMMAR_PROMPT, \
    GUIDELINES_PROMPT,CAPS_RE
from backend.ml_models.rag_and_compliance.guideline_retriever import GuidelineRetriever
from backend.ml_models.rag_and_compliance.output_schema_llm import TextAnalysisResponse,Issue,GuidelinesEvaluation, \
    ComplianceResult


# ============================================================
#  Compliance Checker — heuristic check + LLM check
# ============================================================

class ComplianceChecker:
    """
    Evaluates the compliance of a review in two phases:
      1. _check_generic            → heuristic check (regex, length, uppercase)
      2. check_guidelines_by_model → deep evaluation via local LLM (Ollama)
      3. check_what_to_highlights  → identifies real errors with precise indices (not just stylistic suggestions)
    The LLM check is performed AFTER the heuristic one.
    Results are reported in the final ComplianceResult.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.retriever: Optional[GuidelineRetriever] = None
        
        # Locale or Cloud with Auth
        ollama_base_url = os.getenv("OLLAMA_API_URL")
        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        
        headers = {}
        if ollama_api_key:
            auth_value = ollama_api_key if ollama_api_key.startswith("Bearer ") else f"Bearer {ollama_api_key}"
            headers["Authorization"] = auth_value
            headers["X-API-Key"] = ollama_api_key.replace("Bearer ", "")

        self.llm = ChatOllama(
            model=self.model_name, 
            temperature=0.0,
            base_url=ollama_base_url,
            headers=headers if headers else None
        )

        # check_guidelines_by_model chain
        guidelines_prompt = ChatPromptTemplate.from_template(GUIDELINES_PROMPT)
        guidelines_llm = self.llm.with_structured_output(GuidelinesEvaluation)
        self.guidelines_chain = guidelines_prompt | guidelines_llm

        # check_what_to_highlights
        grammar_prompt = ChatPromptTemplate.from_template(GRAMMAR_PROMPT)
        grammar_llm = self.llm.with_structured_output(TextAnalysisResponse)
        self.grammar_chain = grammar_prompt | grammar_llm


    # ------------------------------------------------------------------
    #  Phase 1 — heuristic check
    # ------------------------------------------------------------------
    def _check_generic(self, review: str, guideline: dict) -> List[Issue]:
        """
        Performs heuristic check and returns a list of Issues.
        """
        issues = []

        # URL Check
        for match in URL_RE.finditer(review):
            issues.append(Issue(
                type="Violation",
                start=match.start(),
                end=match.end(),
                token=match.group(),
                message="Presence of unauthorized URLs/links",
                suggestion=None,
                is_highlight=True
            ))

        # Email Check
        for match in EMAIL_RE.finditer(review):
            issues.append(Issue(
                type="Violation",
                start=match.start(),
                end=match.end(),
                token=match.group(),
                message="Presence of email address — violates privacy policy",
                suggestion=None,
                is_highlight=True
            ))

        # Phone Check
        for match in PHONE_RE.finditer(review):
            issues.append(Issue(
                type="Violation",
                start=match.start(),
                end=match.end(),
                token=match.group(),
                message="Presence of phone number — violates privacy policy",
                suggestion=None,
                is_highlight=True
            ))

        words = review.split()
        wc = len(words)

        # Heuristic checks (SIDEBAR ONLY)
        if wc < 10 and words:
            issues.append(Issue(
                type="Warning",
                start=0,
                end=len(words[0]),
                token=words[0],
                message=f"Very short review ({wc} words). Add details.",
                suggestion=None,
                is_highlight=False
            ))
        elif wc < 30 and words:
            issues.append(Issue(
                type="Suggestion",
                start=0,
                end=len(words[0]),
                token=words[0],
                message="Add specific details about the experience to increase usefulness",
                suggestion=None,
                is_highlight=False
            ))

        # Caps check (SIDEBAR ONLY)
        for match in CAPS_RE.finditer(review):
            issues.append(Issue(
                type="Warning",
                start=match.start(),
                end=match.end(),
                token=match.group(),
                message="Excessive use of CAPS. Reduce for a more professional tone",
                suggestion=None,
                is_highlight=True
            ))

        if not any(c.isalpha() for c in review):
            issues.append(Issue(
                type="Violation",
                start=0,
                end=len(review),
                token=review,
                message="The review does not contain alphabetic text",
                suggestion=None,
                is_highlight=True
            ))

        return issues

    # ------------------------------------------------------------------
    #  Phase 2 — LLM evaluation with retrieved guidelines
    # ------------------------------------------------------------------
    def check_guidelines_by_model(self, review: str, guideline: dict) -> tuple[bool, str, str]:
        """
        Uses local LLM (Ollama) to verify if the review follows the guidelines.
        """
        guidelines_text = guideline.get("text", "")

        try:
            result = self.guidelines_chain.invoke({
                "review_text": review,
                "guidelines_text": guidelines_text,
            })
            return result.valide, result.reasoning, result.title
        except Exception as e:
            print(f"[ComplianceChecker] WARNING: Fallback activated in check_guidelines_by_model due to an LLM error: {e}")
            # Ritorna dei valori di default in caso di crash del parser LLM
            return False, f"[Errore di parsing del modello: {str(e)[:100]}]", "Review"

    # ------------------------------------------------------------------
    #  Phase 3 — LLM grammar evaluation
    # ------------------------------------------------------------------
    def check_what_to_highlights(self, review: str) -> TextAnalysisResponse:
        """
        Analyzes the text to identify ONLY real errors in the text.
        Returns the precise indices (start, end) of each problem in the original text.
        """

        try:
            result = self.grammar_chain.invoke({
                "review_text": review,
            })
            # Assicuriamoci che il testo sia presente anche se l'LLM lo ha omesso
            if result.text is None:
                result.text = review
        except Exception as e:
            print(f"[ComplianceChecker] WARNING: Fallback activated in check_what_to_highlights due to an LLM error: {e}")
            # If the LLM fails (e.g. hallucinations, cut output, invalid JSON)
            # we return an empty response to avoid crashing the whole pipeline.
            return TextAnalysisResponse(text=review, issues=[], grammar_errors=False)

        result.issues = self._get_filtered_issues(result, review)
        # Recalculate grammar_errors in case we removed all alleged errors
        result.grammar_errors = any(i.type in ["Error", "Violation"] for i in result.issues)

        return result


    def _get_filtered_issues(self, result, review: str):
        # Post-processing: verify and correct hallucinations and indices
        filtered_issues = []
        for issue in result.issues:
            # 1. Filter unhelpful suggestions
            if issue.type == "Suggestion":
                similarity = self._calculate_similarity(issue.token, issue.suggestion or "")
                if similarity >= 0.85:
                    continue  # Discard if almost identical

            # 2. Avoid hallucination of correct "è" marked as error
            if issue.token.lower() == "è" and "accent" in issue.message.lower():
                continue

            # 3. Verify/adjust indices
            actual_substring = review[issue.start:issue.end]
            
            # If token is empty (hallucinated away), fill it from indices
            if not issue.token and issue.start < issue.end:
                issue.token = actual_substring
                
            if actual_substring != issue.token and issue.token:
                # The LLM got the indices wrong. Let's try to find the token nearby.
                # Search the entire string first to find occurrences
                matches = list(re.finditer(re.escape(issue.token), review))
                if not matches:
                    continue # Token does not exist in the original text (severe hallucination)

                # Find the match closest to the indices proposed by the LLM
                best_match = min(matches, key=lambda m: abs(m.start() - issue.start))
                issue.start = best_match.start()
                issue.end = best_match.end()

            # 4. Add to the final list
            filtered_issues.append(issue)

        return filtered_issues



    @staticmethod
    def _calculate_similarity(str1: str, str2: str) -> float:
        """Calculates similarity between two strings (0 = completely different, 1 = identical)"""
        str1 = str1.lower().strip()
        str2 = str2.lower().strip()

        if str1 == str2:
            return 1.0

        # Count common characters in order
        common = sum(1 for c in str2 if c in str1)
        max_len = max(len(str1), len(str2))

        return common / max_len if max_len > 0 else 0.0


    # ------------------------------------------------------------------
    #  TOTAL CHECK
    # ------------------------------------------------------------------
    def check(self, review: str, platform_id: str, guideline: dict) -> ComplianceResult:
        """
        Executes _check_generic first, then check_guidelines_by_model, and then highlights.
        Combina all results into a single TextAnalysisResponse with unified Issues.
        """
        # Phase 1 — heuristic check (returns list of Issues)
        generic_issues = self._check_generic(review, guideline)

        # Check if there are violations (not just warnings/suggestions)
        has_violations = any(issue.type == "Violation" for issue in generic_issues)
        is_generic_compliant = not has_violations

        # Phase 2 — error/suggestion analysis with LLM
        highlights_issues = []
        grammar_errors = None
        try:
            highlights_response = self.check_what_to_highlights(review)
            highlights_issues = highlights_response.issues
            grammar_errors = highlights_response.grammar_errors
        except Exception as e:
            print(f"[ComplianceChecker] Error in check_what_to_highlights: {e}")

        
        # Phase 3 — LLM check for guidelines compliance
        follow_guidelines, reasoning, title= None, None, None
        try:
            follow_guidelines, reasoning, title = self.check_guidelines_by_model(review,guideline)
        except Exception as e:
            reasoning = f"[LLM not available: {e}]"



        # Combine all Issues into a single TextAnalysisResponse
        all_issues = generic_issues + highlights_issues

        highlights = TextAnalysisResponse(
            text=review,
            issues=all_issues,
            grammar_errors=grammar_errors if grammar_errors is not None else False
        ) if all_issues else None


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


