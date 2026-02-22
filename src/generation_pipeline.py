from __future__ import annotations

import logging
from typing import Any

from generation_prompting import (
    render_consistency_prompt,
    render_plan_prompt,
    render_section_prompt,
    render_section_repair_prompt,
)
from generation_quality import (
    EntityPresenceChecker,
    NumericFactChecker,
    OutlineCoverageChecker,
    RepetitionAndDriftChecker,
    StrictConsistencyEditGuard,
    TerminologyConsistencyChecker,
    _token_jaccard as token_jaccard_helper,
)
from generation_state import initialize_state, update_state
from generation_types import (
    GenerationConfig,
    GenerationPlan,
    GenerationResult,
    GenerationState,
    QualityReport,
    SectionSpec,
)
from model import LLMModel, LLMRequest
from tokenizer import Tokenizer, take_last_tokens

logger = logging.getLogger(__name__)


class ChunkWiseGenerationPipeline:
    def __init__(
        self,
        model: LLMModel,
        tokenizer: Tokenizer,
        config: GenerationConfig | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config or GenerationConfig()
        self._coverage_checker = OutlineCoverageChecker()
        self._terminology_checker = TerminologyConsistencyChecker()
        self._entity_checker = EntityPresenceChecker()
        self._numeric_checker = NumericFactChecker()
        self._repetition_drift_checker = RepetitionAndDriftChecker(
            repetition_threshold=self._config.repetition_similarity_threshold,
            drift_overlap_threshold=self._config.drift_overlap_threshold,
        )
        self._consistency_guard = StrictConsistencyEditGuard(
            min_token_jaccard=self._config.consistency_guard_min_token_jaccard,
            min_length_ratio=self._config.consistency_guard_min_length_ratio,
            max_length_ratio=self._config.consistency_guard_max_length_ratio,
            max_added_sentences=self._config.consistency_guard_max_added_sentences,
        )

    def run(
        self,
        topic: str = "",
        objective: str = "",
        target_tokens: int = 4000,
        audience: str = "",
        tone: str = "",
        manual_plan: GenerationPlan | dict[str, Any] | str | None = None,
    ) -> GenerationResult:
        plan = self._resolve_plan(
            topic=topic,
            objective=objective,
            target_tokens=target_tokens,
            audience=audience,
            tone=tone,
            manual_plan=manual_plan,
        )
        state = initialize_state(plan)
        quality_report = QualityReport()
        section_outputs: list[str] = []

        for index, section in enumerate(plan.sections):
            logger.info(f"[Pipeline] Generating section {index + 1}/{len(plan.sections)}: '{section.title}'")
            
            # P0: Generate section with retries
            prev_section = section_outputs[-1] if section_outputs else None
            section_text, section_issues = self._generate_section_with_retries(
                section=section,
                plan=plan,
                state=state,
                section_index=index,
                prev_section=prev_section,
                previous_outputs=section_outputs,
            )
            
            # Add section issues to quality report
            for issue in section_issues:
                quality_report.section_warnings.append(issue)

            actual_len = len(section_text)
            if not section_text:
                section_text = self._build_minimal_section(section)
                quality_report.section_warnings.append(
                    f"Section {index + 1} generated empty output; used key-point fallback."
                )
                logger.warning(f"[Pipeline] Section {index + 1}: empty output, using fallback ({len(section_text)} chars)")
            else:
                logger.info(f"[Pipeline] Section {index + 1} final: {actual_len} chars")

            if section_outputs:
                similarity = self._token_jaccard(section_outputs[-1], section_text)
                if similarity >= self._config.repetition_similarity_threshold:
                    quality_report.section_warnings.append(
                        (
                            f"Section {index + 1} is too similar to the previous section "
                            f"(score={similarity:.2f})."
                        )
                    )
                    logger.warning(f"[Pipeline] Section {index + 1} repetition warning: similarity={similarity:.2f}")

            section_outputs.append(section_text)
            state = update_state(
                state=state,
                plan=plan,
                section_spec=section,
                section_text=section_text,
            )
            logger.info(f"[Pipeline] Section {index + 1} done. State entities: {len(state.known_entities)}, covered_points: {len(state.covered_key_points)}")

        draft_text = "\n\n".join(section_outputs).strip()
        logger.info(f"[Pipeline] Draft complete: {len(draft_text)} chars from {len(section_outputs)} sections")

        logger.info("[Pipeline] Running quality checks...")
        quality_report.coverage_missing = self._coverage_checker.find_missing(
            plan=plan,
            text=draft_text,
        )
        quality_report.terminology_issues = self._terminology_checker.find_issues(
            plan=plan,
            text=draft_text,
        )
        quality_report.entity_missing = self._entity_checker.find_missing(
            plan=plan,
            section_outputs=section_outputs,
        )
        repetition_issues, drift_issues = self._repetition_drift_checker.find_issues(
            plan=plan,
            section_outputs=section_outputs,
        )
        quality_report.repetition_issues = repetition_issues
        quality_report.drift_issues = drift_issues
        logger.info(f"[Pipeline] Quality check: missing={len(quality_report.coverage_missing)}, "
                   f"terminology={len(quality_report.terminology_issues)}, "
                   f"entity_missing={len(quality_report.entity_missing)}, "
                   f"repetition={len(repetition_issues)}, drift={len(drift_issues)}")
        if quality_report.entity_missing:
            for warning in quality_report.entity_missing:
                logger.warning(f"[Pipeline] Entity check: {warning}")

        final_text = draft_text
        if self._config.consistency_pass_enabled and draft_text:
            quality_report.consistency_pass_applied = True
            logger.info("[Pipeline] Running consistency pass...")
            consistency_prompt = render_consistency_prompt(
                plan=plan,
                state=state,
                draft_text=draft_text,
                quality_report=quality_report,
            )
            candidate = self._model.generate(
                LLMRequest(task="consistency_pass", prompt=consistency_prompt)
            )
            final_text, used_fallback = self._consistency_guard.apply(
                original_text=draft_text,
                candidate_text=candidate,
            )
            quality_report.consistency_pass_used_fallback = used_fallback
            if used_fallback:
                logger.warning("[Pipeline] Consistency pass rejected, using draft text")
            else:
                logger.info("[Pipeline] Consistency pass applied successfully")

        return GenerationResult(
            final_text=final_text,
            plan=plan,
            section_outputs=section_outputs,
            final_state=state,
            qc_report=quality_report,
        )

    def _resolve_plan(
        self,
        topic: str,
        objective: str,
        target_tokens: int,
        audience: str,
        tone: str,
        manual_plan: GenerationPlan | dict[str, Any] | str | None,
        max_retries: int = 2,
    ) -> GenerationPlan:
        if manual_plan is not None:
            return self._coerce_plan(manual_plan)

        topic_value = topic.strip()
        objective_value = objective.strip()
        if not topic_value or not objective_value:
            raise ValueError("topic and objective are required when manual_plan is absent")

        prompt = render_plan_prompt(
            topic=topic_value,
            objective=objective_value,
            target_tokens=target_tokens,
            audience=audience.strip(),
            tone=tone.strip(),
        )
        
        # Retry plan generation if parsing fails (e.g., truncated output)
        for attempt in range(max_retries):
            raw_plan = self._model.generate(
                LLMRequest(task="plan_generation", prompt=prompt)
            )
            logger.debug(f"Raw plan output from model (attempt {attempt + 1}):\n{raw_plan}")
            
            try:
                return GenerationPlan.from_json(raw_plan)
            except ValueError as exc:
                is_truncated = "truncated" in str(exc).lower()
                
                if attempt < max_retries - 1:
                    if is_truncated:
                        logger.warning(
                            f"Plan generation attempt {attempt + 1} failed: output truncated. "
                            f"Retrying..."
                        )
                    else:
                        logger.warning(
                            f"Plan generation attempt {attempt + 1} failed: {exc}. "
                            f"Retrying..."
                        )
                    continue
                else:
                    logger.error(f"Failed to parse plan after {max_retries} attempts: {exc}")
                    logger.error(f"Raw plan output:\n{raw_plan}")
                    raise

    @staticmethod
    def _coerce_plan(manual_plan: GenerationPlan | dict[str, Any] | str) -> GenerationPlan:
        if isinstance(manual_plan, GenerationPlan):
            manual_plan.validate()
            return manual_plan
        if isinstance(manual_plan, str):
            return GenerationPlan.from_json(manual_plan)
        if isinstance(manual_plan, dict):
            return GenerationPlan.from_dict(manual_plan)
        raise ValueError("manual_plan must be GenerationPlan, dict, or JSON string")

    def _build_recent_text(self, section_outputs: list[str]) -> str:
        if not section_outputs:
            return ""
        combined = " ".join(section_outputs)
        return take_last_tokens(
            text=combined,
            tokenizer=self._tokenizer,
            max_tokens=self._config.prefix_window_tokens,
        )

    def _build_minimal_section(self, section: SectionSpec) -> str:
        parts = list(section.key_points)
        for entity in section.required_entities:
            if entity not in parts:
                parts.append(entity)
        return " ".join(parts).strip()

    def _check_length(self, section: SectionSpec, section_text: str) -> str:
        token_count = len(self._tokenizer.encode(section_text))
        lower_bound = max(1, int(section.target_length * self._config.min_section_length_ratio))
        upper_bound = max(lower_bound, int(section.target_length * self._config.max_section_length_ratio))
        if token_count < lower_bound or token_count > upper_bound:
            return (
                f"Section '{section.title}' length {token_count} tokens is outside "
                f"target range [{lower_bound}, {upper_bound}]."
            )
        return ""

    def _token_jaccard(self, left_text: str, right_text: str) -> float:
        left_tokens = set(self._tokenizer.encode(left_text))
        right_tokens = set(self._tokenizer.encode(right_text))
        if not left_tokens and not right_tokens:
            return 1.0
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    # P0: Section generation with retry mechanism

    def _calculate_section_quality(
        self,
        section: SectionSpec,
        section_text: str,
        prev_section: str | None,
    ) -> tuple[float, list[str]]:
        """Calculate quality score for a generated section.
        
        Returns:
            Tuple of (quality_score: 0.0-1.0, list_of_issues)
        """
        score = 1.0
        issues: list[str] = []
        
        # 1. Check for missing required entities (critical)
        missing_entities = self._get_missing_entities(section, section_text)
        if missing_entities:
            for entity in missing_entities:
                score -= self._config.entity_missing_penalty
                issues.append(f"Missing required entity: '{entity}'")
        
        # 2. Check length constraints
        length_issue = self._check_length(section, section_text)
        if length_issue:
            if self._config.retry_on_length_violation:
                score -= self._config.length_violation_penalty
                issues.append(length_issue)
        
        # 3. Check repetition with previous section
        if prev_section:
            similarity = token_jaccard_helper(prev_section, section_text)
            if similarity >= self._config.repetition_similarity_threshold:
                score -= self._config.repetition_penalty
                issues.append(
                    f"Too similar to previous section (score={similarity:.2f})"
                )
        
        return max(0.0, score), issues

    def _get_missing_entities(
        self,
        section: SectionSpec,
        section_text: str,
    ) -> list[str]:
        """Return list of required entities missing from section text."""
        text_lower = section_text.lower()
        missing: list[str] = []
        
        for entity in section.required_entities:
            entity_lower = entity.lower()
            # Check exact match
            if entity_lower in text_lower:
                continue
            # Check hyphen/underscore variants
            if entity_lower.replace(" ", "-") in text_lower:
                continue
            if entity_lower.replace(" ", "_") in text_lower:
                continue
            # Check if words appear in order
            words = entity_lower.split()
            if len(words) > 1 and self._words_in_order(words, text_lower):
                continue
            missing.append(entity)
        
        return missing

    def _words_in_order(self, words: list[str], text: str) -> bool:
        """Check if all words appear in order within the text."""
        text_words = text.split()
        word_idx = 0
        
        for tw in text_words:
            tw_clean = tw.strip(".,;:!?()[]{}\"'")
            if tw_clean == words[word_idx]:
                word_idx += 1
                if word_idx >= len(words):
                    return True
        
        return False

    def _generate_section_with_retries(
        self,
        section: SectionSpec,
        plan: GenerationPlan,
        state: GenerationState,
        section_index: int,
        prev_section: str | None,
        previous_outputs: list[str],
    ) -> tuple[str, list[str]]:
        """Generate a section with quality-based retries.
        
        P0: Base retry framework + entity detection
        P1: Repair prompts with issue-specific guidance
        
        Args:
            previous_outputs: List of already generated section texts for context building
        
        Returns:
            Tuple of (final_section_text, list_of_all_issues)
        """
        best_text = ""
        best_score = -1.0
        all_issues: list[str] = []
        
        retry_limit = max(self._config.max_section_retries, 1)
        
        for retry in range(retry_limit):
            logger.info(f"[Pipeline] Section {section_index + 1}: attempt {retry + 1}/{retry_limit}")
            
            # Build context from previous outputs
            generated_prefix = self._build_recent_text(previous_outputs)
            
            # Build appropriate prompt
            if retry == 0:
                # First attempt: normal generation
                prompt = render_section_prompt(
                    plan=plan,
                    state=state,
                    recent_text=generated_prefix,
                    section_spec=section,
                )
            else:
                # P1: Repair attempt with targeted guidance
                prompt = render_section_repair_prompt(
                    plan=plan,
                    state=state,
                    section_spec=section,
                    current_text=best_text,
                    quality_issues=all_issues,
                    retry_index=retry,
                )
            
            # Generate
            section_text = self._model.generate(
                LLMRequest(task="section_generation", prompt=prompt)
            ).strip()
            
            if not section_text:
                logger.warning(f"[Pipeline] Section {section_index + 1} attempt {retry + 1}: empty output")
                continue
            
            # Calculate quality
            score, issues = self._calculate_section_quality(
                section, section_text, prev_section
            )
            
            logger.info(f"[Pipeline] Section {section_index + 1} attempt {retry + 1}: score={score:.2f}, issues={len(issues)}")
            
            # Track best result
            if score > best_score:
                best_score = score
                best_text = section_text
                all_issues = issues
            
            # Check if quality meets threshold
            # Critical check: entity coverage must be satisfied if configured
            missing_entities = self._get_missing_entities(section, section_text)
            has_critical_issues = (
                self._config.retry_on_missing_entities 
                and missing_entities
            )
            
            if score >= self._config.section_quality_threshold and not has_critical_issues:
                logger.info(f"[Pipeline] Section {section_index + 1}: quality threshold met")
                return section_text, issues
            
            if retry < retry_limit - 1:
                logger.warning(f"[Pipeline] Section {section_index + 1}: retrying due to quality issues")
                if missing_entities:
                    logger.warning(f"  Missing entities: {missing_entities}")
                for issue in issues:
                    logger.warning(f"  - {issue}")
        
        # Exhausted retries, return best result
        if retry_limit > 1:
            logger.info(f"[Pipeline] Section {section_index + 1}: using best candidate (score={best_score:.2f})")
        
        if not best_text:
            # Complete failure, return minimal section
            best_text = self._build_minimal_section(section)
            all_issues.append("All generation attempts failed; using minimal fallback")
        
        return best_text, all_issues
