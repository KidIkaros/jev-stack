"""Adapter layer mapping [STATE][CHOICE][ANSWER] protocol to Laya's API.

This module bridges the original NanoCore-S1 structured decision format:
    [STATE] context [/STATE] [CHOICE] opt1 opt2 opt3 [/CHOICE] [ANSWER] opt2 [/ANSWER]

To Laya's typed decision API:
    agent.system_one(state="...", questions={"route": {"type": "choice", "criteria": [...], "instructions": "..."}})

The adapter also maps Laya's output back to the NanoCore-S1 format for compatibility
with existing downstream consumers.

API:
    adapter = DecisionAdapter(agent)
    result = adapter.decide("[STATE] ... [/STATE] [CHOICE] A B C [/CHOICE]")
    # Returns DecisionResult with:
    #   - choice: selected option
    #   - probabilities: per-option
    #   - confidence: calibrated confidence score
    #   - ece: expected calibration error
    - format_output: re-serialize to [ANSWER]...[/ANSWER][SCORE]...[/SCORE]

Sources:
    - Laya API: https://github.com/receptron/laya
    - Laya Python package: https://pypi.org/project/laya/
    - DecisionResult format: laya.Agent.system_one()
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import OrderedDict


# ── Domain Types ──────────────────────────────────────────────────────────

@dataclass
class Option:
    """A decision option with its label and optional description."""
    label: str
    description: Optional[str] = None


@dataclass
class ChoiceQuestion:
    """A choice question with multiple options."""
    options: List[Option]
    instructions: str = "Select the best option."
    id: str = "route"


@dataclass
class ScoreQuestion:
    """A score question with numerical levels."""
    levels: List[str]
    instructions: str = "Rate your confidence from 0 to 1."
    id: str = "confidence"


@dataclass
class NoulQuestion:
    """A noul (null/yes/no) question."""
    instructions: str = "Can this be decided?"
    true_label: str = "yes"
    false_label: str = "no"
    id: str = "noul"


@dataclass
class DecisionSpec:
    """Parsed decision specification from [STATE][CHOICE] format.

    Attributes:
        state: The context/state text (from [STATE] block)
        choices: List of choice questions
        scores: List of score questions
        nouls: List of noul questions
    """
    state: str = ""
    choices: List[ChoiceQuestion] = field(default_factory=list)
    scores: List[ScoreQuestion] = field(default_factory=list)
    nouls: List[NoulQuestion] = field(default_factory=list)


@dataclass
class DecisionResult:
    """Structured decision result.

    Maps Laya's output to NanoCore-S1's expected format.
    """
    # Per-question answers
    answers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Convenience fields for common single-question case
    choice: Optional[str] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    score: Optional[float] = None
    noul: Optional[bool] = None
    
    # Calibration metrics
    ece: Optional[float] = None
    answer_confidence: Optional[float] = None
    
    # Raw Laya output (for debugging/advanced use)
    raw: Optional[Dict[str, Any]] = None


# ── Parser ────────────────────────────────────────────────────────────────

class DecisionParser:
    """Parses [STATE][CHOICE][ANSWER] text format into structured questions.
    
    This is the format used by NanoCore-S1's original training data:
    
        [STATE] customer query [/STATE]
        [CHOICE] billing technical_support security_fraud [/CHOICE]
        [ANSWER] billing [/ANSWER]
    
    The parser also supports [SCORE], [NOUL], and named questions:
    
        [CHOICE route] billing tech_support [/CHOICE]
        [SCORE confidence] low medium high [/SCORE]
    """
    
    # Token patterns
    STATE_PATTERN = re.compile(
        r'\[STATE\]\s*(.*?)\s*\[/STATE\]', re.DOTALL | re.IGNORECASE
    )
    CHOICE_PATTERN = re.compile(
        r'\[CHOICE(?:\s+(\w+))?\]\s*(.*?)\s*\[/CHOICE\]', re.DOTALL | re.IGNORECASE
    )
    SCORE_PATTERN = re.compile(
        r'\[SCORE(?:\s+(\w+))?\]\s*(.*?)\s*\[/SCORE\]', re.DOTALL | re.IGNORECASE
    )
    NOUL_PATTERN = re.compile(
        r'\[NOUL(?:\s+(\w+))?\]\s*(.*?)\s*\[/NOUL\]', re.DOTALL | re.IGNORECASE
    )
    ANSWER_PATTERN = re.compile(
        r'\[ANSWER(?:\s+(\w+))?\]\s*(.*?)\s*\[/ANSWER\]', re.DOTALL | re.IGNORECASE
    )
    CONTEXT_PATTERN = re.compile(
        r'\[CONTEXT\]\s*(.*?)\s*\[/CONTEXT\]', re.DOTALL | re.IGNORECASE
    )
    
    @classmethod
    def parse(cls, text: str) -> DecisionSpec:
        """Parse structured decision text into a DecisionSpec.
        
        Args:
            text: Text containing [STATE], [CHOICE], [SCORE], [NOUL] blocks
            
        Returns:
            DecisionSpec with parsed state and questions
        """
        text = text.strip()
        
        # Extract state
        state_match = cls.STATE_PATTERN.search(text)
        # Also check for [CONTEXT] as alias
        context_match = cls.CONTEXT_PATTERN.search(text) if not state_match else state_match
        
        state = ""
        if context_match:
            state = context_match.group(1).strip()
        
        spec = DecisionSpec(state=state)
        
        # Parse choice questions
        for match in cls.CHOICE_PATTERN.finditer(text):
            question_id = match.group(1) or "route"
            options_text = match.group(2).strip()
            options = cls._parse_options(options_text)
            
            # Look for answer for this question
            answer = cls._find_answer(text, question_id)
            instructions = f"Select the best option for {question_id}."
            if answer:
                instructions += f" The expected answer is {answer}."
            
            spec.choices.append(ChoiceQuestion(
                options=options,
                instructions=instructions,
                id=question_id,
            ))
        
        # Parse score questions
        for match in cls.SCORE_PATTERN.finditer(text):
            question_id = match.group(1) or "confidence"
            levels_text = match.group(2).strip()
            levels = levels_text.split()
            
            spec.scores.append(ScoreQuestion(
                levels=levels,
                instructions=f"Rate the {question_id} from 0 to 1.",
                id=question_id,
            ))
        
        # Parse noul questions
        for match in cls.NOUL_PATTERN.finditer(text):
            question_id = match.group(1) or "decision"
            instruction_text = match.group(2).strip() or "Can this be decided?"
            
            spec.nouls.append(NoulQuestion(
                instructions=instruction_text,
                id=question_id,
            ))
        
        return spec
    
    @staticmethod
    def _parse_options(text: str) -> List[Option]:
        """Parse space-separated options into Option objects."""
        tokens = text.split()
        options = []
        for token in tokens:
            options.append(Option(label=token))
        return options
    
    @staticmethod
    def _find_answer(text: str, question_id: str) -> Optional[str]:
        """Find the answer for a named question in the text."""
        # Look for [ANSWER] or [ANSWER question_id]
        for match in DecisionParser.ANSWER_PATTERN.finditer(text):
            match_id = match.group(1)
            if match_id == question_id or (not match_id and question_id == "route"):
                return match.group(2).strip()
        return None
    
    @staticmethod
    def format_output(
        choice: Optional[str] = None,
        probabilities: Optional[Dict[str, float]] = None,
        confidence: Optional[float] = None,
        score: Optional[float] = None,
        noul: Optional[bool] = None,
    ) -> str:
        """Format results back to [ANSWER]...[/ANSWER] format.
        
        Args:
            choice: Selected option
            probabilities: Per-option probabilities
            confidence: Confidence score (0-1)
            score: Raw score value
            noul: Boolean decision
            
        Returns:
            Formatted string with [ANSWER], [SCORE], [NOUL] blocks
        """
        parts = []
        
        if choice is not None:
            parts.append(f"[ANSWER] {choice} [/ANSWER]")
        
        if probabilities:
            probs_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(probabilities.items()))
            parts.append(f"[PROBS] {probs_str} [/PROBS]")
        
        if confidence is not None:
            parts.append(f"[SCORE] {confidence:.4f} [/SCORE]")
        
        if score is not None:
            parts.append(f"[SCORE_RAW] {score:.4f} [/SCORE_RAW]")
        
        if noul is not None:
            answer = "true" if noul else "false"
            parts.append(f"[NOUL] {answer} [/NOUL]")
        
        return " ".join(parts) if parts else ""


# ── Adapter ───────────────────────────────────────────────────────────────

class DecisionAdapter:
    """Adapter bridging [STATE][CHOICE] format to Laya's system_one API.
    
    Usage:
        from laya import RLAgent
        agent = RLAgent(model_id_or_path='convaiinnovations/laya')
        adapter = DecisionAdapter(agent)
        
        spec = DecisionParser.parse("[STATE] ... [/STATE] [CHOICE] A B C [/CHOICE]")
        result = adapter.decide(spec)
    """
    
    def __init__(self, agent):
        """Initialize adapter with a Laya agent.
        
        Args:
            agent: A laya.RLAgent instance (or any object with system_one method)
        """
        self.agent = agent
        self.parser = DecisionParser()
    
    def decide(self, spec: DecisionSpec, min_confidence: float = 0.5) -> DecisionResult:
        """Run a decision through the Laya agent.
        
        Args:
            spec: Parsed decision specification
            min_confidence: Minimum confidence to accept an answer
            
        Returns:
            DecisionResult with answers mapped from Laya's output
        """
        # Build questions dict for Laya API
        questions = OrderedDict()
        
        for i, choice in enumerate(spec.choices):
            qid = choice.id or f"choice_{i}"
            criteria = {}
            for opt in choice.options:
                criteria[opt.label] = opt.description or opt.label
            
            questions[qid] = {
                "type": "choice",
                "criteria": criteria,
                "instructions": choice.instructions,
            }
        
        for i, score in enumerate(spec.scores):
            qid = score.id or f"score_{i}"
            questions[qid] = {
                "type": "score",
                "criteria": score.levels,
                "instructions": score.instructions,
            }
        
        for i, noul in enumerate(spec.nouls):
            qid = noul.id or f"noul_{i}"
            questions[qid] = {
                "type": "noul",
                "instructions": noul.instructions,
            }
        
        if not questions:
            # If no explicit questions, infer from state text
            # (Laya can handle just a state + instructions)
            questions = self._infer_questions(spec.state)
        
        # Call Laya's system_one
        raw_result = self.agent.system_one(
            state=spec.state,
            questions=questions,
            min_confidence=min_confidence if min_confidence else None,
        )
        
        # Map results back
        return self._map_result(raw_result, questions)
    
    def decide_text(self, text: str, min_confidence: float = 0.5) -> DecisionResult:
        """Parse text and run decision in one step.
        
        Args:
            text: Structured decision text with [STATE][CHOICE] format
            min_confidence: Minimum confidence threshold
            
        Returns:
            DecisionResult
        """
        spec = self.parser.parse(text)
        return self.decide(spec, min_confidence=min_confidence)
    
    def _infer_questions(self, state: str) -> Dict[str, Dict]:
        """Infer default questions from state text.
        
        For simple use cases where no [CHOICE] block is provided,
        infer a general decision question from the state.
        """
        return {
            "decision": {
                "type": "choice",
                "criteria": ["yes", "no", "uncertain"],
                "instructions": "Based on the context, what is the appropriate action? Select one option.",
            },
            "confidence": {
                "type": "score",
                "criteria": ["low", "medium", "high"],
                "instructions": "How confident are you in this decision?",
            },
        }
    
    def _map_result(self, raw: Dict[str, Any], questions: Dict[str, Dict]) -> DecisionResult:
        """Map Laya's raw output to DecisionResult."""
        result = DecisionResult()
        result.raw = raw
        result.answers = raw.get("answers", {})
        
        # Extract convenience fields from the first choice question
        first_choice_qid = None
        for qid, qdef in questions.items():
            if qdef.get("type") == "choice":
                first_choice_qid = qid
                break
        
        if first_choice_qid and first_choice_qid in result.answers:
            ans = result.answers[first_choice_qid]
            result.choice = ans.get("choice")
            result.probabilities = ans.get("probabilities", {})
            result.confidence = ans.get("answer_confidence")
            result.ece = ans.get("confidence")  # Laya's normalized-entropy confidence
        
        # Extract score if available
        for qid, qdef in questions.items():
            if qdef.get("type") == "score" and qid in result.answers:
                ans = result.answers[qid]
                result.score = ans.get("score")
        
        # Extract noul if available
        for qid, qdef in questions.items():
            if qdef.get("type") == "noul" and qid in result.answers:
                ans = result.answers[qid]
                result.noul = ans.get("noul", ans.get("choice") == "true")
        
        return result
