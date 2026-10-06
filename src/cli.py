#!/usr/bin/env python
"""CLI for NanoCore-S1 / Laya decision stack.

Usage:
    # From a state string (no [STATE] wrapper needed)
    echo "Customer says: I was charged $50" | python -m jev_stack.cli --choices "billing,tech,support,security,refund"

    # From a structured decision text
    python -m jev_stack.cli --input "[STATE] Charged $50 [/STATE] [CHOICE] billing refund [/CHOICE]"

    # Interactive mode
    python -m jev_stack.cli --interactive

    # With custom model
    python -m jev_stack.cli --input "Test" --choices "yes,no" --model convaiinnovations/laya

Options:
    --input TEXT         Structured decision text (with [STATE][CHOICE])
    --state TEXT         Raw state text (wraps in [STATE] automatically)
    --choices LIST       Comma-separated choice options
    --scores LIST        Comma-separated score levels (e.g. "low,medium,high")
    --noul TEXT          If set, adds a NOUL question with this text as instructions
    --interactive, -i    Interactive mode
    --model PATH         HuggingFace model path (default: convaiinnovations/laya)
    --min-confidence F   Minimum confidence threshold (default: 0.5)
    --format json|text   Output format (default: text)
"""

import sys
import os
import json
import argparse
import readline

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_query(args) -> str:
    """Build structured query text from CLI args."""
    parts = []

    # State
    if args.input:
        # Input is already structured
        return args.input

    if args.state:
        state = args.state
    elif not sys.stdin.isatty():
        # Read from stdin
        state = sys.stdin.read().strip()
    else:
        state = ""

    if state:
        parts.append(f"[STATE] {state} [/STATE]")

    if args.choices:
        parts.append(f"[CHOICE] {' '.join(args.choices)} [/CHOICE]")

    if args.scores:
        parts.append(f"[SCORE] {' '.join(args.scores)} [/SCORE]")

    if args.noul:
        parts.append(f"[NOUL] {args.noul} [/NOUL]")

    return " ".join(parts)


def format_result_text(result) -> str:
    """Format DecisionResult as human-readable text."""
    from src.adapter import DecisionParser

    lines = ["=" * 50]
    lines.append("Decision Result")
    lines.append("=" * 50)

    if result.choice:
        lines.append(f"\nChoice:     {result.choice}")

    if result.confidence is not None:
        lines.append(f"Confidence: {result.confidence:.4f}")

    if result.ece is not None:
        lines.append(f"ECE:        {result.ece:.4f}")

    if result.probabilities:
        lines.append("\nProbabilities:")
        for opt, prob in sorted(result.probabilities.items(), key=lambda x: -x[1]):
            lines.append(f"  {opt:30s} {prob:.4f} ({prob*100:.1f}%)")

    if result.score is not None:
        lines.append(f"\nScore:      {result.score:.4f}")

    if result.noul is not None:
        lines.append(f"\nNoul:       {result.noul}")

    formatted = DecisionParser.format_output(
        choice=result.choice,
        probabilities=result.probabilities,
        confidence=result.confidence,
    )
    lines.append(f"\nFormatted:  {formatted}")

    return "\n".join(lines)


def format_result_json(result) -> str:
    """Format DecisionResult as JSON."""
    if result.raw:
        return json.dumps(result.raw, indent=2, default=str)
    return json.dumps({
        "choice": result.choice,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "score": result.score,
        "noul": result.noul,
    }, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="NanoCore-S1 / Laya Decision CLI")
    parser.add_argument("--input", type=str, default=None,
                        help="Structured decision text (with [STATE][CHOICE])")
    parser.add_argument("--state", type=str, default=None,
                        help="Raw state text")
    parser.add_argument("--choices", type=lambda s: s.split(","), default=None,
                        help="Comma-separated choice options")
    parser.add_argument("--scores", type=lambda s: s.split(","), default=None,
                        help="Comma-separated score levels")
    parser.add_argument("--noul", type=str, default=None,
                        help="NOUL question instructions")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--model", type=str, default="convaiinnovations/laya",
                        help="Model path")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Minimum confidence threshold")
    parser.add_argument("--format", type=str, default="text", choices=["json", "text"],
                        help="Output format")
    args = parser.parse_args()

    # Import here to avoid loading model before --help
    from laya import RLAgent
    from src.adapter import DecisionAdapter, DecisionParser

    # Initialize
    agent = RLAgent(model_id_or_path=args.model)
    agent.warmup()
    adapter = DecisionAdapter(agent)

    if args.interactive:
        print("=== Interactive Decision Mode ===")
        print("Enter state text, then choices when prompted.")
        print("Type 'quit' or Ctrl+C to exit.\n")

        while True:
            try:
                state = input("[STATE] ").strip()
                if state.lower() in ("quit", "exit", "q", ""):
                    break

                choices_str = input("[CHOICES] ").strip()
                if choices_str:
                    choices = choices_str.split()
                    query = f"[STATE] {state} [/STATE] [CHOICE] {' '.join(choices)} [/CHOICE]"
                else:
                    query = f"[STATE] {state} [/STATE]"

                result = adapter.decide_text(query)

                if args.format == "json":
                    print(format_result_json(result))
                else:
                    print(format_result_text(result))
                print()

            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

        return

    # Build query
    query = build_query(args)
    if not query.strip():
        print("Error: No input provided. Use --input, --state, --choices, or stdin.")
        parser.print_help()
        sys.exit(1)

    # Run decision
    result = adapter.decide_text(query, min_confidence=args.min_confidence)

    # Output
    if args.format == "json":
        print(format_result_json(result))
    else:
        print(format_result_text(result))


if __name__ == "__main__":
    main()
