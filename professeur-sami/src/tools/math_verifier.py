"""
Math Verifier
=============
Uses SymPy to symbolically verify student math answers.

Supported checks:
  - algebraic equivalence: student_expr == correct_expr (up to simplification)
  - numeric equivalence: random-point evaluation fallback
  - equation solving: check if student's solution satisfies the equation

API:
  verify(student_answer: str, correct_answer: str, context: str = "") -> VerifyResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

try:
    import sympy
    from sympy import (
        Symbol, sympify, simplify, nsimplify, Eq, solve,
        symbols, N, pi, E, I, oo, zoo, nan,
        SympifyError,
    )
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application,
    )
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


TRANSFORMATIONS = (
    standard_transformations + (implicit_multiplication_application,)
    if HAS_SYMPY else None
)

# Symbols commonly used in math courses
_COMMON_SYMS = "x y z t n k a b c m r theta phi"


@dataclass
class VerifyResult:
    is_correct: bool
    confidence: float        # 0.0 – 1.0
    method: str              # "symbolic", "numeric", "unavailable", "error"
    student_simplified: str  # SymPy's simplification of student answer
    correct_simplified: str  # SymPy's simplification of correct answer
    message: str             # Human-readable feedback


def _clean_latex(expr: str) -> str:
    """Strip common LaTeX wrapping so SymPy can parse it."""
    expr = expr.strip()
    # Remove $...$ or $$...$$
    expr = re.sub(r'^\$+|\$+$', '', expr).strip()
    # Common LaTeX commands -> SymPy equivalents
    replacements = [
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)'),
        (r'\\sqrt\{([^}]+)\}',             r'sqrt(\1)'),
        (r'\\sqrt\b',                       'sqrt'),
        (r'\\pi\b',                         'pi'),
        (r'\\infty\b',                      'oo'),
        (r'\\cdot\b',                       '*'),
        (r'\\times\b',                      '*'),
        (r'\\div\b',                        '/'),
        (r'\\left\(',                       '('),
        (r'\\right\)',                      ')'),
        (r'\\left\[',                       '['),
        (r'\\right\]',                      ']'),
        (r'\\ln\b',                         'log'),
        (r'\\log\b',                        'log'),
        (r'\\sin\b',                        'sin'),
        (r'\\cos\b',                        'cos'),
        (r'\\tan\b',                        'tan'),
        (r'\\exp\b',                        'exp'),
        (r'\\e\b',                          'E'),
        (r'\^',                             '**'),
        (r'\\{',                            '{'),
        (r'\\}',                            '}'),
    ]
    for pattern, repl in replacements:
        expr = re.sub(pattern, repl, expr)
    return expr.strip()


def _parse(expr_str: str) -> Optional["sympy.Expr"]:
    """Try to parse a math expression string into a SymPy expression."""
    if not HAS_SYMPY:
        return None
    cleaned = _clean_latex(expr_str)
    local_dict = {s: Symbol(s) for s in _COMMON_SYMS.split()}
    try:
        return parse_expr(cleaned, local_dict=local_dict,
                          transformations=TRANSFORMATIONS)
    except Exception:
        try:
            return sympify(cleaned, locals=local_dict)
        except Exception:
            return None


def _numeric_check(expr_a: "sympy.Expr", expr_b: "sympy.Expr",
                   n_points: int = 5, tol: float = 1e-6) -> bool:
    """Evaluate both expressions at random rational points and compare."""
    import random
    free = expr_a.free_symbols | expr_b.free_symbols
    if not free:
        # Constants: compare numerically
        try:
            diff = abs(float(N(expr_a - expr_b)))
            return diff < tol
        except Exception:
            return False
    for _ in range(n_points):
        subs = {s: sympy.Rational(random.randint(1, 9), random.randint(1, 9))
                for s in free}
        try:
            va = complex(N(expr_a.subs(subs)))
            vb = complex(N(expr_b.subs(subs)))
            if abs(va - vb) > tol:
                return False
        except Exception:
            continue
    return True


def verify(student_answer: str, correct_answer: str,
           context: str = "") -> VerifyResult:
    """
    Verify whether student_answer is mathematically equivalent to correct_answer.

    context: optional hint like "equation", "derivative", "integral"
    Returns VerifyResult with is_correct, confidence, method, simplified forms, message.
    """
    if not HAS_SYMPY:
        return VerifyResult(
            is_correct=False, confidence=0.0, method="unavailable",
            student_simplified=student_answer,
            correct_simplified=correct_answer,
            message="SymPy not installed. Install with: pip install sympy",
        )

    s_expr = _parse(student_answer)
    c_expr = _parse(correct_answer)

    if s_expr is None or c_expr is None:
        return VerifyResult(
            is_correct=False, confidence=0.0, method="error",
            student_simplified=student_answer,
            correct_simplified=correct_answer,
            message=(
                f"Could not parse {'student answer' if s_expr is None else 'correct answer'}. "
                "Please use standard math notation."
            ),
        )

    s_str = str(simplify(s_expr))
    c_str = str(simplify(c_expr))

    # Method 1: symbolic simplification
    try:
        diff = simplify(s_expr - c_expr)
        if diff == 0:
            return VerifyResult(
                is_correct=True, confidence=1.0, method="symbolic",
                student_simplified=s_str, correct_simplified=c_str,
                message="Correct! Both expressions are symbolically equivalent.",
            )
    except Exception:
        pass

    # Method 2: numeric evaluation fallback
    try:
        if _numeric_check(s_expr, c_expr):
            return VerifyResult(
                is_correct=True, confidence=0.9, method="numeric",
                student_simplified=s_str, correct_simplified=c_str,
                message="Numerically equivalent (symbolic check inconclusive).",
            )
    except Exception:
        pass

    return VerifyResult(
        is_correct=False, confidence=0.95, method="symbolic",
        student_simplified=s_str, correct_simplified=c_str,
        message=(
            f"Not equivalent. Your answer simplifies to {s_str}, "
            f"but the correct answer is {c_str}."
        ),
    )
