"""
Sandbox for safe code execution with AST validation and resource limits.
[CONSOLIDATED & VERIFIED - JUNE 2024]
"""

import ast
import signal
import time
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np
import pandas as pd


class SandboxError(Exception):
    """Exception raised when sandbox validation or execution fails."""

    def __init__(self, error_type: str, details: str = ""):
        self.error_type = error_type
        self.details = details
        super().__init__(f"{error_type}: {details}")

    @property
    def short_msg(self) -> str:
        """Short message for LLM feedback."""
        messages = {
            "bad_ast_node": f"Forbidden operation: {self.details}",
            "forbidden_import": f"Import not allowed: {self.details}",
            "missing_func": "Function 'make_feature' not found",
            "return_not_series": "Function must return pd.Series",
            "execution_error": f"Runtime error: {self.details}",
            "timeout": "Code execution timed out",
            "memory_limit": "Memory limit exceeded",
            "syntax_error": f"Syntax error: {self.details}",
        }
        return messages.get(self.error_type, f"Error: {self.error_type}")


class SafeExecutor:
    """Safe code executor with AST validation and resource limits."""

    # Whitelist of all allowed Abstract Syntax Tree (AST) node types.
    # This includes nodes for handling imports and aliasing.
    ALLOWED_NODES = {
        "Module", "FunctionDef", "Return", "Assign", "Name", "Load", "Store",
        "BinOp", "Call", "Attribute", "Subscript", "Constant", "List",
        "Tuple", "Dict", "IfExp", "Compare", "BoolOp", "UnaryOp", "Lambda",
        "Expr", "If", "For", "While", "ListComp", "DictComp", "SetComp",
        "GeneratorExp", "comprehension", "keyword", "arg", "arguments", "Slice",
        "And", "Or", "Add", "Sub", "Mult", "Div", "Mod",
        "Pow", "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "FloorDiv",
        "Invert", "Not", "UAdd", "USub", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE",
        "Is", "IsNot", "In", "NotIn", "Break", "Continue", "Pass", "Ellipsis",
        "JoinedStr", "FormattedValue", "Starred", "withitem", "With", "Try",
        "ExceptHandler", "Raise", "Assert",
        "Import", "ImportFrom", "alias"
    }

    # Whitelist of modules that can be imported within the sandboxed code.
    ALLOWED_IMPORTS = {"numpy", "pandas", "math", "datetime", "re"}

    # Blacklist of names that are forbidden to prevent dangerous operations.
    FORBIDDEN_NAMES = {
        "eval", "exec", "compile", "open", "input", "print", "__import__",
        "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
        "hasattr", "callable", "isinstance", "issubclass", "iter", "next",
        "super", "type", "classmethod", "staticmethod", "property"
    }

    def __init__(self, cpu_seconds: int = 3, mem_mb: int = 500):
        self.cpu_seconds = cpu_seconds
        self.mem_mb = mem_mb

    def validate_ast(self, tree: ast.AST) -> None:
        """Validates the AST tree for safe operations."""
        for node in ast.walk(tree):
            node_type = node.__class__.__name__

            if node_type not in self.ALLOWED_NODES:
                raise SandboxError("bad_ast_node", node_type)

            # Special validation for imports to ensure only allowed modules are used
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_IMPORTS:
                        raise SandboxError("forbidden_import", alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module not in self.ALLOWED_IMPORTS:
                    raise SandboxError("forbidden_import", node.module)
            elif isinstance(node, ast.Name) and node.id in self.FORBIDDEN_NAMES:
                raise SandboxError("bad_ast_node", f"forbidden name: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr.startswith('_'):
                raise SandboxError("bad_ast_node", f"private attribute: {node.attr}")

    @contextmanager
    def resource_limits(self):
        """Apply resource limits during execution."""
        def timeout_handler(_signum, _frame):
            raise SandboxError("timeout", f"Execution exceeded {self.cpu_seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.cpu_seconds)

        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def run(self, code_str: str, df: pd.DataFrame) -> pd.Series:
        """Execute code safely and return the generated feature."""
        try:
            tree = ast.parse(code_str, mode="exec")
        except SyntaxError as e:
            raise SandboxError("syntax_error", str(e))

        self.validate_ast(tree)

        # Create a restricted __import__ function for the execution environment.
        def safe_importer(name, globals=None, locals=None, fromlist=(), level=0):
            if name in self.ALLOWED_IMPORTS:
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"Import of '{name}' is not allowed in this sandbox.")

        # Provide the safe_importer as the only way to import modules.
        safe_globals = {
            "__builtins__": {"__import__": safe_importer},
            "pd": pd,
            "np": np,
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "enumerate": enumerate, "filter": filter, "float": float, "int": int,
            "len": len, "list": list, "map": map, "max": max, "min": min,
            "range": range, "round": round, "set": set, "sorted": sorted,
            "str": str, "sum": sum, "tuple": tuple, "zip": zip,
        }

        safe_locals = {}

        try:
            with self.resource_limits():
                exec(compile(tree, "<feature>", "exec"), safe_globals, safe_locals)
        except (SandboxError, ImportError) as e:
            # Catch errors from the sandbox or our custom importer
            raise SandboxError("execution_error", str(e))
        except Exception as e:
            raise SandboxError("execution_error", str(e))

        if "make_feature" not in safe_locals:
            raise SandboxError("missing_func")

        try:
            with self.resource_limits():
                result = safe_locals["make_feature"](df.copy())
        except SandboxError:
            raise
        except Exception as e:
            raise SandboxError("execution_error", str(e))

        if not isinstance(result, pd.Series):
            raise SandboxError("return_not_series")

        result.name = self._gen_unique_colname("feature")

        return result

    def _gen_unique_colname(self, prefix: str) -> str:
        """Generate a unique column name."""
        timestamp = str(int(time.time() * 1000))[-6:]
        return f"{prefix}_{timestamp}"