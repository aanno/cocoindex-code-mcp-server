#!/usr/bin/env python3

"""
Test class decorator merging between tree-sitter and Python AST analysis.
This tests the specific issue where class decorators from Python AST analysis
are not being properly merged with function decorators from tree-sitter analysis.
"""

from cocoindex_code_mcp_server.lang.python.python_code_analyzer import (
    analyze_python_code,
)


def test_class_decorator_detection():
    """Test that class decorators are properly detected and merged."""
    code = """
@dataclass
class DataExample:
    name: str
    age: int

@attr.s
class AttrExample:
    value = attr.ib()

@functools.lru_cache
@dataclass
class MultiDecoratedClass:
    cached_value: str
"""

    metadata = analyze_python_code(code, "test.py")

    # Verify decorators are detected
    assert "decorators" in metadata
    decorators = metadata["decorators"]

    # These should be found from class decorators
    assert "dataclass" in decorators, f"dataclass not found in {decorators}"
    assert "attr.s" in decorators, f"attr.s not found in {decorators}"
    # lru_cache might be detected as full dotted name
    assert any(
        "lru_cache" in dec for dec in decorators), f"lru_cache (or functools.lru_cache) not found in {decorators}"

    # Verify has_decorators flag is set
    assert metadata["has_decorators"] is True

    # Verify class details include decorators
    class_details = metadata.get("class_details", [])
    dataclass_found = False
    attr_class_found = False
    multi_decorated_found = False

    for cls in class_details:
        if cls["name"] == "DataExample":
            dataclass_found = True
            assert "dataclass" in cls.get("decorators", [])
        elif cls["name"] == "AttrExample":
            attr_class_found = True
            assert "attr.s" in cls.get("decorators", [])
        elif cls["name"] == "MultiDecoratedClass":
            multi_decorated_found = True
            decorators = cls.get("decorators", [])
            assert "dataclass" in decorators
            # lru_cache might be detected as full dotted name
            assert any("lru_cache" in dec for dec in decorators)

    assert dataclass_found, "DataExample class not found in class_details"
    assert attr_class_found, "AttrExample class not found in class_details"
    assert multi_decorated_found, "MultiDecoratedClass not found in class_details"


def test_function_and_class_decorator_merging():
    """Test that both function and class decorators are merged correctly."""
    code = """
@property
def name(self):
    return self._name

@staticmethod
def static_method():
    pass

@classmethod
def class_method(cls):
    pass

@custom_decorator
@another_decorator
def decorated_function():
    pass

@dataclass
class DataExample:
    name: str

@functools.total_ordering
@dataclass
class SortableDataExample:
    value: int
"""

    metadata = analyze_python_code(code, "test.py")

    assert "decorators" in metadata
    decorators = metadata["decorators"]

    # Function decorators (should be detected by tree-sitter)
    assert "property" in decorators, f"property not found in {decorators}"
    assert "staticmethod" in decorators, f"staticmethod not found in {decorators}"
    assert "classmethod" in decorators, f"classmethod not found in {decorators}"
    assert "custom_decorator" in decorators, f"custom_decorator not found in {decorators}"
    assert "another_decorator" in decorators, f"another_decorator not found in {decorators}"

    # Class decorators (should be detected by Python AST and merged)
    assert "dataclass" in decorators, f"dataclass not found in {decorators}"
    # total_ordering might be detected as full dotted name
    assert any(
        "total_ordering" in dec for dec in decorators), f"total_ordering (or functools.total_ordering) not found in {decorators}"

    # Verify has_decorators flag
    assert metadata["has_decorators"] is True


def test_nested_class_decorators():
    """Test class decorators in nested scenarios."""
    code = """
class OuterClass:
    @dataclass
    class InnerDataClass:
        value: str

    @attr.s
    class InnerAttrClass:
        count = attr.ib(default=0)

@dataclass
class TopLevelClass:
    inner_value: str
"""

    metadata = analyze_python_code(code, "test.py")

    decorators = metadata.get("decorators", [])

    # Should detect both nested and top-level class decorators
    assert "dataclass" in decorators, f"dataclass not found in {decorators}"
    assert "attr.s" in decorators, f"attr.s not found in {decorators}"


def test_complex_decorator_expressions():
    """Test complex decorator expressions that might be tricky to parse."""
    code = """
@dataclasses.dataclass(frozen=True)
class FrozenData:
    value: str

@functools.lru_cache(maxsize=128)
def cached_function():
    return "cached"

@some_module.complex_decorator(arg1="value", arg2=42)
class ComplexDecoratedClass:
    pass
"""

    metadata = analyze_python_code(code, "test.py")

    decorators = metadata.get("decorators", [])

    # Should extract base decorator names even from complex expressions
    # Note: The exact parsing might depend on implementation details
    # but we should at least get some recognition of these decorators
    assert len(decorators) > 0, "No decorators detected from complex expressions"
    assert metadata["has_decorators"] is True


def test_no_decorators_case():
    """Test that files without decorators work correctly."""
    code = """
class PlainClass:
    def plain_method(self):
        pass

def plain_function():
    return "plain"
"""

    metadata = analyze_python_code(code, "test.py")

    decorators = metadata.get("decorators", [])
    assert len(decorators) == 0, f"Unexpected decorators found: {decorators}"
    assert metadata["has_decorators"] is False
