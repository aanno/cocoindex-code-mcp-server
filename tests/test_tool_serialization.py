"""Test MCP Tool serialization behavior."""

import json

import mcp.types as types


def test_tool_standard_creation():
    """Test creating a tool with standard fields."""
    tool = types.Tool(
        name="test_tool",
        description="A test tool",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Test query"}
            },
            "required": ["query"]
        }
    )

    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.title is None
    assert tool.outputSchema is None
    assert tool.annotations is None


def test_tool_model_dump():
    """Test tool model_dump() method."""
    tool = types.Tool(
        name="test_tool",
        description="A test tool",
        inputSchema={"type": "object", "properties": {}}
    )

    dumped = tool.model_dump()
    assert "name" in dumped
    assert "description" in dumped
    assert dumped["title"] is None
    assert dumped["outputSchema"] is None
    assert dumped["annotations"] is None


def test_tool_model_dump_exclude_none():
    """Test tool model_dump(exclude_none=True) removes null fields."""
    tool = types.Tool(
        name="test_tool",
        description="A test tool",
        inputSchema={"type": "object", "properties": {}}
    )

    dumped_clean = tool.model_dump(exclude_none=True)
    assert "name" in dumped_clean
    assert "description" in dumped_clean
    assert "title" not in dumped_clean
    assert "outputSchema" not in dumped_clean
    assert "annotations" not in dumped_clean


def test_tool_json_serialization():
    """Test tool JSON serialization produces clean output."""
    tool = types.Tool(
        name="test_tool",
        description="A test tool",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Test query"}
            },
            "required": ["query"]
        }
    )

    dumped_clean = tool.model_dump(mode='json', exclude_none=True)
    json_str = json.dumps(dumped_clean, indent=2)

    assert "test_tool" in json_str
    assert "A test tool" in json_str
    assert "title" not in json_str
    assert "outputSchema" not in json_str
    assert "annotations" not in json_str


def test_tool_with_optional_fields():
    """Test tool creation with optional fields set."""
    tool = types.Tool(
        name="advanced_tool",
        description="An advanced test tool",
        title="Advanced Tool",
        inputSchema={"type": "object", "properties": {}},
        outputSchema={"type": "object"},
        annotations=None  # ToolAnnotations doesn't have experimental parameter
    )

    dumped = tool.model_dump(exclude_none=True)
    assert dumped["name"] == "advanced_tool"
    assert dumped["description"] == "An advanced test tool"
    assert dumped["title"] == "Advanced Tool"
    assert "outputSchema" in dumped
    # annotations=None should be excluded when exclude_none=True
    assert "annotations" not in dumped
