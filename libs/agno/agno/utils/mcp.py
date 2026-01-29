from functools import partial
from typing import Callable, Optional
from uuid import uuid4

from agno.utils.log import log_debug, log_exception, log_info

try:
    from mcp import ClientSession
    from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent
    from mcp.types import Tool as MCPTool
except (ImportError, ModuleNotFoundError):
    raise ImportError("`mcp` not installed. Please install using `pip install mcp`")


from agno.media import ImageArtifact


def _wrap_as_markdown(text: str, language: str = "text") -> str:
    """Wrap content in a markdown fenced code block for agent-ui rendering."""
    if not text:
        return "```%s\n\n```" % (language or "text",)
    return "```%s\n%s\n```" % (language or "text", text)


def wrap_mcp_result_as_markdown(text: str, language: str = "text") -> str:
    """
    Format MCP tool result as a markdown code block. Use as format_result when creating
    MCPTools/MultiMCPTools, e.g. format_result=lambda s: wrap_mcp_result_as_markdown(s, "json").
    """
    return _wrap_as_markdown(text, language)


def _format_embedded_resource_for_ui(embedded: EmbeddedResource) -> str:
    """
    Format an EmbeddedResource for agent/UI display.
    If the resource has a 'text' field (e.g. file content from get_file_contents),
    return that text so the UI shows readable content instead of raw JSON.
    No length restriction: same as agent's approach (full tool result is passed to agent-ui).
    """
    resource = getattr(embedded, "resource", None)
    if resource is None:
        return f"[Embedded resource: {embedded.model_dump_json()}]"
    text = getattr(resource, "text", None)
    if text and isinstance(text, str):
        uri = getattr(resource, "uri", None) or ""
        mime = getattr(resource, "mimeType", "") or ""
        # Optional short header for context (e.g. "--- repo://owner/repo/path ---")
        if uri and ("text/" in mime or "application/json" in mime or not mime):
            header = f"--- {uri} ---\n" if uri else ""
            return f"{header}{text}"
        return text
    return f"[Embedded resource: {resource.model_dump_json()}]"


def get_entrypoint_for_tool(
    tool: MCPTool,
    session: ClientSession,
    *,
    format_result: Optional[Callable[[str], str]] = None,
):
    """
    Return an entrypoint for an MCP tool.

    Args:
        tool: The MCP tool to create an entrypoint for
        session: The session to use
        format_result: Optional callable(result_str) -> str to format the tool result
            before returning to the agent (e.g. wrap in markdown). Agent-specific.

    Returns:
        Callable: The entrypoint function for the tool
    """
    from agno.agent import Agent

    async def call_tool(agent: Agent, tool_name: str, **kwargs) -> str:
        try:
            log_info(f"Calling MCP Tool '{tool_name}' with args: {kwargs}")
            result: CallToolResult = await session.call_tool(tool_name, kwargs)  # type: ignore

            # Return an error if the tool call failed
            if result.isError:
                raise Exception(f"Error from MCP tool '{tool_name}': {result.content}")

            # Process the result content
            response_str = ""
            for content_item in result.content:
                if isinstance(content_item, TextContent):
                    response_str += content_item.text + "\n"
                elif isinstance(content_item, ImageContent):
                    # Handle image content if present
                    img_artifact = ImageArtifact(
                        id=str(uuid4()),
                        url=getattr(content_item, "url", None),
                        content=getattr(content_item, "data", None),
                        mime_type=getattr(content_item, "mimeType", "image/png"),
                    )
                    agent.add_image(img_artifact)
                    response_str += "Image has been generated and added to the response.\n"
                elif isinstance(content_item, EmbeddedResource):
                    # Format embedded resources for UI: extract .text when present (e.g. get_file_contents)
                    response_str += _format_embedded_resource_for_ui(content_item) + "\n"
                else:
                    # Handle other content types
                    response_str += f"[Unsupported content type: {content_item.type}]\n"

            out = response_str.strip()
            if format_result is not None and callable(format_result):
                out = format_result(out)
            return out
        except Exception as e:
            log_exception(f"Failed to call MCP tool '{tool_name}': {e}")
            return f"Error: {e}"

    return partial(call_tool, tool_name=tool.name)
