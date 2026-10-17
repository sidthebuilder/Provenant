"""Check the MCP server registers every tool and that one tool call works end to end."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenant.mcp_server import mcp  # noqa: E402


async def main():
    tools = await mcp.list_tools()
    names = sorted(t.name for t in tools)
    print(len(names), "tools:", names)
    out = await mcp.call_tool("steering_channels", {"swarm": 0})
    print("call ok:", str(out)[:200])
    out = await mcp.call_tool("who_owns", {"site": "grok-ai-village-news-496089.gitlab.io"})
    print("call ok:", str(out)[:200])


asyncio.run(main())
