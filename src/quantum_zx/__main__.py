"""
Local execution script for Quantum ZX-Calculus MCP server.

Usage:
    python -m quantum_zx

This runs the server locally for testing and development.
For production, use FastMCP Cloud deployment.
"""

from .server import mcp

if __name__ == "__main__":
    mcp.run()
