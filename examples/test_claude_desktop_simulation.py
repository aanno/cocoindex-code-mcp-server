#!/usr/bin/env python3
"""
Simulate what Claude Desktop should be doing to connect to the MCP server.
This test runs supergateway as a subprocess and sends MCP protocol messages.
"""

import asyncio
import json
import subprocess
import sys


async def test_claude_desktop_simulation():
    """Simulate Claude Desktop connecting through supergateway."""

    print("🚀 Starting Claude Desktop simulation test...")

    # Start supergateway as subprocess
    cmd = ["pnpm", "dlx", "supergateway", "--streamableHttp", "http://localhost:3033/mcp", "--logLevel", "debug"]

    print(f"📡 Starting supergateway: {' '.join(cmd)}")

    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered
        )

        print("⏳ Waiting for supergateway to start...")
        await asyncio.sleep(2)  # Give it time to start

        # Check if process is still running
        if process.poll() is not None:
            stdout0, stderr0 = process.communicate()
            print("❌ Supergateway exited early!")
            print(f"STDOUT: {stdout0}")
            print(f"STDERR: {stderr0}")
            return False

        print("✅ Supergateway started successfully")

        # Send initialize message
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "claude-desktop-simulation", "version": "1.0.0"},
            },
        }

        print("📤 Sending initialize message...")

        stdin = process.stdin
        stdout = process.stdout

        if stdin is None or stdout is None:
            print("❌ None: process.stdin or process.stdout")
        else:
            stdin.write(json.dumps(init_message) + "\n")
            stdin.flush()

            # Read response
            print("📥 Reading initialize response...")
            response_line = await asyncio.wait_for(asyncio.create_task(asyncio.to_thread(stdout.readline)), timeout=5.0)

            if response_line:
                try:
                    response = json.loads(response_line.strip())
                    print(f"✅ Initialize response: {json.dumps(response, indent=2)}")
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse response: {response_line}")
                    print(f"Error: {e}")
            else:
                print("❌ No response received for initialize")

            # Send tools/list message
            tools_message = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

            print("📤 Sending tools/list message...")
            stdin.write(json.dumps(tools_message) + "\n")
            stdin.flush()

            # Read response
            print("📥 Reading tools/list response...")
            response_line = await asyncio.wait_for(asyncio.create_task(asyncio.to_thread(stdout.readline)), timeout=5.0)

            if response_line:
                try:
                    response = json.loads(response_line.strip())
                    print(f"✅ Tools/list response: {json.dumps(response, indent=2)}")

                    # Check if tools are present
                    if "result" in response and "tools" in response["result"]:
                        tools = response["result"]["tools"]
                        print(f"🔧 Found {len(tools)} tools:")
                        for tool in tools:
                            print(f"   - {tool['name']}: {tool['description']}")
                        return True
                    else:
                        print("❌ No tools found in response")
                        return False

                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse response: {response_line}")
                    print(f"Error: {e}")
                    return False
            else:
                print("❌ No response received for tools/list")
                return False

    except asyncio.TimeoutError:
        print("❌ Timeout waiting for response")
        return False
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        return False
    finally:
        # Clean up
        if process.poll() is None:
            print("🧹 Terminating supergateway...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    result = asyncio.run(test_claude_desktop_simulation())
    if result:
        print("🎉 Claude Desktop simulation successful!")
        sys.exit(0)
    else:
        print("💥 Claude Desktop simulation failed!")
        sys.exit(1)
