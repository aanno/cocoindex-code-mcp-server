<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

## Handling Server Exceptions and Propagating to MCP Clients

### 1. Server-side Exception Handling

When building an MCP server using the Python MCP library, you should:

- **Catch exceptions** inside your tool or resource functions.
- **Convert exceptions to an MCP-compatible error structure**, so that clients can identify, display, or react programmatically.
- **Return errors via protocol responses** so clients receive structured error information.

#### Recommended Data Structure

The MCP protocol typically uses a structured dictionary (Python `dict`) to communicate responses. Errors should be sent in a response like:

```python
{
    "error": {
        "type": "mcp_protocol_error",
        "code": 32603,  # Example error code
        "message": "Server error details",
        "data": {}      # Optional additional error context
    }
}
```

This aligns with the protocol conventions for HTTP streaming and error propagation[^1_1].

#### Example: Server Code

Suppose you define a tool/resource that can raise an exception:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Example")

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide two numbers (demo for exception case)"""
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b

# In the server framework, wrap the invocation:
async def handle_request(request):
    try:
        result = divide(**request["args"])
        # Respond with result
        return {"result": result}
    except Exception as ex:
        # Respond with an MCP error object
        return {
            "error": {
                "type": "mcp_protocol_error",
                "code": 32603,
                "message": str(ex)
            }
        }
```

- **Tip:** If you use the MCP library’s default server tools, most exceptions will be caught and returned as errors, but it’s best practice to make this explicit in custom handlers for clarity and debugging[^1_2][^1_3][^1_4].

### 2. Client-side Handling

On the MCP Python client, always **inspect the response** for an `"error"` key before consuming values.

#### Example: Client Code

```python
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client(mcp_server_url) as (read_stream, write_stream, _):
    # Example tool call: division
    args = {'a': 4, 'b': 0}
    await write_stream({"tool": "divide", "args": args})
    response = await read_stream()
    if "error" in response:
        print("Error from server:", response['error']["message"])
    else:
        print("Result:", response["result"])
```

- **Best Practices:**
  + Always check `if "error" in response`.
  + Display or handle the `error["message"]` appropriately.
  + Optionally, react based on `error["type"]` or `code` for automated workflows[^1_5][^1_1].

### 3. Notes and Improvements

- **Data Structure:** Always use a dict with an `"error"` key (and nested `type`, `code`, `message`, optionally `data`)[^1_1].
- **Up-to-date Library:** Use at least MCP Python SDK v1.10.0, as earlier versions may have improper server-side error propagation and critical bugs[^1_6][^1_7].
- **Testing:** Use pytest or similar to simulate error cases and ensure clients correctly process errors received from the server[^1_7].

#### References

- For an authoritative structure of errors and advanced handling, see the official protocol and examples[^1_2][^1_1][^1_3][^1_7].
- For custom server streaming patterns or for handling resource closure, review actual MCP server implementations and released code in the MCP Python repo and documentation[^1_8][^1_4].

<div style="text-align: center">⁂</div>

[^1_1]: <https://community.openai.com/t/openai-mcp-client-starts-to-fail-when-moving-from-sse-to-streamable-http/1275728>

[^1_2]: <https://blog.csdn.net/universsky2015/article/details/146590165>

[^1_3]: <https://simplescraper.io/blog/how-to-mcp>

[^1_4]: <https://www.mcpevals.io/blog/debugging-mcp-servers-tips-and-best-practices>

[^1_5]: <https://composio.dev/blog/mcp-client-step-by-step-guide-to-building-from-scratch>

[^1_6]: <https://bugzilla.redhat.com/show_bug.cgi?id=2376486>

[^1_7]: <https://dev.to/akitana-airtanker/python-mcp-remote-server-the-dawn-of-the-streamable-http-era-with-a-minimalist-template-1o6j>

[^1_8]: <https://github.com/modelcontextprotocol/python-sdk>

---

# ah I'm using the raw stateless server on the server. could you adopt the code for this? and on client I use httpx with MCP library

To adapt your exception handling for a **raw stateless MCP server** using the MCP Python library, you need to:

- Catch all tool errors on the server.
- Return errors in a consistent MCP error structure (as a dict).
- On the client (using httpx), inspect for these error structures in responses.

Below is direct, minimal code for **both stateless server and httpx client**.

### Stateless MCP Server: Error Handling Example

The critical parts are:

- Initialize FastMCP in stateless mode with `stateless_http=True`.
- Wrap each tool handler with a try/except to deliver a standard MCP error response.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoStateless", stateless_http=True)

@mcp.tool()
def divide(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b

# To run server (adapt to your own ASGI/WSGI as needed):
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

- For a **raw stateless server**, the MCP framework will already wrap errors and present them as error dicts, but if you want to handle custom error formatting or add debugging, you can wrap the tool logic yourself[^2_1][^2_5].

### Example: Custom Handling in Tool

If you want to ensure every tool returns a uniform error:

```python
@mcp.tool()
def divide(a: int, b: int) -> float:
    try:
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a / b
    except Exception as ex:
        # Return error in standard MCP format
        return {
            "error": {
                "type": "mcp_protocol_error",
                "code": 32603,
                "message": str(ex)
            }
        }
```

However, the MCP Python library (1.10.0+) will usually catch unhandled exceptions in tool functions and propagate them to the client in the standard structure automatically[^2_1][^2_5].

### httpx Client: MCP Error Handling

When using **httpx** to call the stateless MCP server, just check for the `"error"` key in the JSON response:

```python
import httpx

payload = {
    "tool": "divide",
    "args": {"a": 4, "b": 0}
}

async def call_mcp_server():
    async with httpx.AsyncClient() as client:
        # Use correct endpoint path (adapt as necessary)
        resp = await client.post("http://localhost:8000/slmcp", json=payload)
        data = resp.json()
        if "error" in data:
            print("MCP Error:", data["error"]["message"])
        else:
            print("Result:", data["result"])

# Run with: asyncio.run(call_mcp_server())
```

*Replace `"/slmcp"` with your endpoint if different.*

### Key Points

- **Error Propagation:** Use the standard MCP error dict (`{"error": ...}`) for both custom and unhandled exceptions on the server.
- **No Sessions:** Stateless mode means errors/results are atomic per request/response[^2_1][^2_5].
- **Client Handling:** Check `"error"` in each response and handle as needed.

This setup ensures robust, uniform server-client error handling with MCP for raw, stateless servers and generic HTTP clients[^2_1][^2_5].

<div style="text-align: center">⁂</div>

[^2_1]: <https://github.com/modelcontextprotocol/python-sdk>

[^2_5]: <https://www.youtube.com/watch?v=PYMEspZPcmc\&rut=1f682865304ef12dc177fa897b2fb1c333002480aacb40dc069c9da0f03994d8>

---

# ok, I also have resources that I list on server. but then the client wants to access one, it gets on error. code for raw stateless server, MCP client and httpx please

For your MCP raw stateless server with resources, and an MCP client using httpx that properly handles errors when accessing resources, here is an adapted example that covers your request:

### 1. Raw Stateless MCP Server with Resources and Error Handling

Use the FastMCP server in `stateless_http=True` mode. Define resources with the `@mcp.resource` decorator. Raise exceptions in the resource if the requested resource does not exist, which MCP will propagate as error dictionaries automatically.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ResourceDemo", stateless_http=True)

# Define some example resources
@mcp.resource("resource://hello")
def hello_resource():
    return "Hello, world!"

@mcp.resource("resource://divide")
def divide_resource(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b

@mcp.resource("resource://item/{item_id}")
def get_item(item_id: str):
    # Simulate a resource listing
    items = {"1": "Apple", "2": "Banana", "3": "Cherry"}
    if item_id not in items:
        raise ValueError(f"Item {item_id} not found")
    return items[item_id]

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

- The MCP Python library automatically converts exceptions into error dict responses with keys like `"error": {"type": ..., "message": ...}`.
- The server is stateless, so each request is independent.

### 2. MCP Client Using httpx to Access Server Resources with Error Handling

```python
import asyncio
import httpx

SERVER_URL = "http://localhost:8000/slmcp"  # Adjust if yours differs

async def fetch_resource(resource_uri, args=None):
    payload = {
        "resource": resource_uri,
    }
    if args:
        payload["args"] = args

    async with httpx.AsyncClient() as client:
        response = await client.post(SERVER_URL, json=payload)
        data = response.json()
        if "error" in data:
            print(f"Error accessing resource {resource_uri}: {data['error']['message']}")
            return None
        else:
            print(f"Resource {resource_uri} result: {data.get('result')}")
            return data.get("result")

async def main():
    await fetch_resource("resource://hello")
    await fetch_resource("resource://divide", {"a": 10, "b": 2})
    await fetch_resource("resource://divide", {"a": 10, "b": 0})
    await fetch_resource("resource://item/2")
    await fetch_resource("resource://item/10")  # Non-existent item, should error

if __name__ == "__main__":
    asyncio.run(main())
```

### Explanation

- **Server** registers several resources via `@mcp.resource`. If invalid data or unknown resource parameters are requested, the server raises exceptions.
- These exceptions are automatically converted to structured MCP error responses on the stateless server.
- **Client** makes POST requests to the MCP stateless endpoint and checks for the `"error"` key to detect and handle errors gracefully.
- The `"resource"` key in the JSON request identifies the resource URI; `"args"` provides parameters to the resource method.

This approach follows MCP's stateless mode and error propagation conventions with resources accessed over raw HTTP using `httpx`.

If you want, I can also provide a minimal example using the MCP Python client's built-in streamable HTTP client for resource access. Let me know!

<div style="text-align: center">⁂</div>

---

# and client with MCP as client lib?

Here is an example of how to use the MCP Python client library to access resources on a raw stateless MCP server, handling errors properly. This builds on your previous example, but now using the official MCP Python client rather than raw httpx calls.

### MCP Python Client: Resource Access with Error Handling Example

```python
import asyncio
from mcp.client.streamable_http import streamablehttp_client

MCP_SERVER_URL = "http://localhost:8000/slmcp"  # Adapt to your actual URL

async def fetch_resource(resource_uri, args=None):
    async with streamablehttp_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
        # Send request to resource with optional arguments
        request = {"resource": resource_uri}
        if args:
            request["args"] = args
        await write_stream(request)

        # Await and handle response
        response = await read_stream()
        if "error" in response:
            print(f"Error from server for resource {resource_uri}: {response['error']['message']}")
            return None
        else:
            result = response.get("result")
            print(f"Resource {resource_uri} result: {result}")
            return result

async def main():
    # Examples accessing resources and handling errors
    await fetch_resource("resource://hello")
    await fetch_resource("resource://divide", {"a": 10, "b": 2})
    await fetch_resource("resource://divide", {"a": 10, "b": 0})  # Should trigger error
    await fetch_resource("resource://item/2")
    await fetch_resource("resource://item/10")  # Should trigger error for missing resource

if __name__ == "__main__":
    asyncio.run(main())
```

### Explanation

- The MCP Python client provides `streamablehttp_client` to connect asynchronously to the MCP server over HTTP streaming.
- You send resource requests by passing a dictionary with `"resource"` URI and optionally `"args"` for parameters.
- The client reads the single JSON response per request.
- If the response contains an `"error"` key, the client prints the server error message.
- Otherwise, it processes the `"result"` field.
- This design supports the MCP raw stateless server mode where each request is atomic.

This example should integrate cleanly with your existing stateless server and demonstrates idiomatic MCP Python client usage for resource access with error propagation. Let me know if you want a combined full example with server and client!

<div style="text-align: center">⁂</div>

---

# is is advisable to combine MCP with httpx on the client? or should I prefer a MCP only solution? with code

It is generally **better to use the MCP Python client library alone** when interacting with an MCP server rather than mixing it with raw httpx calls at the client side, for these reasons:

- The MCP client library is **specifically designed to handle MCP protocol nuances**, such as streaming, message framing, error structures, and response parsing, which raw HTTP clients do not handle out of the box.
- MCP client SDKs provide **idiomatic async API abstractions** (like `streamablehttp_client`), making client code simpler, more readable, and less error prone.
- Using the MCP client ensures **correct protocol compliance**, including proper error handling, session management, reconnecting, and performance optimizations inherent in the library.
- Raw httpx or other HTTP clients are useful if you need **custom or lightweight HTTP handling**, or integration with non-MCP endpoints wrapped as MCP resources, but it requires more manual work handling errors and framing.

Below is a minimal code comparison illustrating the idiomatic MCP client vs httpx approach for calling an MCP resource with error handling.

### MCP Python Client Example (Preferred)

```python
import asyncio
from mcp.client.streamable_http import streamablehttp_client

MCP_SERVER_URL = "http://localhost:8000/slmcp"

async def fetch_resource(resource_uri, args=None):
    async with streamablehttp_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
        request = {"resource": resource_uri}
        if args:
            request["args"] = args
        await write_stream(request)
        response = await read_stream()
        if "error" in response:
            print(f"Error: {response['error']['message']}")
            return None
        return response.get("result")

async def main():
    result = await fetch_resource("resource://hello")
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Raw httpx Client Example (More Manual)

```python
import asyncio
import httpx

MCP_SERVER_URL = "http://localhost:8000/slmcp"

async def fetch_resource(resource_uri, args=None):
    payload = {"resource": resource_uri}
    if args:
        payload["args"] = args
    async with httpx.AsyncClient() as client:
        resp = await client.post(MCP_SERVER_URL, json=payload)
        data = resp.json()
        if "error" in data:
            print(f"Error: {data['error']['message']}")
            return None
        return data.get("result")

async def main():
    result = await fetch_resource("resource://hello")
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Summary

- **Use the MCP client library** when you want clean, protocol-safe, and idiomatic MCP interactions in Python.
- **Use httpx or similar HTTP clients** when you need custom HTTP handling unavailable in MCP clients, or for lightweight testing or bridging non-MCP APIs underneath MCP endpoints.
- Mixing both is possible but adds complexity for little benefit unless you have specific HTTP client needs beyond MCP protocol support.

This recommendation aligns with best practices seen in MCP server and client usage guides and MCP ecosystem examples[^5_1][^5_2][^5_5].

If you want, I can help with more complex usage examples or how to integrate custom HTTP calls within MCP resources or tools effectively.

<div style="text-align: center">⁂</div>

[^5_1]: <https://blog.openreplay.com/expose-internal-api-mcp/>

[^5_2]: <https://github.com/mcp-use/mcp-use>

[^5_5]: <https://www.byteplus.com/en/topic/541540>

---

# I have to combine MCP library with asycio and longer running computation (for MCP answers) on the server and with asycio and pytest on the client. with code examples

To combine the MCP Python library with **asyncio** for handling longer-running computations on the server and also use **asyncio** with **pytest** on the client side, you can design both server and client as fully async, allowing the server to run async tools/resources and the client to call them in async test functions.

Here are minimal working examples for both:

### 1. Async MCP Server with Long-running Computation and Asyncio

```python
import asyncio
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AsyncLongRunningServer", stateless_http=True)


@mcp.tool()
async def long_running_task(duration: int) -> str:
    # Simulate a long-running async task
    await asyncio.sleep(duration)
    return f"Completed after {duration} seconds"


if __name__ == "__main__":
    # Run stateless streamable-http server
    mcp.run(transport="streamable-http")
```

- The **tool function is async** and uses `await asyncio.sleep()` to simulate a long task.
- The server runs stateless mode so each request is handled atomically.

### 2. Async MCP Client Using MCP Library with pytest Async Tests

Create a pytest test file, e.g., `test_client.py`:

```python
import asyncio
import pytest
from mcp.client.streamable_http import streamablehttp_client

SERVER_URL = "http://localhost:8000/slmcp"  # Adapt to your server URL


async def fetch_long_running_task(client_url: str, duration: int):
    async with streamablehttp_client(client_url) as (read_stream, write_stream, _):
        await write_stream({"tool": "long_running_task", "args": {"duration": duration}})
        response = await read_stream()
        return response


@pytest.mark.asyncio
async def test_long_running_task_success():
    response = await fetch_long_running_task(SERVER_URL, 3)
    assert "error" not in response
    assert response.get("result") == "Completed after 3 seconds"


@pytest.mark.asyncio
async def test_long_running_task_invalid_duration():
    # Negative duration should raise error or be handled by server
    response = await fetch_long_running_task(SERVER_URL, -1)
    assert "error" in response
```

- The client uses the MCP Python client library's `streamablehttp_client` async context to communicate.
- The test functions are async with `pytest.mark.asyncio` decorator for pytest to handle async tests.
- `fetch_long_running_task()` wraps the MCP client call.

### How to run

- Start the server on `localhost:8000` with the provided server code.
- Install pytest and MCP SDK:

```bash
pip install pytest mcp-sdk httpx
```

- Run tests with:

```bash
pytest test_client.py
```

### Explanation

- The **server tool is an async function** that uses `asyncio.sleep()` to simulate a long computation, fitting the asyncio event loop model.
- The **client uses MCP Python library's async streamablehttp_client** to send and receive messages asynchronously.
- The **pytest tests are async-aware** and await client operations.
- Error and result handling respects MCP's error dictionary pattern.

If you want, I can also help with more advanced examples such as cancellation handling or streaming intermediate results during long computations. Let me know!

<div style="text-align: center">⁂</div>

---

# what are the limits for server throughout with MCP lib and asycio compared to a solution with celery or other solution?

The throughput limits of an MCP server implemented using the MCP Python library with asyncio compared to solutions using Celery or other task queue systems largely depend on the nature of the workload, concurrency management, and architecture.

### MCP Library with asyncio: Throughput Characteristics and Limits

- **Asyncio-based MCP servers** run on a single event loop thread by default. They are efficient for **I/O-bound** tasks (network, DB calls) because async concurrency can interleave many waiting operations without blocking threads.
- For **CPU-bound or long-running computations**, pure asyncio servers will be limited by Python's GIL and the single-threaded event loop, throttling throughput if tasks are computationally heavy on the CPU.
- **Typical throughput** depends on task complexity, but real-world usage of async MCP servers often handles **several tens to hundreds of concurrent requests per second** well when tasks are mostly I/O-bound and short-lived.
- Scaling beyond this can be achieved by running **multiple server instances** behind a load balancer or using multiprocessing or async workers manually.
- MCP library itself does not provide built-in worker queue or task distribution; it relies on the async event loop and your custom tool/resource code design.

### Celery (or Other Task Queues) Solution: Throughput Characteristics and Limits

- Celery is designed for **distributed task processing**, supporting multiple worker processes and machines.
- It excels at **CPU-bound, long-running, or blocking tasks** by offloading them asynchronously to worker pools.
- Throughput scales horizontally by adding workers; it can handle **thousands or tens of thousands of tasks per second** depending on the broker and hardware.
- Celery adds complexity: requires a message broker (Redis, RabbitMQ), task serialization, and result backend management.
- It supports retries, task prioritization, scheduling, and better fault tolerance compared to a pure asyncio setup.

### Summary Comparison

| Aspect | MCP with Asyncio | Celery or Other Task Queue |
| :-- | :-- | :-- |
| Primary concurrency model | Async single-threaded event loop | Multi-process or distributed workers |
| Ideal for workload | I/O-bound, lightweight async tasks | CPU-bound, long-running, blocking tasks |
| Throughput (per process/server) | Tens to hundreds req/s (I/O-bound) | Thousands to tens of thousands tasks/s (with scaling) |
| Horizontal scaling | Multiple server instances + load balancer | Add more worker processes/machines |
| Complexity | Lower, single runtime | Higher, requires brokers and infrastructure |
| Features | Minimal, relies on custom code | Rich task management features (retries, scheduling) |

### Recommendations

- Use **MCP+asyncio server** if your workloads are primarily **I/O-bound**, latency-sensitive, and benefit from the simplicity of async Python without external dependencies.
- Use **Celery or similar** if your workload includes **heavy CPU-bound or long-running tasks** requiring durable, distributed processing with retries and task management.
- You can *combine both*: Use the MCP async server as a frontend for quick request handling and dispatch CPU-heavy tasks asynchronously via Celery workers.

### Code snippet outline for MCP async server invoking Celery tasks

```python
import asyncio
from mcp.server.fastmcp import FastMCP
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')

mcp = FastMCP("AsyncCPULoadServer", stateless_http=True)

@app.task
def cpu_intensive_task(x):
    # Simulate CPU work
    total = 0
    for i in range(x * 10**6):
        total += i
    return total

@mcp.tool()
async def start_cpu_task(x: int):
    # Dispatch to Celery worker
    task = cpu_intensive_task.delay(x)
    return {"task_id": task.id}

@mcp.resource("resource://task_result")
async def get_task_result(task_id: str):
    task = cpu_intensive_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        return {"status": "pending"}
    elif task.state == 'SUCCESS':
        return {"status": "done", "result": task.result}
    else:
        return {"status": task.state, "info": str(task.info)}

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

This pattern offloads heavy tasks to Celery, maintaining MCP server responsiveness.

### References from search results and knowledge

- Asyncio is ideal for I/O-bound workloads but limited by Python GIL for CPU-bound tasks[^7_1].
- Real MCP implementations report practical throughput in the range of a few tens to hundreds requests per second depending on device and load[^7_3].
- Celery is highly scalable and suited for distributed asynchronous task processing but requires additional infrastructure[general knowledge].
- MCP protocol supports stateless async server designs but does not by itself manage worker queues or distributed workloads (users combine with other tools as needed)[^7_2][^7_5].

If you want, I can provide more detailed benchmarks, scaling architectures, or guidance on integrating Celery with MCP Python server for your use case.

<div style="text-align: center">⁂</div>

[^7_1]: <https://dev.to/yoshan0921/accelerate-python-programs-with-concurrent-programming-28j9>

[^7_2]: <https://blog.openreplay.com/extend-mcp-server-database-access/>

[^7_3]: <https://github.com/datalab-to/marker/issues/735>

[^7_5]: <https://www.getambassador.io/blog/model-context-protocol-mcp-connecting-llms-to-apis>
