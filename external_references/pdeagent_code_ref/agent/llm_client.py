"""
LLM API 客户端，带合规日志记录

日志格式要求（每行一条合法JSON）：
- timestamp: ISO 8601 时间戳，含时区
- elapsed_seconds: 本次LLM调用耗时（秒）
- response 或 tool_calls: 至少存在其一
"""
import json
import time
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Generator

import httpx


class LLMClient:
    """OpenAI 兼容格式的 LLM 客户端，自动记录合规日志"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        log_path: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.log_path = log_path
        
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )
        
        # 确保日志目录存在
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    def _log(
        self,
        elapsed: float,
        response_text: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        error: Optional[str] = None,
    ):
        """写入一条合规日志记录"""
        if not self.log_path:
            return
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "model": self.model,
        }
        
        if error:
            record["error"] = error
        if response_text is not None:
            record["response"] = response_text
        if tool_calls is not None:
            record["tool_calls"] = tool_calls
        
        # 确保至少存在 response 或 tool_calls
        if "response" not in record and "tool_calls" not in record:
            record["response"] = ""
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _send_request(self, payload: Dict, max_retries: int = 5) -> Dict:
        """
        发送请求，带自动重试和错误恢复
        - 温度不兼容：自动调整为 1 并重试
        - 速率限制：指数退避
        """
        last_exception = None
        for attempt in range(max_retries):
            try:
                resp = self.client.post("/chat/completions", json=payload)
                
                # 尝试解析错误响应体以获取具体错误信息
                if resp.status_code != 200:
                    try:
                        err_data = resp.json()
                        err_msg = err_data.get("error", {}).get("message", resp.text)
                    except Exception:
                        err_msg = resp.text
                    
                    # 温度参数不兼容：自动调整
                    if "temperature" in err_msg.lower() and "only 1 is allowed" in err_msg.lower():
                        print(f"[LLMClient] Model requires temperature=1, auto-adjusting...")
                        payload["temperature"] = 1
                        self.temperature = 1
                        continue  # 立即重试，不增加 attempt
                    
                    # 速率限制：指数退避
                    if resp.status_code == 429:
                        wait = min(2 ** attempt, 30)
                        print(f"[LLMClient] Rate limited (429), waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    
                    resp.raise_for_status()
                
                return resp.json()
            
            except httpx.HTTPStatusError as e:
                last_exception = e
                # 4xx 客户端错误（除429外）不应重试——无效请求重试也无效
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise
                wait = min(2 ** attempt, 30)
                print(f"[LLMClient] HTTP error on attempt {attempt+1}/{max_retries}: {e.response.status_code}, retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                last_exception = e
                wait = min(2 ** attempt, 30)
                print(f"[LLMClient] Request error on attempt {attempt+1}/{max_retries}: {e}, retrying in {wait}s...")
                time.sleep(wait)
        
        raise last_exception

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        发送聊天请求，返回解析后的响应字典
        
        返回格式模拟 OpenAI ChatCompletion:
        {
            "content": "...",
            "tool_calls": [{"name": "...", "arguments": {...}}],
            "finish_reason": "...",
        }
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        
        start = time.time()
        try:
            data = self._send_request(payload)
            elapsed = time.time() - start
            
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            # 某些模型（如 Kimi）返回 reasoning_content 但 content 为空
            reasoning = message.get("reasoning_content", "")
            if not content and reasoning:
                content = reasoning
            
            tool_calls_raw = message.get("tool_calls", [])
            finish_reason = choice.get("finish_reason", "")
            
            # 解析 tool_calls
            parsed_tools = []
            for tc in tool_calls_raw:
                if tc.get("type") == "function":
                    func = tc["function"]
                    parsed_tools.append({
                        "id": tc.get("id", ""),
                        "name": func["name"],
                        "arguments": json.loads(func["arguments"]),
                    })
            
            self._log(
                elapsed=elapsed,
                response_text=content if not parsed_tools else None,
                tool_calls=parsed_tools if parsed_tools else None,
            )
            
            return {
                "content": content,
                "tool_calls": parsed_tools,
                "finish_reason": finish_reason,
                "reasoning_content": reasoning if reasoning else None,
            }
        
        except Exception as e:
            elapsed = time.time() - start
            self._log(elapsed=elapsed, error=str(e))
            raise

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
    ) -> Generator[str, None, None]:
        """流式聊天，逐字返回内容（不记录详细日志，仅记录总耗时）"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        
        start = time.time()
        full_content = []
        try:
            with self.client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        full_content.append(token)
                        yield token
        finally:
            elapsed = time.time() - start
            self._log(elapsed=elapsed, response_text="".join(full_content))

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
