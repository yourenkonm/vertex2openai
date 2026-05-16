import builtins
import time
import asyncio
import re
import threading
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# 初始化色彩渲染引擎
console = Console()
original_print = builtins.print
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class ProxyStats:
    """全局数据监控中枢"""
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.success_requests = 0
        self.error_requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.lock = threading.Lock()

    def add_request(self, success=True):
        with self.lock:
            self.total_requests += 1
            if success:
                self.success_requests += 1
            else:
                self.error_requests += 1

    def add_tokens(self, p_tokens, c_tokens):
        with self.lock:
            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens

    def get_stats_panel(self):
        uptime = round(time.time() - self.start_time, 2)
        table = Table(title="📊 Vertex2OpenAI 运行监控核心", style="cyan", border_style="blue")
        table.add_column("监控指标", style="magenta", justify="left")
        table.add_column("当前数值", style="green", justify="right")

        table.add_row("⏱️ 运行时长 (秒)", f"{uptime:,.2f}")
        table.add_row("🌐 累计请求总数", f"{self.total_requests:,}")
        table.add_row("✅ 成功响应 (200)", f"[bold green]{self.success_requests:,}[/]")
        table.add_row("❌ 异常阻断", f"[bold red]{self.error_requests:,}[/]")
        table.add_row("⬆️ 总计 Prompt Tokens", f"{self.prompt_tokens:,}")
        table.add_row("⬇️ 总计 Completion Tokens", f"{self.completion_tokens:,}")
        
        return Panel(table, expand=False, title="[bold yellow]神性中枢状态报告[/]")

# 实例化全局统计
stats = ProxyStats()

class SSELogger:
    """负责将净化后的日志发送给 Web UI"""
    def __init__(self):
        self.queues: List[asyncio.Queue] = []
        self.max_history = 100
        self.history = []

    def push(self, plain_text):
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {plain_text}"
        self.history.append(formatted_msg)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        for q in self.queues:
            try:
                q.put_nowait(formatted_msg)
            except asyncio.QueueFull:
                pass

rt_logger = SSELogger()

def custom_print(*args, **kwargs):
    """全局无侵入式 Print 劫持，自动着色与数据抓取"""
    import io
    buf = io.StringIO()
    original_print(*args, file=buf, **kwargs)
    raw_msg = buf.getvalue().strip()
    
    if not raw_msg:
        return

    # 1. 自动嗅探 Token 消耗并计入统计大盘
    if "💰" in raw_msg and "Tokens" in raw_msg:
        try:
            m = re.search(r'提示词:\s*(\d+).*?思考与生成:\s*(\d+)', raw_msg)
            if m:
                stats.add_tokens(int(m.group(1)), int(m.group(2)))
        except:
            pass

    # 2. Rich 终端色彩渲染
    styled_msg = raw_msg
    if "ERROR:" in raw_msg or "❌" in raw_msg or "Exception" in raw_msg:
        styled_msg = f"[bold red]{raw_msg}[/bold red]"
    elif "WARNING:" in raw_msg or "⚠️" in raw_msg:
        styled_msg = f"[bold yellow]{raw_msg}[/bold yellow]"
    elif "INFO:" in raw_msg or "✅" in raw_msg or "DEBUG:" in raw_msg:
        # 高亮模型名字段
        styled_msg = re.sub(r"(gemini-[a-zA-Z0-9\-\.]+)", r"[bold green]\1[/bold green]", raw_msg)
        styled_msg = f"[cyan]{styled_msg}[/cyan]"
    elif "💰" in raw_msg:
        styled_msg = f"[bold magenta]{raw_msg}[/bold magenta]"

    # 追加时间戳并打印到物理控制台
    timestamp = time.strftime("%H:%M:%S")
    console.print(f"[dim]\\[{timestamp}][/dim] {styled_msg}")

    # 3. 剥离颜色代码，推送到前端 Web SSE 面板
    plain_msg = ANSI_ESCAPE.sub('', raw_msg)
    rt_logger.push(plain_msg)

# 狸猫换太子：瞬间接管全局 print
builtins.print = custom_print
