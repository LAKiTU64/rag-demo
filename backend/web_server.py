#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent Web服务器
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Iterable
from pydantic import BaseModel
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
UTILS_DIR = BACKEND_DIR / "utils"

for path in (ROOT_DIR, BACKEND_DIR, UTILS_DIR):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

from utils.nsys_to_ncu_analyzer import NSysToNCUAnalyzer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
from backend.offline_llm import get_offline_qwen_client

# 导入AI Agent核心 & 知识库摄取
try:
    from backend.agent_core import AIAgent
except Exception as e:
    print(f"无法导入 AIAgent: {e}")
try:
    from backend.knowledge_bases.kb_ingest import ingest_json_to_faiss, ingest_model_config
except Exception as e:
    print(f"⚠️ 无法导入知识库摄取模块: {e}")
    ingest_json_to_faiss = None  # type: ignore
    ingest_model_config = None  # type: ignore
class FinalReportRequest(BaseModel):
    job_id: str
    extra_query: Optional[str] = None

class SGLangAnalyzeRequest(BaseModel):
    """请求: 一次性触发 sglang nsys+ncu 快速分析。

    仅用于同步接口 /analyze/sglang (不生成高级/增强报告)。"""
    model_path: str
    batch_size: int = 1
    input_len: int = 128
    output_len: int = 1
    top_k: int = 20
    min_duration_ms: float = 1.0
    max_ncu_kernels: int = 10

class AnalysisSubmitRequest(BaseModel):
    """提交异步分析作业的请求体。

    支持附加生成高级 / 增强报告，以及远程代码信任等参数。"""
    model_path: str
    batch_size: int = 1
    input_len: int = 128
    output_len: int = 1
    top_k: int = 20
    min_duration_ms: float = 1.0
    max_ncu_kernels: int = 10
    allow_remote_code: bool = False
    advanced: bool = False
    advanced_detailed: bool = False
    advanced_json: bool = False
    generate_enriched: bool = False
    ingest_advanced: bool = False
    kb_path: Optional[str] = None
    note: Optional[str] = None

class FullAnalysisRequest(BaseModel):
    """端到端综合分析请求体。

    用于 /analysis/full 接口: nsys + ncu + 基础报告 + (可选) 高阶/增强报告 + (可选) KB 摄取。"""
    model_path: str
    batch_size: int = 1
    input_len: int = 128
    output_len: int = 1
    top_k: int = 20
    min_duration_ms: float = 1.0
    max_ncu_kernels: int = 10
    allow_remote_code: bool = False
    advanced: bool = False
    advanced_detailed: bool = False
    advanced_json: bool = False
    generate_enriched: bool = False
    ingest_advanced: bool = False
    kb_path: Optional[str] = None
    note: Optional[str] = None

# 加载配置
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

app = FastAPI(
    title="AI Agent LLM性能分析器",
    description="智能的大语言模型性能分析Web服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

agent = None
active_connections: Dict[str, WebSocket] = {}
OFFLINE_QWEN_PATH = Path(os.getenv("QWEN_LOCAL_MODEL_PATH", "/workspace/Qwen3-32B"))


class AnalysisJob:
    def __init__(self, job_id: str, params: Dict[str, Any]):
        self.job_id = job_id
        self.params = params
        self.status = 'pending'
        self.error: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.artifacts: Dict[str, Any] = {}
        self.started_at = datetime.now().isoformat()
        self.finished_at: Optional[str] = None

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, AnalysisJob] = {}
    def create(self, params: Dict[str, Any]) -> AnalysisJob:
        import uuid
        jid = uuid.uuid4().hex[:12]
        job = AnalysisJob(jid, params)
        self.jobs[jid] = job
        return job
    def get(self, job_id: str) -> Optional[AnalysisJob]:
        return self.jobs.get(job_id)

job_manager = JobManager()

API_LABELS = {
    "auto": "智能推荐",
    "langchain": "LangChain Agent",
    "nsys": "NSys 性能分析",
    "ncu": "NCU 深度分析",
    "custom": "自定义工具链"
}

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"🔗 连接建立: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"❌ 连接断开: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                print(f"发送消息失败: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global agent
    if AIAgent:
        try:
            agent = AIAgent(CONFIG)
            print("✅ AI Agent初始化成功")
        except Exception as e:
            print(f"⚠️ AI Agent初始化失败: {e}")
    
    print("🤖 AI Agent Web服务器启动完成")
    print(f"📡 服务地址: http://{CONFIG['server']['host']}:{CONFIG['server']['port']}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """主页"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Agent LLM性能分析器</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1>🤖 AI Agent LLM性能分析器</h1>
        <p>请访问 <a href="/chat">/chat</a> 开始使用</p>
        <p>API文档: <a href="/docs">/docs</a></p>
    </body>
    </html>
    """

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """聊天页面"""
    chat_file = Path(__file__).parent.parent / "frontend" / "chat.html"
    if chat_file.exists():
        return chat_file.read_text(encoding='utf-8')
    else:
        return HTMLResponse(
            content="<h1>聊天页面未找到</h1>",
            status_code=404
        )

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket连接端点"""
    await manager.connect(websocket, session_id)
    
    # 发送欢迎消息
    await manager.send_message(session_id, {
        "type": "assistant_message",
        "content": """🤖 **欢迎使用AI Agent LLM性能分析器！**

我可以帮您：
• 🔍 分析各种LLM模型的性能
• 📊 进行NSys全局性能分析
• 🔬 执行NCU深度kernel分析
• 💡 提供性能优化建议

请告诉我您的分析需求！例如：
"分析 llama-7b 模型，batch_size=1"
""",
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            await handle_websocket_message(session_id, message_data)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        print(f"WebSocket错误: {e}")
        manager.disconnect(session_id)

async def handle_websocket_message(session_id: str, message_data: dict):
    """处理WebSocket消息"""
    
    message_type = message_data.get("type", "")
    content = message_data.get("content", "")
    api_choice = message_data.get("api", "auto")
    
    if message_type == "user_message":
        await process_user_message(session_id, content, api_choice)
    
    elif message_type == "ping":
        await manager.send_message(session_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })

def resolve_api_selection(api: str) -> str:
    if api == "auto":
        return "langchain" if agent else "nsys"
    return api or "langchain"


async def process_user_message(session_id: str, message: str, api: str = "auto"):
    """处理用户消息"""
    
    try:
        resolved_api = resolve_api_selection(api)
        api_label = API_LABELS.get(resolved_api, resolved_api)

        await manager.send_message(session_id, {
            "type": "assistant_message",
            "content": (
                "🔄 正在处理您的请求\n\n"
                f"• 选择的API: **{api_label}**\n"
                f"• 请求内容: {message}"
            ),
            "timestamp": datetime.now().isoformat()
        })
        
        response = await dispatch_api_request(resolved_api, message)
        
        if resolved_api == "langchain" and agent is not None:
            latest_dir = getattr(agent, "last_analysis_dir", None)
            if latest_dir:
                global last_analysis_dir
                last_analysis_dir = str(latest_dir)

        await manager.send_message(session_id, {
            "type": "assistant_message",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        await manager.send_message(session_id, {
            "type": "error",
            "content": f"❌ 处理失败: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })


async def dispatch_api_request(api: str, message: str) -> str:
    if api == "langchain":
        if agent:
            return await agent.process_message(message)
        return generate_placeholder_response("langchain", message)
    if api == "nsys":
        return generate_placeholder_response("nsys", message)
    if api == "ncu":
        return generate_placeholder_response("ncu", message)
    if api == "custom":
        return generate_placeholder_response("custom", message)
    if api == "auto":
        # 理论上不会走到这里，但保底处理
        return await dispatch_api_request(resolve_api_selection(api), message)
    return generate_placeholder_response("unknown", message)


def generate_placeholder_response(api: str, message: str) -> str:
    if api == "langchain":
        return f"""✅ 已接收到您的请求

**请求内容**: {message}

📋 **解析结果**:
• 这是一个模拟响应（未加载LangChain Agent）
• 完整功能需要配置SGlang和模型路径
• 请查看 config.yaml 进行配置

💡 **下一步**:
1. 配置 config.yaml 中的路径
2. 确保SGlang已安装
3. 准备好模型文件
4. 重新启动服务

详细配置说明请查看 README.md
"""
    if api == "nsys":
        return f"""🧪 **NSys 性能分析工作流 (示例)**

**请求内容**: {message}

🔧 建议步骤:
1. 使用 `nsys profile` 收集全局性能数据
2. 将生成的 `.nsys-rep` 文件上传至 `analysis_results/`
3. 运行 `utils/nsys_to_ncu_analyzer.py` 获取瓶颈函数
4. 若需要深度分析，切换到 NCU API 继续

📘 详细说明: 请参考 `configs_and_docs/README_AI_Agent.md` 中的 NSys 部分
"""
    if api == "ncu":
        return f"""🔬 **NCU 深度分析 (示例)**

**请求内容**: {message}

🔧 建议步骤:
1. 基于 NSys 输出定位热点 kernel
2. 使用 `ncu --set full` 针对热点 kernel 收集数据
3. 将输出导入 `utils/nsys_to_ncu_analyzer.py` 生成综合报告
4. 对比不同 batch/input 参数下的瓶颈

📘 使用指南: 请参考 `configs_and_docs/README_AI_Agent.md` 中的 NCU 部分
"""
    if api == "custom":
        return f"""🛠️ **自定义工具链工作流**

**请求内容**: {message}

可以在 `backend/` 目录下新增自定义处理逻辑，例如:
• 集成内部监控或告警系统
• 调用企业内部 LangChain 工具
• 结合 FAISS / Chroma 知识库检索

💡 提示: 新增的处理函数可在 `dispatch_api_request` 中注册
"""
    return f"""ℹ️ 当前选择的 API ({api}) 暂未实现定制逻辑。

您可以在 `backend/web_server.py` 中扩展 `dispatch_api_request` 来满足特定需求。

原始请求: {message}
"""

@app.post("/upload_config")
async def upload_config(file: UploadFile = File(...)):
    """上传配置文件"""
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # 解析配置
        if file.filename.endswith('.json'):
            config_data = json.loads(content_str)
        elif file.filename.endswith(('.yaml', '.yml')):
            config_data = yaml.safe_load(content_str)
        else:
            return {"error": "不支持的文件格式"}
        
        return {
            "filename": file.filename,
            "message": "配置文件上传成功",
            "config": config_data
        }
        
    except Exception as e:
        return {"error": f"上传失败: {str(e)}"}

@app.post("/knowledge/upload")
async def upload_knowledge_json(
    request: Request,
    file: UploadFile = File(None),
    raw_json: Optional[str] = Body(None, description="字符串形式的 JSON 或直接传递 JSON 对象"),
    embedding_model: Optional[str] = Body(None, description="可选的嵌入模型名；支持 ms:<model-id> 通过 ModelScope 下载 (例如 ms:damo/nlp_gte-base-zh)"),
    force_tfidf: Optional[bool] = Body(False, description="强制使用 TF-IDF fallback 而不加载任何嵌入模型"),
    segmentation_mode: Optional[str] = Body("window", description="文本切分模式: window(滑窗)/sentence(按句)/auto(本地嵌入自动句子)"),
    request_body: Optional[dict] = Body(None),
    debug: Optional[bool] = Body(False)
):
    """上传并摄取 JSON 知识库到 FAISS

    支持多种提交与嵌入来源:
        1. multipart/form-data 文件上传: file=@xxx.json (+ 可选 embedding_model)
        2. application/json 方式: {"raw_json": "{...}"} 或 {"raw_json": {...}}
        3. 直接提交纯 JSON 对象 (不包裹 raw_json), 例如: {"section": "Intro", "text": "Hello"}
        4. 嵌入模型来源 embedding_model:
           - HuggingFace (默认): sentence-transformers/all-MiniLM-L6-v2 或任意 HF 名称
           - ModelScope: 使用前缀 ms:<model-id> (需 pip install modelscope)，示例 ms:damo/nlp_gte-base-zh
           - TF-IDF fallback: 自动在嵌入模型加载失败时启用，或设置 force_tfidf=true (后续可扩展)

    返回: 索引构建结果统计 (成功) 或错误详情
    错误码:
        400 - 输入缺失或不合法
        500 - 服务内部异常 / 摄取模块未加载
    """
    if ingest_json_to_faiss is None:
        return JSONResponse(status_code=500, content={"error": "知识库摄取模块未加载"})

    try:
        json_str: Optional[str] = None
        diagnostics = {}
        # 优先文件
        if file is not None:
            content_bytes = await file.read()
            try:
                json_str = content_bytes.decode('utf-8')
            except Exception:
                return JSONResponse(status_code=400, content={"error": "文件编码解析失败，需 UTF-8"})
        else:
            # raw_json 参数处理
            if raw_json is not None:
                if isinstance(raw_json, (dict, list)):
                    # FastAPI 可能会把原始 JSON 映射为 dict/list (若参数类型为 Any)
                    try:
                        json_str = json.dumps(raw_json, ensure_ascii=False)
                    except Exception as e:
                        return JSONResponse(status_code=400, content={"error": f"raw_json 序列化失败: {e}"})
                else:
                    # 字符串形式
                    json_str = raw_json
            elif request_body is not None:
                # 未显式提供 raw_json, 直接使用主体对象
                try:
                    json_str = json.dumps(request_body, ensure_ascii=False)
                except Exception as e:
                    return JSONResponse(status_code=400, content={"error": f"请求主体序列化失败: {e}"})
            else:
                # 最后尝试直接读取原始 body (解决未声明字段时直接传对象的情况)
                if request.headers.get("content-type", "").startswith("application/json"):
                    try:
                        raw_bytes = await request.body()
                        if raw_bytes:
                            parsed = json.loads(raw_bytes.decode('utf-8'))
                            json_str = json.dumps(parsed, ensure_ascii=False)
                            diagnostics['fallback_body_parse'] = True
                    except Exception as e:
                        return JSONResponse(status_code=400, content={"error": f"直接主体解析失败: {e}"})

        if not json_str:
            return JSONResponse(status_code=400, content={"error": "缺少 JSON 内容: 请提供 file、raw_json 或直接 JSON 对象", "received_headers": dict(request.headers), "debug": True if debug else None})

        model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        # 环境变量触发强制 TF-IDF (优先级低于明确传参 force_tfidf)
        env_force = os.getenv('OFFLINE_FORCE_TFIDF', '0').lower() in ('1', 'true', 'yes')
        effective_force_tfidf = bool(force_tfidf) or env_force
        # 尝试摄取索引并捕获构建阶段具体错误
        try:
            seg_mode_eff = segmentation_mode or 'window'
            if seg_mode_eff == 'auto':
                if model_name.startswith('local:') or model_name == 'local-simple':
                    seg_mode_eff = 'sentence'
                else:
                    seg_mode_eff = 'window'
            # 本地哈希嵌入默认改为句子模式（若用户未明确指定）
            if segmentation_mode == 'window' and (model_name.startswith('local:') or model_name == 'local-simple'):
                seg_mode_eff = 'sentence'
            result = ingest_json_to_faiss(json_str, embedding_model=model_name, force_tfidf=effective_force_tfidf, segmentation_mode=seg_mode_eff)
        except Exception as ingest_exc:
            # 未在底层函数内被处理的异常
            err_payload = {
                "status": "error",
                "stage": "ingest_call_exception",
                "message": str(ingest_exc),
                "embedding_model": model_name,
                "force_tfidf": effective_force_tfidf,
                "embedding_provider": None,
            }
            return JSONResponse(status_code=500 if not debug else 400, content=err_payload)

        # 若底层返回错误状态, 以 400 显示
        if result.get("status") == "error":
            if debug:
                result['diagnostics'] = diagnostics
                result['embedding_model'] = model_name
                result['force_tfidf'] = effective_force_tfidf
                result['force_tfidf_used'] = effective_force_tfidf
                result['embedding_provider'] = result.get('embedding_provider')
                result['segmentation_mode_used'] = seg_mode_eff
                result['json_length'] = len(json_str)
                try:
                    parsed_tmp = json.loads(json_str)
                    if isinstance(parsed_tmp, dict):
                        result['root_keys'] = list(parsed_tmp.keys())
                except Exception:
                    pass
            return JSONResponse(status_code=400, content=result)
        if debug:
            result['diagnostics'] = diagnostics
            result['content_type'] = request.headers.get('content-type')
            result['embedding_model'] = model_name
            result['force_tfidf'] = effective_force_tfidf
            result['force_tfidf_used'] = effective_force_tfidf or (result.get('embedding_provider') == 'tfidf_fallback')
            result['embedding_provider'] = result.get('embedding_provider')
            result['segmentation_mode_used'] = seg_mode_eff
        # 正常返回也补充 force_tfidf_used 字段
        result['force_tfidf_used'] = effective_force_tfidf or (result.get('embedding_provider') == 'tfidf_fallback')
        result['segmentation_mode_used'] = seg_mode_eff
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"摄取失败: {e}"})

@app.get("/report/full")
async def generate_full_report():
    """生成增强版性能报告 (结合理论知识库)。"""
    try:
        from backend.report_generator import generate_enriched_report
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"报告生成模块不可用: {e}"})
    # 选择目标目录
    target_dir = last_analysis_dir or "/workspace/Agent/AI_Agent_Complete"
    from pathlib import Path
    try:
        enriched = generate_enriched_report(Path(target_dir))
        return {"status": "ok", "enriched_report": enriched, "output_dir": target_dir}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"生成失败: {e}"})

@app.get("/report/advanced")
async def generate_advanced(detailed: bool = False, ncu_metrics: bool = False):
    """生成高阶优化建议报告 (不要求理论索引存在)

    参数:
        detailed: 解析 comprehensive_analysis.json 生成关键指标快照与 granular kernel tasks
        ncu_metrics: 额外尝试解析 NCU CSV 文件 (若存在) 提取 SM Efficiency / Memory Bandwidth
    """
    try:
        from backend.advanced_report import generate_advanced_report
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"模块不可用: {e}"})
    target_dir = last_analysis_dir or "/workspace/Agent/AI_Agent_Complete/sglang_analysis_b8_i512_o64"
    from pathlib import Path
    try:
        path = generate_advanced_report(Path(target_dir), detailed=detailed)
        extra = {}
        if ncu_metrics:
            from pathlib import Path as _P
            import glob, csv
            metrics_list = []
            for csv_file in glob.glob(str(_P(target_dir) / 'ncu_kernel_*/*.csv')):
                # Some visualization dirs; skip for now
                continue
            # flat pattern (top-level)
            for csv_file in glob.glob(str(_P(target_dir) / 'ncu_kernel_*.csv')):
                p = _P(csv_file)
                if p.stat().st_size == 0:
                    continue
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        header = next(reader, [])
                        rows = list(reader)[:5]
                    metrics_list.append({"file": p.name, "header": header, "sample_rows": rows})
                except Exception:
                    pass
            extra['ncu_csv_samples'] = metrics_list
        return {"status": "ok", "advanced_report": path, "output_dir": target_dir, **extra}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"生成失败: {e}"})

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections),
        "agent_ready": agent is not None,
        "config_loaded": CONFIG is not None
    }

@app.get("/config")
async def get_config():
    """获取配置信息"""
    return {
        "sglang_path": CONFIG.get('sglang_path'),
        "models_path": CONFIG.get('models_path'),
        "server": CONFIG.get('server'),
        "model_mappings": CONFIG.get('model_mappings', {})
    }

@app.post("/analyze/sglang")
async def analyze_sglang(req: SGLangAnalyzeRequest):
    """触发一体化 nsys + ncu 性能分析 (SGlang 专用)"""
    if NSysToNCUAnalyzer is None:
        return JSONResponse(status_code=500, content={"error": "分析模块未加载"})
    try:
        # 构建 sglang 命令
        sglang_cmd = [
            'python', '-m', 'sglang.bench_one_batch',
            '--model-path', req.model_path,
            '--batch-size', str(req.batch_size),
            '--input-len', str(req.input_len),
            '--output-len', str(req.output_len),
            '--load-format', 'dummy'
        ]
        analyzer = NSysToNCUAnalyzer(
            f"sglang_analysis_b{req.batch_size}_i{req.input_len}_o{req.output_len}"
        )
        nsys_file = analyzer.step1_nsys_analysis(sglang_cmd, "sglang_overview")
        hot = analyzer.step2_extract_hot_kernels(
            nsys_file, top_k=req.top_k, min_duration_ms=req.min_duration_ms
        )
        full_rep, focus_metrics = analyzer.step3_ncu_global_focus(
            sglang_cmd, hot, top_focus=min(req.max_ncu_kernels, len(hot)), set_name='compute'
        )
        ncu_files = [full_rep] if full_rep else []
        results = analyzer.step4_comprehensive_analysis(ncu_files, focus_metrics=focus_metrics)
        report = analyzer.generate_final_report(results)
        global last_analysis_dir
        last_analysis_dir = str(analyzer.output_dir)
        return {
            "status": "ok",
            "output_dir": str(analyzer.output_dir),
            "report_file": report,
            "hot_kernels": hot[:10],
            "ncu_files": ncu_files,
            "focus_metrics_count": len(focus_metrics),
            "json_kernel_candidates": results.get('nsys_overview', {}).get('kernel_analysis', {}).get('unique_kernels', None)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post('/analysis/submit')
async def submit_analysis(req: AnalysisSubmitRequest):
    """提交异步性能分析作业。"""
    if NSysToNCUAnalyzer is None:
        return JSONResponse(status_code=500, content={'error': '分析模块未加载'})
    job = job_manager.create(req.dict())
    import threading

    def _worker():
        job.status = 'running'
        try:
            analyzer = NSysToNCUAnalyzer(
                f"sglang_analysis_b{req.batch_size}_i{req.input_len}_o{req.output_len}_job_{job.job_id}"
            )
            cmd = [
                'python', '-m', 'sglang.bench_one_batch',
                '--model-path', req.model_path,
                '--batch-size', str(req.batch_size),
                '--input-len', str(req.input_len),
                '--output-len', str(req.output_len),
                '--load-format', 'dummy'
            ]
            if req.allow_remote_code:
                cmd.append('--trust-remote-code')
            nsys_rep = analyzer.step1_nsys_analysis(cmd, 'overview')
            hot = analyzer.step2_extract_hot_kernels(nsys_rep, top_k=req.top_k, min_duration_ms=req.min_duration_ms)
            full_rep, focus_metrics = analyzer.step3_ncu_global_focus(cmd, hot, top_focus=min(req.max_ncu_kernels, len(hot)), set_name='compute')
            ncu_files = [full_rep] if full_rep else []
            comp = analyzer.step4_comprehensive_analysis(ncu_files, focus_metrics=focus_metrics)
            base_report = analyzer.generate_final_report(comp)
            # 高阶报告
            if req.advanced:
                try:
                    from backend.advanced_report import generate_advanced_report
                    generate_advanced_report(analyzer.output_dir, detailed=req.advanced_detailed)
                except Exception as _e:
                    print(f"高级报告失败: {_e}")
            # 增强报告
            if req.generate_enriched:
                try:
                    from backend.report_generator import generate_enriched_report
                    generate_enriched_report(analyzer.output_dir, comprehensive=comp)
                except Exception as _e:
                    print(f"增强报告失败: {_e}")
            job.output_dir = str(analyzer.output_dir)
            job.artifacts = {
                'hot_kernels': hot,
                'ncu_files': ncu_files,
                'focus_metrics_count': len(focus_metrics),
                'base_report': base_report
            }
            job.status = 'done'
        except Exception as e:
            job.status = 'error'
            job.error = str(e)
        finally:
            job.finished_at = datetime.now().isoformat()
    threading.Thread(target=_worker, daemon=True).start()
    return {'job_id': job.job_id, 'status': job.status}

@app.get('/analysis/status/{job_id}')
async def analysis_status(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={'error': 'job not found'})
    return {
        'job_id': job.job_id,
        'status': job.status,
        'error': job.error,
        'started_at': job.started_at,
        'finished_at': job.finished_at,
        'output_dir': job.output_dir
    }

@app.post('/analysis/final_report')
async def final_report(req: FinalReportRequest):
    job = job_manager.get(req.job_id)
    if not job:
        return JSONResponse(status_code=404, content={'error': 'job not found'})
    if job.status != 'done':
        return JSONResponse(status_code=409, content={'error': f'job not finished: {job.status}'})
    from pathlib import Path as _P
    try:
        from backend.langchain_synthesis import synthesize_final_report
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': f'无法加载综合生成模块: {e}'})
    perf_dir = _P(job.output_dir)
    result = synthesize_final_report(perf_dir, extra_query_text=req.extra_query)
    return {
        'job_id': req.job_id,
        'final_report': result['markdown_path'],
        'summary': result['summary'],
        'kb_queries': list(result['kb_hits'].keys()),
        'model_info': result.get('model_info', {})
    }

@app.post("/analysis/full")
async def full_analysis(req: FullAnalysisRequest):
    """一键端到端分析: nsys + ncu + 基础报告 + 高阶报告 + (可选) enriched + (可选) KB 摄取

    前端只需提交模型与批次配置，即返回所有产物路径与关键指标。
    """
    if NSysToNCUAnalyzer is None:
        return JSONResponse(status_code=500, content={"error": "分析模块未加载"})
    try:
        sglang_cmd = [
            'python', '-m', 'sglang.bench_one_batch',
            '--model-path', req.model_path,
            '--batch-size', str(req.batch_size),
            '--input-len', str(req.input_len),
            '--output-len', str(req.output_len),
            '--load-format', 'dummy'
        ]
        if req.allow_remote_code:
            sglang_cmd.append('--trust-remote-code')
        analyzer = NSysToNCUAnalyzer(
            f"sglang_analysis_b{req.batch_size}_i{req.input_len}_o{req.output_len}"
        )
        # Step1-4
        nsys_file = analyzer.step1_nsys_analysis(sglang_cmd, "sglang_overview")
        hot = analyzer.step2_extract_hot_kernels(
            nsys_file, top_k=req.top_k, min_duration_ms=req.min_duration_ms
        )
        full_rep, focus_metrics = analyzer.step3_ncu_global_focus(
            sglang_cmd,
            hot,
            top_focus=min(req.max_ncu_kernels, len(hot)),
            set_name='compute'
        )
        ncu_files = [full_rep] if full_rep else []
        results = analyzer.step4_comprehensive_analysis(
            ncu_files, focus_metrics=focus_metrics
        )
        base_report = analyzer.generate_final_report(results)
        # Advanced report
        advanced_report_path = None
        advanced_json = None
        if req.advanced:
            try:
                from backend.advanced_report import generate_advanced_report
                advanced_report_path = generate_advanced_report(analyzer.output_dir, detailed=req.advanced_detailed)
                if req.advanced_json:
                    # 复用 analyzer 中的辅助函数 (已在 utils 脚本里定义)，此处轻量 re-import
                    from backend.utils.nsys_to_ncu_analyzer import _extract_advanced_json  # type: ignore
                    md_text = Path(advanced_report_path).read_text(encoding='utf-8')
                    advanced_json = _extract_advanced_json(md_text)
                    json_path = analyzer.output_dir / 'advanced_performance_report.json'
                    json_path.write_text(json.dumps(advanced_json, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception as e:
                print(f"⚠️ 高阶报告生成失败: {e}")
        # Enriched report (FAISS support)
        enriched_path = None
        if req.generate_enriched:
            try:
                from backend.report_generator import generate_enriched_report
                enriched_path = generate_enriched_report(analyzer.output_dir, comprehensive=results)
            except Exception as e:
                print(f"⚠️ 增强报告生成失败: {e}")
        # Optional KB ingestion
        kb_ingest_status = None
        if req.ingest_advanced and advanced_json:
            try:
                from backend.knowledge_bases.kb_ingest import ingest_json_to_faiss
                ingest_json_to_faiss(json.dumps(advanced_json, ensure_ascii=False), embedding_model="sentence-transformers/all-MiniLM-L6-v2", kb_path=req.kb_path or 'knowledge_store')
                kb_ingest_status = 'ok'
            except Exception as e:
                kb_ingest_status = f'failed: {e}'
        # Metrics aggregation
        metrics = {}
        try:
            from backend.perf_data_parser import aggregate_metrics
            metrics = aggregate_metrics({**results, 'hot_kernels': hot})
        except Exception as e:
            print(f"⚠️ 指标聚合失败: {e}")
        global last_analysis_dir
        last_analysis_dir = str(analyzer.output_dir)
        return {
            'status': 'ok',
            'output_dir': str(analyzer.output_dir),
            'base_report': base_report,
            'advanced_report': advanced_report_path,
            'enriched_report': enriched_path,
            'advanced_json_excerpt': advanced_json.get('summary') if isinstance(advanced_json, dict) else None,
            'hot_kernels_top': hot[:10],
            'ncu_files': ncu_files,
            'focus_metrics_count': len(focus_metrics),
            'metrics': metrics,
            'kb_ingest_status': kb_ingest_status,
            'note': req.note
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

async def _summarize_report_to_table(report_path: Path) -> str:
    if not report_path.exists():
        raise FileNotFoundError(f"报告缺失: {report_path}")
    report_text = report_path.read_text(encoding='utf-8')
    loop = asyncio.get_running_loop()
    client = get_offline_qwen_client(OFFLINE_QWEN_PATH)
    return await loop.run_in_executor(None, client.report_to_table, report_text)

@app.get("/report/table")
async def report_table():
    if agent is not None:
        table_md = getattr(agent, "last_analysis_table", None)
        if table_md:
            report_paths = getattr(agent, "last_analysis_reports", None)
            response = {"status": "ok", "table_markdown": table_md}
            if report_paths:
                response["report_paths"] = report_paths
                response["report_path"] = report_paths[0]
            return response

    if not last_analysis_dir:
        return JSONResponse(status_code=404, content={"error": "暂无分析结果"})
    report_path = Path(last_analysis_dir) / "integrated_performance_report.md"
    try:
        table_md = await _summarize_report_to_table(report_path)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"status": "ok", "table_markdown": table_md, "report_path": str(report_path)}

if __name__ == "__main__":
    # 获取配置
    host = CONFIG.get('server', {}).get('host', '0.0.0.0')
    port = CONFIG.get('server', {}).get('port', 8000)
    
    # 启动服务
    uvicorn.run(
        "web_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

