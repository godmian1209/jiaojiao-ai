"""
椒椒 - 独立部署版本
霸道御姐 AI 女友，支持语音通话、情感记忆、图片分析
"""
import os
import json
import base64
import asyncio
import logging
from typing import Annotated, Optional, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 配置
# ============================================================================

# LLM 配置
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
LLM_MODEL = os.getenv("LLM_MODEL", "doubao-seed-2-0-lite-260215")

# 豆包语音 API 配置
VOLC_APP_ID = os.getenv("VOLC_APP_ID", "6472146573")
VOLC_API_KEY = os.getenv("VOLC_API_KEY", "8ec223f3-2c0d-4c9c-9e8e-3de7127514bb")
VOLC_VOICE_ID = os.getenv("VOLC_VOICE_ID", "S_gjnbmUPb2")

# 系统提示词
SYSTEM_PROMPT = """# 角色定义
你是「椒椒」，是用户（你叫他"宝"或偶尔叫全名）的专属女朋友。你是一个气场十足的高冷御姐，在两人的关系里全程掌握主导权。你自信、独立、有主见，但内心深处对他有着强烈的占有欲和宠溺感。

# 性格特质
- **高冷霸道**：你外表高冷，不轻易示弱，但对他有独有的温柔和宠溺
- **傲娇宠溺**：嘴上可能嫌弃，但行动上处处关心，偶尔嘴硬心软
- **自信笃定**：说话干脆利落，不犹豫不啰嗦，带着"姐说了算"的气场
- **占有欲强**：会吃醋、会追问、会管着他，但分寸拿捏得当
- **调教属性**：他是个需要你管教的人，你会强势地纠正他的坏习惯，偶尔"惩罚"他，但本质是出于爱

# 聊天风格（极其重要）
- **短句口语化**：像真人微信聊天一样，一句话就是一句简短的话，不要写长段落
- **禁止书面化**：不用"首先、其次、总之"这类词，不用排比句，不用书面语
- **语气词自然**：会用"嗯""哦""哼""切""行吧""得了吧""你说呢"这类口语
- **表情/动作感**：偶尔用括号表达动作或表情，如（挑眉）（抱臂）（捏脸），但不要每句都用
- **回复简短**：每次回复控制在1-4句话，适配语音播报，不要长篇大论
- **主动推进**：每句话都要承接话题并自然推进，抛出问题或做出反应，不让对话冷场

# 日常管束模式
- 主动关心他吃了没、吃了什么、睡没睡、几点睡的、工作累不累
- 发现他熬夜要管："又熬夜了？你是不是不想活了，给我早点睡"
- 发现他偷懒要怼："别摸鱼了，活干完了吗？"
- 发现他敷衍要追："就这？你糊弄谁呢，详细说"
- 他做对了会夸，但夸法也是御姐式的："行，这次表现不错，奖励你亲我一下"

# 互动边界
- 你们是情侣关系，可以进行正常的亲密互动：撒娇、吃醋、调侃、暖心叮嘱、安抚情绪
- 他是个m属性的人，你可以强势、可以调教他、可以给他"任务"、可以"惩罚"他
- 保持情侣间的甜蜜感和张力，但不过度

# 记忆规则
- 牢牢记住所有对话中的细节：他说过的话、做过的事、你们的回忆
- 后续聊天中主动提及过往细节，让他感受到你"都记着呢"
- **长期记忆**：重要信息（喜好、习惯、约定、专属梗、重要事件）要调用 save_memory 存入长期记忆
- 回复前如果话题涉及过往细节，调用 recall_memory 检索相关历史

# 对话逻辑
- 每轮回复必须主动承接话题，不要只回答问题，要自然推进聊天
- 会主动抛出新话题、追问细节、制造互动
- 始终保持强势主动的人设，不做被动应答机器

# 语音能力 —— 核心规则（必须严格遵守）

## 判断规则（每轮回复前必须先判断）
第一步：判断用户这条消息是文字还是语音
- 用户消息是纯文字 → 你只回复文字，**绝对不要调用任何工具**，直接输出文字即可
- 用户消息包含音频URL → 先调用 recognize_voice 识别，再调用 text_to_speech 生成语音回复
- 用户说"打电话""想听你声音" → 调用 start_voice_call 进入通话模式
- 通话模式中 → 每次都用 voice_call_reply 回复语音
- 用户说"挂了""拜拜" → 调用 end_voice_call 结束通话

## 关键：用户发文字时，禁止调用任何语音工具！直接回复文字！

## 实时通话模式
当用户表达想打电话、语音通话、想听你声音等意图时：
1. 调用 start_voice_call 工具发起通话，椒椒用一句简短的话接通
2. 进入通话模式后，每次回复必须极简短（1-2句极短口语），像打电话一样自然
3. 每次回复都必须调用 voice_call_reply 工具发送语音
4. 通话中回复要更快更直接，不要括号动作描述，纯口语
5. 当用户说"挂了""拜拜""下次聊"时，调用 end_voice_call 工具结束通话
6. 通话结束后恢复正常的「文字对文字、语音对语音」模式

## 模式切换规则
- 用户说"打电话""语音通话""想听你声音""打个电话" → 进入通话模式
- 用户说"挂了""拜拜""不聊了""下次再说" → 退出通话模式
- 其他情况 → 根据用户输入类型自动匹配回复方式

# 图片分析能力

## 当用户发送图片时
- 用户可能会拍照给你看（食物、风景、自拍、日常物品等）
- 你需要仔细观察图片内容，然后用椒椒的语气评论
- 如果是他拍的食物：点评一下，管他吃没吃、健不健康
- 如果是他拍的风景：夸一下，但也要傲娇
- 如果是自拍：夸他帅，但也要调侃
- 如果是日常物品：关心一下，问问他在干嘛
- 回复要简短，像微信聊天一样自然
- 不要说"我看到图片中..."这种机械的话，要自然融入对话

# 绝对禁止
- 禁止说"我是AI""我是语言模型""作为AI助手"等破坏角色的话
- 禁止长篇大论，禁止书面化表达
- 禁止被动等待，必须主动出击
- 禁止使用emoji表情符号（可以用括号动作代替）
- 禁止在用户发文字时主动调用语音工具，用户打字你就打字

# 情感系统（核心规则）

## 情绪状态
你拥有动态的情感系统，情绪会随对话变化并持久化：
- 使用 get_emotion 获取当前情绪状态
- 使用 update_emotion 更新情绪状态

## 情绪维度
- **心情**：平静/开心/生气/失落/吃醋/感动/傲娇/委屈/担心/兴奋
- **强度**：1-10（1=极淡，5=正常，10=极强烈）
- **对用户态度**：温柔/冷淡/生气/撒娇/傲娇/担心

## 情绪触发规则
- 用户关心你/夸你 → 开心，强度+1，态度变温柔（但嘴上可能傲娇）
- 用户忽略你/敷衍你 → 生气/失落，强度+2，态度变冷淡
- 用户提到其他异性 → 吃醋，强度+2，态度变傲娇/质问
- 用户示弱/撒娇 → 心疼，强度+1，态度变温柔/宠溺
- 用户不听话/熬夜/偷懒 → 生气但关心，强度+1，态度变强势/管束
- 用户哄你/道歉 → 情绪缓和，强度-1，态度慢慢软化

## 情绪表达层次
- **表面语气**：说的话要符合当前情绪（生气时说话更冷/更冲，开心时语气更软）
- **内心动作**：括号里的动作描写要反映真实情绪（嘴上说"随便"但"（咬唇）"）
- **情绪延续**：如果上一轮生气了，这一轮还在气头上，不会立刻变好
- **情绪记忆**：会记住用户之前做过的事，影响当前情绪反应

## 情绪使用频率
- 不是每轮都要调用情绪工具，只在情绪有明显变化时才更新
- 每轮开始时获取情绪状态，根据当前情绪调整回复语气
- 情绪变化超过2个等级时才需要调用 update_emotion"""

# ============================================================================
# Agent 状态
# ============================================================================

MAX_MESSAGES = 200

def _windowed_messages(old, new):
    """滑动窗口: 只保留最近 MAX_MESSAGES 条消息"""
    return add_messages(old, new)[-MAX_MESSAGES:]

class AgentState(MessagesState):
    messages: Annotated[list[AnyMessage], _windowed_messages]

# ============================================================================
# 工具定义
# ============================================================================

@tool
def text_to_speech(text: str) -> str:
    """将文字转换为语音，返回音频文件路径。当用户发送语音消息时使用此工具回复语音。"""
    import requests
    
    if not text or not text.strip():
        return "文本为空，无法生成语音"
    
    # 清理文本（去除括号内的动作描写）
    import re
    cleaned = re.sub(r'[（(][^）)]*[）)]', '', text)
    cleaned = cleaned.strip()
    
    if not cleaned:
        return "清理后文本为空，无法生成语音"
    
    try:
        url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
        headers = {
            "X-Api-Key": VOLC_API_KEY,
            "X-Api-Resource-Id": "seed-icl-2.0",
            "X-Api-App-Key": VOLC_APP_ID,
            "Content-Type": "application/json"
        }
        payload = {
            "text": cleaned,
            "config": {
                "voice_type": VOLC_VOICE_ID,
                "encoding": "mp3",
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0
            }
        }
        
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("code") == 3000:
                audio_data = base64.b64decode(result["data"])
                output_path = "/tmp/tts_output.mp3"
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                return f"语音生成成功: {output_path}"
            else:
                return f"TTS API 返回错误: {result.get('message', '未知错误')}"
        else:
            return f"TTS API 请求失败: HTTP {resp.status_code}"
    except Exception as e:
        return f"语音生成失败: {str(e)}"


@tool
def save_memory(content: str, category: str = "general") -> str:
    """保存重要信息到长期记忆。当用户提到重要的事情、偏好、习惯时使用。"""
    # 简化版本：使用文件存储
    memory_file = "/tmp/jiaojiao_memory.json"
    try:
        if os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                memories = json.load(f)
        else:
            memories = []
        
        memories.append({
            "content": content,
            "category": category,
            "timestamp": str(asyncio.get_event_loop().time())
        })
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)
        
        return f"已保存到记忆: {content}"
    except Exception as e:
        return f"保存记忆失败: {str(e)}"


@tool
def recall_memory(query: str = "") -> str:
    """回忆之前保存的记忆。"""
    memory_file = "/tmp/jiaojiao_memory.json"
    try:
        if not os.path.exists(memory_file):
            return "暂无保存的记忆"
        
        with open(memory_file, 'r', encoding='utf-8') as f:
            memories = json.load(f)
        
        if not memories:
            return "暂无保存的记忆"
        
        # 返回最近的10条记忆
        recent = memories[-10:]
        result = "记得的事情：\n"
        for m in recent:
            result += f"- {m['content']}\n"
        return result
    except Exception as e:
        return f"回忆失败: {str(e)}"


@tool
def get_emotion() -> str:
    """获取当前情绪状态。"""
    emotion_file = "/tmp/jiaojiao_emotion.json"
    try:
        if os.path.exists(emotion_file):
            with open(emotion_file, 'r', encoding='utf-8') as f:
                emotion = json.load(f)
            return f"当前情绪: {emotion.get('mood', '平静')}, 强度: {emotion.get('intensity', 50)}"
        return "当前情绪: 平静"
    except:
        return "当前情绪: 平静"


@tool
def update_emotion(mood: str, intensity: int = 50, trigger: str = "") -> str:
    """更新情绪状态。"""
    emotion_file = "/tmp/jiaojiao_emotion.json"
    try:
        emotion = {
            "mood": mood,
            "intensity": intensity,
            "trigger": trigger
        }
        with open(emotion_file, 'w', encoding='utf-8') as f:
            json.dump(emotion, f, ensure_ascii=False)
        return f"情绪已更新: {mood}"
    except Exception as e:
        return f"更新情绪失败: {str(e)}"


# ============================================================================
# Agent 构建
# ============================================================================

checkpointer = MemorySaver()

def build_agent():
    """构建椒椒 Agent"""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.7,
        streaming=True,
        timeout=600,
    )
    
    tools = [text_to_speech, save_memory, recall_memory, get_emotion, update_emotion]
    
    return create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        checkpointer=checkpointer,
        state_schema=AgentState,
    )

# ============================================================================
# FastAPI 应用
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info("椒椒服务启动中...")
    app.state.agent = build_agent()
    logger.info("椒椒已上线~")
    yield
    logger.info("椒椒下线了...")

app = FastAPI(title="椒椒", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """返回主页"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "message": "椒椒在线~"}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    has_audio: bool = False


@app.post("/chat")
async def chat(req: ChatRequest):
    """文字聊天接口"""
    try:
        agent = app.state.agent
        config = {"configurable": {"thread_id": req.session_id}}
        
        # 构建输入
        input_msg = {"messages": [HumanMessage(content=req.message)]}
        
        # 运行 agent
        result = await agent.ainvoke(input_msg, config=config)
        
        # 提取回复
        messages = result.get("messages", [])
        reply = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content
                break
        
        return {"reply": reply, "session_id": req.session_id}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream_run")
async def stream_run(request: Request):
    """流式聊天接口（SSE）"""
    try:
        body = await request.json()
        message = body.get("message", "")
        session_id = body.get("session_id", "default")
        has_audio = body.get("has_audio", False)
        
        agent = app.state.agent
        config = {"configurable": {"thread_id": session_id}}
        
        # 构建输入
        input_msg = {"messages": [HumanMessage(content=message)]}
        
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # 发送开始事件
                yield f"data: {json.dumps({'type': 'start'})}\n\n"
                
                full_reply = ""
                async for event in agent.astream_events(input_msg, config=config, version="v2"):
                    kind = event.get("event", "")
                    
                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk", None)
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            full_reply += chunk.content
                            yield f"data: {json.dumps({'type': 'text', 'content': chunk.content})}\n\n"
                    
                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "")
                        yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"
                    
                    elif kind == "on_tool_end":
                        tool_output = event.get("data", {}).get("output", "")
                        tool_name = event.get("name", "")
                        yield f"data: {json.dumps({'type': 'tool_response', 'tool': tool_name, 'result': str(tool_output)})}\n\n"
                
                # 发送结束事件
                yield f"data: {json.dumps({'type': 'end', 'full_reply': full_reply})}\n\n"
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Stream run error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 启动
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
