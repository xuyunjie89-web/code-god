"""
main.py — 多模态电商导购 RAG 服务
===================================
流水线: 用户请求 → [图片:Chinese-CLIP] + [文本:BGE-M3] → 特征融合 → ChromaDB检索 → DeepSeek导购文案 → 返回前端

启动方式:
  1. 设置环境变量: set DEEPSEEK_API_KEY=sk-your-key
  2. 启动服务: uvicorn main:app --reload --host 0.0.0.0 --port 8000
  3. API文档: http://localhost:8000/docs
"""

import io, json, os
from contextlib import asynccontextmanager

import torch
import chromadb
from PIL import Image
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer, AutoModel

# === 全局配置 ====================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.6  # 图文融合权重: 0.6偏向图片, 0.4偏向文本

# 预加载的模型实例 (在 lifespan 中初始化)
clip_model = clip_processor = None   # Chinese-CLIP: 图片 → 512维向量
bge_model = bge_tokenizer = None     # BGE-M3: 文本 → 1024维向量, 取前512维对齐
collection = None                    # ChromaDB 商品向量库
llm_client = None                    # DeepSeek 大模型客户端


# === FastAPI 生命周期: 启动时加载所有模型 =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 启动时自动执行: 加载模型 → 连接数据库 → 灌入示例数据
    yield 之前是启动逻辑, yield 之后是关闭逻辑(此处无需清理)
    """
    global clip_model, clip_processor, bge_model, bge_tokenizer, collection, llm_client

    # ---- 1. 加载 Chinese-CLIP 图片编码器 ----
    print("正在加载模型...")
    clip_processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    clip_model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    clip_model.eval()  # 推理模式, 关闭 Dropout/BN

    # ---- 2. 加载 BGE-M3 文本编码器 ----
    bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    bge_model = AutoModel.from_pretrained("BAAI/bge-m3")
    bge_model.eval()

    # ---- 3. 连接 ChromaDB 本地向量库 ----
    chroma_client = chromadb.PersistentClient(path=os.path.join(MODEL_DIR, "chroma_db"))
    collection = chroma_client.get_or_create_collection(
        name="products",
        metadata={"hnsw:space": "cosine"},  # 余弦相似度检索
    )

    # ---- 4. 初始化 DeepSeek 客户端 ----
    # API Key 从环境变量读取, 未设置则跳过 LLM 导购功能
    llm_client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    print("系统初始化完成")

    # ---- 5. 首次运行自动灌入示例商品, 方便测试 ----
    if collection.count() == 0:
        print("检测到空库，正在灌入示例商品...")
        test_img_path = os.path.join(MODEL_DIR, "OIP.jpg")
        if os.path.isfile(test_img_path):
            img = Image.open(test_img_path).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                img_emb = clip_model.get_image_features(**inputs)[0].tolist()
            text_emb = [0.0] * 512  # 无文本描述时用零向量占位
            fused = [ALPHA * i + (1 - ALPHA) * t for i, t in zip(img_emb, text_emb)]
            collection.add(
                ids=["prod_001"],
                embeddings=[fused],
                metadatas=[{
                    "title": "法式复古碎花连衣裙",
                    "price": 189.0,
                    "category": "女装/连衣裙",
                }],
            )
            print("示例商品入库完成")

    yield  # 服务运行期间停在此处, 关闭时继续往下执行


app = FastAPI(title="AI多模态电商导购 API", lifespan=lifespan)


# === 特征提取工具函数 ==============================================

def _img_embed(img: Image.Image) -> list[float]:
    """Chinese-CLIP 编码图片 → 512维浮点向量"""
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        return clip_model.get_image_features(**inputs)[0].tolist()


def _text_embed(text: str) -> list[float]:
    """BGE-M3 编码文本 → 取[CLS]向量前512维 (与图片维度对齐)"""
    inputs = bge_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        # last_hidden_state[:, 0, :] = [CLS] token 表示整个句子的语义
        return bge_model(**inputs).last_hidden_state[:, 0, :][0][:512].tolist()


def _fuse(img_emb: list[float], text_emb: list[float]) -> list[float]:
    """加权融合图文特征: alpha*图片 + (1-alpha)*文本"""
    return [ALPHA * i + (1 - ALPHA) * t for i, t in zip(img_emb, text_emb)]


# === API 接口 =====================================================

@app.post("/api/products")
def add_product(
    product_id: str = Form(...),      # 商品唯一ID
    title: str = Form(...),           # 商品标题
    price: float = Form(...),         # 售价
    category: str = Form(...),        # 分类
    description: str = Form(""),      # 详情描述(可选)
    image: UploadFile = File(...),    # 商品图片
):
    """商品入库: 上传图片+信息 → 提取特征 → 写入 ChromaDB"""
    img_bytes = image.file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_emb = _img_embed(img)
    text_emb = _text_embed(f"{title} {description}")
    fused = _fuse(img_emb, text_emb)
    collection.add(
        ids=[product_id],
        embeddings=[fused],
        metadatas=[{"title": title, "price": price, "category": category}],
    )
    return {"status": "ok", "id": product_id, "total": collection.count()}


@app.post("/api/search")
async def search(
    text: str = Form(""),             # 用户文本描述 (如 "便宜点的类似款")
    budget: float = Form(0),          # 预算上限, 0表示不限
    image: UploadFile = File(None),   # 用户上传的图片 (可选)
):
    """
    核心检索接口:
    1. 提取图文特征 → 融合 → 向量检索
    2. 按预算过滤 → 返回 Top-5 候选
    3. 调用 DeepSeek 生成导购文案
    """
    # 步骤1: 提取特征
    img_emb = _img_embed(Image.open(io.BytesIO(await image.read())).convert("RGB")) if image and image.filename else [0.0] * 512
    text_emb = _text_embed(text) if text else [0.0] * 512
    fused = _fuse(img_emb, text_emb)

    # 步骤2: ChromaDB 向量检索 + 价格过滤
    where = {"price": {"$lte": budget}} if budget > 0 else None
    results = collection.query(query_embeddings=[fused], n_results=5, where=where)

    # 步骤3: 整理候选商品列表
    candidates = []
    for i in range(len(results["ids"][0])):
        candidates.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "price": results["metadatas"][0][i]["price"],
            "distance": round(results["distances"][0][i], 4),  # 余弦距离, 越小越相似
        })

    if not candidates:
        return {"candidates": [], "guide": None}

    # 步骤4: 调用 DeepSeek 大模型生成导购文案 (仅当 API Key 已配置)
    guide = None
    if llm_client.api_key:
        prompt = (
            f"你是专业电商导购。用户需求: {text if text else '图片搜索'}。"
            f"候选商品: {json.dumps(candidates, ensure_ascii=False)}。"
            f"请: 1.选出最合适的Top推荐 2.对每个商品写一句导购理由(不超过30字) 3.给出整体选购建议。"
            f'返回JSON: {{"products":[{{"id","title","price","reason"}}],"reasoning":"分析","tip":"建议"}}'
        )
        try:
            resp = llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个输出 JSON 格式的专业 AI 导购。"},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},  # 强制 JSON 输出, 便于前端解析
            )
            guide = json.loads(resp.choices[0].message.content)
        except Exception:
            guide = None  # LLM 调用失败不影响检索结果返回

    return {"candidates": candidates, "guide": guide}


@app.get("/api/health")
def health():
    """健康检查: 返回当前库中商品总数"""
    return {"status": "ok", "products_count": collection.count()}
