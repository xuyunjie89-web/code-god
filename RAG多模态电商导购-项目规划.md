# RAG 多模态电商导购 — 从规划到落地

> 字节跳动 AI 全栈挑战赛 · 课题 4  
> 时间：2026.5.20 — 6.10（3 周）  
> 规模：1-3 人，独立评比

---

## 一、课题理解

### 1.1 赛题本质

**用户侧**：电商场景下，用户看到一张图（街拍、截图、实物照），想找相似或搭配商品，同时需要 AI 理解他的偏好，给出有说服力的导购文案。

**技术侧**：图片 → 多模态 Embedding → 向量检索 → 候选商品 → LLM 重排 + 生成导购理由 → 返回结构化结果。

### 1.2 与其他课题的差异

不是纯文本 RAG，不是纯闲聊导购 Bot。核心难点在**图文模态对齐**——同样的 Embedding 空间里，「黑色连衣裙」的文本描述和用户拍的裙子照片，要能召回同一批商品。

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      前端（AI 生成）                      │
│  上传图片 / 输入描述 → 展示商品卡 + 导购文案 + 理由          │
└────────────────────┬────────────────────────────────────┘
                     │ POST /api/search
┌────────────────────▼────────────────────────────────────┐
│                    FastAPI 后端                          │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐    │
│  │ 图片编码  │   │ 文本编码  │   │  混合检索引擎      │    │
│  │ CLIP-ViT │   │ BGE-M3   │   │  Dense + Sparse   │    │
│  └────┬─────┘   └────┬─────┘   └────────┬─────────┘    │
│       │              │                  │               │
│       └──────────────┴──────────────────┘               │
│                     │                                   │
│              ┌──────▼──────┐                            │
│              │  ChromaDB   │  ← 商品库 Embedding 离线写入 │
│              │  向量 + 元数据│                             │
│              └──────┬──────┘                            │
│                     │ Top-K 候选                          │
│              ┌──────▼──────┐                            │
│              │  LLM 重排    │  ← DeepSeek API            │
│              │  生成导购文案 │                             │
│              └──────┬──────┘                            │
│                     │                                   │
│              返回 JSON：商品卡 + 理由 + 搭配建议            │
└─────────────────────────────────────────────────────────┘
```

### 2.1 数据流

```
离线：商品数据(图+文) → CLIP 图片 Embedding + BGE-M3 文本 Embedding
                        → 写入 ChromaDB collection（id, embedding, metadata）

在线：用户输入(图/文) → Image Encoder / Text Encoder
                        → 向量混合检索（图搜 + 文搜 加权）
                        → Top-20 候选
                        → LLM Rerank + 导购理由生成
                        → 返回 Top-5 结构化结果
```

### 2.2 为什么用混合检索

| 方式           | 优势     | 劣势      |
| ------------ | ------ | ------- |
| 纯图搜 (CLIP)   | 视觉相似精准 | 文字描述搜不到 |
| 纯文搜 (BGE-M3) | 语义检索强  | 忽略视觉特征  |
| **混合加权**     | 图+文互补  | 调超参     |

公式：`score = α × image_similarity + (1-α) × text_similarity`

α 建议初始 0.6（图搜为主，图文互补），根据测试调整。

---

## 三、技术栈

| 层                | 选型                                    | 原因                                |
| ---------------- | ------------------------------------- | --------------------------------- |
| **图片 Encoder**   | CLIP-ViT-B/32 (OpenAI) 或 Chinese-CLIP | 中文电商考虑后者，英文数据集大则前者                |
| **文本 Encoder**   | BGE-M3 (BAAI/bge-m3)                  | 多语言，8192 token，MTEB 榜单 top        |
| **向量库**          | ChromaDB                              | 你已熟练，Python 原生，支持 metadata filter |
| **LLM 推理**       | DeepSeek API (deepseek-chat)          | 便宜，中文好，你已有 key                    |
| **后端框架**         | FastAPI + uvicorn                     | 轻量，async 支持好                      |
| **前端**           | AI Coding 工具生成 (TRAE)                 | 赛制鼓励，非核心不手写                       |
| **图片预处理**        | Pillow + torchvision                  | 标准方案                              |
| **Embedding 推理** | ONNX Runtime / PyTorch                | 按需切换，ONNX 更快                      |


### 3.1 模型选型决策

```
Chinese-CLIP  vs  OpenAI CLIP
├── 中文电商数据 → Chinese-CLIP（阿里达摩院，中文图文对预训练）
├── 英文/通用 → OpenAI CLIP-ViT-B/32（更大规模训练）
└── 建议：先用 Chinese-CLIP，商品库是中文时检索质量明显更好
```

---

## 四、三周执行计划

### Week 1: 核心链路跑通（5.20-5.26）

| 天        | 任务                                                        | 产出                                                              | 优先级 |
| -------- | --------------------------------------------------------- | --------------------------------------------------------------- | --- |
| **D1-2** | 环境搭建：FastAPI + ChromaDB + CLIP + BGE-M3 装通                | `requirements.txt`，各模块 import 成功                                | P0  |
| **D3-4** | 商品数据准备：找公开数据集（Fashion200k / DeepFashion / 淘宝商品图），清洗为统一格式  | `products.jsonl`：{id, image_path, title, desc, price, category} | P0  |
| **D5**   | 离线 Embedding 脚本：遍历商品库 → CLIP 编码图 + BGE 编码文本 → 写入 ChromaDB | `scripts/build_index.py` 跑通                                     | P0  |
| **D6**   | 在线检索 API：接收图片/文本 → 编码 → ChromaDB query → 返回 Top-20        | `GET /api/search` 可用                                            | P0  |
| **D7**   | 缓冲 + 端到端调通：上传图片 → 返回候选商品列表                                | 录屏 10s 演示                                                       | P0  |

**Week 1 检查点**：一张街拍图进去，能返回 20 个商品，肉眼判断前 5 个是否相关。

### Week 2: 质量提升 + LLM 生成（5.27-6.2）

| 天          | 任务                                                              | 产出            | 优先级 |
| ---------- | --------------------------------------------------------------- | ------------- | --- |
| **D8**     | 混合检索：图搜 + 文搜加权融合，调 α                                            | 检索准确率提升       | P0  |
| **D9**     | LLM Rerank：写 Prompt，输入 Top-20 候选 + 用户 query → 重排 Top-5 + 生成导购理由 | Rerank 效果验收   | P0  |
| **D10**    | metadata filter：价格区间、品类筛选（先检索再加过滤 vs 先过滤再检索）                    | `filter` 参数可用 | P1  |
| **D11**    | 多轮对话：用户追问「有便宜点的吗？」→ 保留上下文，重新筛选                                  | 对话历史管理        | P1  |
| **D12-13** | 前端搭建（AI 生成）：上传页 + 结果展示页 + 对话区                                   | 可交互的前端        | P1  |
| **D14**    | **中期评审自测**：完整流程走通，录 Demo                                        | 3 分钟演示视频      | P0  |

**Week 2 检查点**：图片 → 商品 + 导购文案 → 追问 → 更新推荐，全链路闭环。

### Week 3: 打磨 + 答辩（6.3-6.10）

| 天          | 任务                                   | 产出   | 优先级 |
| ---------- | ------------------------------------ | ---- | --- |
| **D15-16** | 评测：构造 50 条测试 query（图+文），计算 Top-5 准确率 | 自评报告 | P0  |
| **D17**    | 性能优化：ONNX 导出 CLIP，推理加速；ChromaDB 索引调优 | 延迟降低 | P1  |
| **D18**    | 边界处理：无结果时降级推荐；大图压缩；错误兜底              | 鲁棒性  | P1  |
| **D19-20** | **答辩准备**：PPT + Demo 视频 + 技术文档 + 架构图  | 答辩材料 | P0  |
| **D21**    | 提交                                   |      | P0  |

---

## 五、核心代码骨架

### 5.1 离线入库脚本 `scripts/build_index.py`

```python
import chromadb
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer
import json

# --- 模型加载 ---
clip_model = CLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
bge_model = AutoModel.from_pretrained("BAAI/bge-m3")

# --- ChromaDB ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="products",
    metadata={"hnsw:space": "cosine"}
)

# --- 离线处理 ---
def build_index(products_jsonl):
    for line in open(products_jsonl, encoding="utf-8"):
        p = json.loads(line)

        # 图片 Embedding
        img = Image.open(p["image_path"]).convert("RGB")
        img_inputs = clip_processor(images=img, return_tensors="pt")
        img_emb = clip_model.get_image_features(**img_inputs)[0].tolist()

        # 文本 Embedding
        text = f"{p['title']} {p['description']}"
        text_inputs = bge_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        text_emb = bge_model(**text_inputs).last_hidden_state[:, 0, :][0].tolist()

        # 融合 Embedding（简单拼接，检索时同样操作）
        fused_emb = img_emb + text_emb  # 维度=d_model，两者同维可拼接/求和

        collection.add(
            ids=[p["id"]],
            embeddings=[fused_emb],
            metadatas=[{
                "title": p["title"],
                "price": p["price"],
                "category": p.get("category", ""),
                "image_path": p["image_path"],
            }]
        )

if __name__ == "__main__":
    build_index("data/products.jsonl")
    print(f"入库完成: {collection.count()} 条")
```

### 5.2 在线检索 API `api/search.py`

```python
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io

app = FastAPI()

@app.post("/api/search")
async def search(
    image: UploadFile = File(None),
    text: str = Form(""),
    price_max: float = Form(None),
    category: str = Form(None),
    top_k: int = Form(5),
):
    alpha = 0.6  # 图搜权重

    # 1. 编码
    img_emb, text_emb = None, None

    if image:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_emb = encode_image(img)  # CLIP -> list[float]

    if text:
        text_emb = encode_text(text)  # BGE-M3 -> list[float]

    # 2. 融合查询向量
    if img_emb and text_emb:
        query_emb = [alpha * i + (1-alpha) * t for i, t in zip(img_emb, text_emb)]
    elif img_emb:
        query_emb = img_emb
    else:
        query_emb = text_emb

    # 3. 构建 filter
    where_filter = {}
    if price_max:
        where_filter["price"] = {"$lte": price_max}
    if category:
        where_filter["category"] = category

    # 4. 检索
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=20,
        where=where_filter if where_filter else None,
    )

    # 5. LLM Rerank + 导购理由
    candidates = format_candidates(results)
    prompt = build_guide_prompt(candidates, text or "图片搜索")
    guide_result = await llm_chat(prompt)  # DeepSeek API

    return {
        "products": guide_result["products"],
        "reasoning": guide_result["reasoning"],
        "tip": guide_result.get("tip", ""),
    }
```

### 5.3 LLM 导购 Prompt

```python
def build_guide_prompt(candidates, user_query):
    return f"""你是专业电商导购。用户需求：「{user_query}」

候选商品（已按相关性排序）：
{candidates}

请：
1. 从中选出最合适的 Top-5
2. 对每个商品写一句导购理由（不超过30字）
3. 给出一条整体搭配/选购建议

返回 JSON 格式：
{{"products": [{{"id", "title", "price", "reason"}}], "reasoning": "整体分析", "tip": "选购建议"}}
"""
```

---

## 六、数据准备

### 6.1 数据集选择

| 数据集             | 规模    | 优势         | 获取        |
| --------------- | ----- | ---------- | --------- |
| **Fashion200k** | 20 万条 | 服装为主，图文配对  | GitHub 下载 |
| **DeepFashion** | 80 万张 | 类别丰富，关键点标注 | 官网申请      |
| **淘宝商品图** (自建)  | 可规模   | 最真实        | 爬虫采集      |

### 6.2 最小可用集

起步不需要 20 万条。先准备 **1000-2000 条**覆盖 10+ 品类的商品，把链路跑通，答辩前再扩到 5000+。

### 6.3 数据格式

```jsonl
{"id": "prod_001", "image_path": "data/images/001.jpg", "title": "法式复古碎花连衣裙", "description": "V领 收腰 A字摆 雪纺 春夏", "price": 189.00, "category": "女装/连衣裙"}
```

---

## 七、答辩策略

### 7.1 差异化亮点（三个必须讲的点）

1. **混合检索不是黑盒**：讲清楚为什么 `α=0.6`，图文互补的消融实验
2. **LLM Rerank vs 纯向量召回的 A/B 对比**：有数据支撑
3. **多轮对话不是 Chat 而是 Agent**：用 ReAct 思路管理上下文和工具调用

### 7.2 答辩 PPT 结构（8 页）

```
1. 课题理解（1页）— 电商导购的痛点和解法
2. 系统架构（1页）— 上面那张架构图
3. 多模态对齐（1页）— CLIP + BGE 混合检索原理
4. LLM 导购生成（1页）— Prompt 设计 + Rerank 策略
5. 效果展示（1页）— 录屏 GIF + Top-5 准确率
6. 消融实验（1页）— 纯图搜 / 纯文搜 / 混合 / 混合+Rerank 对比
7. 技术亮点（1页）— 混合检索权重、Agent 化对话
8. 总结与展望（1页）
```

### 7.3 可能被追问的问题

| 问题 | 准备方向 |
|------|---------|
| 为什么选 Chinese-CLIP 而不是 OpenAI CLIP？ | 中文电商数据分布匹配 |
| 混合检索的 α 怎么调？ | 离线网格搜索 + 人工标注 |
| ChromaDB 比 Milvus 差在哪？ | 功能对比 + 你这个场景够用 |
| 如果商品库 100 万条怎么办？ | IVF 索引 + 粗排精排两阶段 |
| 怎么评估检索质量？ | Recall@K + MRR，构造测试集 |

---

## 八、风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| CLIP 效果不如预期 | 中 | 备选 OpenAI CLIP，英文数据集兜底 |
| 商品数据集难获取 | 中 | Fashion200k 保底，质量不够数量凑 |
| LLM Rerank 延迟高 | 低 | DeepSeek API 很快，备选本地 Qwen |
| 前端开发卡壳 | 中 | 用 TRAE 生成，不行就 Gradio 保底 |
| 答辩与期末冲突 | 高 | Week 3 提前准备好材料，不拖到最后两天 |

---

## 九、环境清单

```txt
# requirements.txt
chromadb>=0.5.0
transformers>=4.40.0
torch>=2.0.0
pillow>=10.0.0
fastapi>=0.110.0
uvicorn>=0.27.0
python-multipart>=0.0.9
openai>=1.0.0        # DeepSeek API 兼容 OpenAI SDK
onnxruntime>=1.17.0  # 可选：推理加速
```

---

## 十、团队分工

> 三人组：徐云杰（算法）+ 赖赛一（后端）+ 李永琦（前端）

### 10.1 分工总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│  徐云杰（算法核心）       赖赛一（后端工程）          李永琦（前端 + 可视化）     │
│  ────────────────       ──────────────          ────────────────────── │
│  · CLIP + BGE-M3 模型   · FastAPI 架构           · 前端页面 + 移动端适配   │
│  · ChromaDB 索引构建     · API 路由 + 数据管道      · 商品卡展示 + 交互动效  │
│  · 混合检索算法 (α 调参)  · Metadata Filter       · 图片批量预处理脚本      │
│  · LLM Rerank Prompt    · 多轮对话上下文管理       · 评测结果可视化看板      │
│  · 消融实验 + 评测        · 边界处理 + 性能优化      · 搜索历史 / 偏好收集 UI  │
│                          · 静态资源服务            · Demo 视频 + PPT 答辩   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 10.2 各周详细分工

#### Week 1（5.20-5.26）：核心链路跑通

| 任务                                           | 负责人          | 产出                                          | 依赖             |
| -------------------------------------------- | ------------ | ------------------------------------------- | -------------- |
| 环境搭建：装通 CLIP + BGE-M3 + ChromaDB + FastAPI   | **三人协作**     | `requirements.txt`，各模块 import 成功            | 无              |
| 商品数据准备：找公开数据集、清洗为 `products.jsonl`           | **赖赛一**       | 1000+ 条标准化商品数据                              | 无              |
| **商品图片批量预处理**：统一尺寸、压缩、生成缩略图                  | **李永琦**       | `scripts/preprocess_images.py`，所有图片 ≤ 512px | 数据准备完成         |
| **API Mock Server**：按接口约定搭静态 mock 服务         | **李永琦**       | 前端独立开发不阻塞，mock 数据可切换                        | 接口约定确定         |
| **前端页面骨架**：上传页 + 结果展示页（静态 mock 数据渲染）         | **李永琦**       | 页面框架 + 路由 + 基础组件                            | mock server 就绪 |
| 离线入库脚本 `build_index.py`（CLIP+BGE → ChromaDB） | **徐云杰**      | 脚本跑通，collection 可查询                         | 数据 + 图片就绪      |
| 在线检索 API `/api/search`（编码 → 检索 → Top-20）     | **徐云杰 + 后端** | API 可用，Postman 可调                           | build_index 完成 |
| 端到端调通：上传图片 → 返回候选商品                          | **徐云杰 + 后端** | 录屏 10s                                      | API 完成         |

#### Week 2（5.27-6.2）：质量提升 + LLM 生成

| 任务                                          | 负责人         | 产出              | 依赖       |
| ------------------------------------------- | ----------- | --------------- | -------- |
| 混合检索：图文加权融合，网格搜索调 α                         | **徐云杰**     | 检索准确率提升 10%+    | Week1 链路 |
| LLM Rerank + 导购理由生成（Prompt 优化）              | **徐云杰**     | Rerank 效果验收通过   | 混合检索     |
| Metadata Filter（价格/品类筛选）                    | **赖赛一**      | `filter` 参数可用   | API 架构   |
| 多轮对话上下文管理                                   | **赖赛一**      | 对话历史追踪，追问可用     | API 架构   |
| **前端页面功能完善**：图片拖拽上传、商品卡展开/详情、筛选器面板、骨架屏加载    | **李永琦**      | 可交互的前端原型（接近最终态） | API mock |
| **搜索历史 + 用户反馈收集**：本地存储搜索历史，商品点赞/踩按钮，埋点记录    | **李永琦**      | 用户行为数据可用于评测分析   | 前端页面     |
| **商品库数据看板**：品类分布饼图、价格分布直方图、Embedding 2D 可视化 | **李永琦**      | 答辩 PPT 可直接截图的看板 | 数据就绪     |
| 前后端联调                                       | **前端 + 后端** | 数据打通，页面可调真实 API | 前端 + API |
| **中期评审自测**（完整流程录 Demo）                      | **三人**      | 3 分钟演示视频        | 全链路      |

#### Week 3（6.3-6.10）：打磨 + 答辩

| 任务                                  | 负责人          | 产出            | 依赖        |
| ----------------------------------- | ------------ | ------------- | --------- |
| 评测：50 条测试 query，Top-5 准确率 + 消融实验    | **徐云杰**      | 自评报告（含图表）     | 全链路稳定     |
| 性能优化（ONNX 导出、索引调优）                  | **赖赛一**       | 延迟降低 30%+     | 评测完成      |
| 边界处理（无结果降级、大图压缩、错误兜底）               | **赖赛一**       | 鲁棒性提升         | 评测完成      |
| **前端 UI 打磨**：交互动效、移动端响应式适配、暗色模式、无障碍 | **李永琦**       | 最终可演示前端（多端适配） | 联调完成      |
| **评测结果可视化**：准确率对比柱状图、消融实验雷达图、检索延迟折线 | **李永琦**       | 答辩 PPT 核心图表素材 | 评测数据出炉    |
| 答辩 PPT（8 页）+ 架构图 + 技术插图             | **前端主导**     | 答辩 PPT 终稿     | 全链路线框图    |
| 技术文档（架构设计、关键决策、踩坑记录）                | **徐云杰 + 后端** | 技术文档 1 份      | —         |
| Demo 视频：录制 + 剪辑 + 配音 + 字幕           | **李永琦**       | 3-5 分钟答辩演示视频  | 前端 + 后端稳定 |
| 提交前最终检查                             | **三人**       | 提交确认          | 全部        |

### 10.3 接口约定（关键：前后端独立开发的契约）

为了防止前端等后端、后端等算法，**Week 1 第一天就定好 API 契约**：

```
POST /api/search
  Request:
    image:   File (optional)    — 上传图片
    text:    str  (optional)    — 文本描述
    price_max: float (optional) — 价格上限
    category: str  (optional)   — 品类筛选
    top_k:   int  (default 5)   — 返回数量

  Response:
    {
      "products": [
        {
          "id": "prod_001",
          "title": "法式复古碎花连衣裙",
          "price": 189.00,
          "image_url": "/static/images/001.jpg",
          "reason": "V领收腰设计修饰颈肩，雪纺面料适合春夏",
          "score": 0.92
        }
      ],
      "reasoning": "整体分析...",
      "tip": "选购建议..."
    }
```

前端用 **mock 数据**（`public/mock/search.json`）先行开发，自己维护 mock server，不依赖后端 API 就绪。真实 API 就绪后切换一个 baseURL 即可。

### 10.4 协作规范

| 规范         | 说明                                                              |
| ---------- | --------------------------------------------------------------- |
| **代码仓库**   | GitHub 私有仓库，main 分支保护，PR 合并                                     |
| **分支命名**   | `feat/build-index` / `feat/api-search` / `feat/frontend-upload` |
| **每日站会**   | 晚 10 点，10 分钟，同步进度 + 阻塞点                                         |
| **文档协作**   | 技术文档用 Markdown 放 `docs/`，PPT 用飞书/腾讯文档协作                         |
| **模型权重**   | Git LFS 或单独网盘共享（不直接 commit）                                     |
| **API 调试** | 后端先给 curl 示例，前端用 Postman/Apifox 自测                              |

### 10.5 风险预案

| 情况               | 预案                                          |
| ---------------- | ------------------------------------------- |
| **前端进度落后**（任务最重） | 后端队友支援：分担图片预处理脚本；徐云杰支援：Gradio 保底 UI（5 分钟搞定） |
| 后端进度落后           | 徐云杰先补上 API 层（FastAPI 代码量少），后端集中精力做数据管道和优化   |
| 数据集找不到合适的        | 三人各找 1 个备选，Day 2 晚上投票决定                     |
| 答辩与期末冲突          | Week 3 材料提前到 Week 2 周末开始准备                  |

---

> 最后更新：2026.05.12  
> 下一步：拉队友进群 → 过一遍分工 → 确认仓库 → Week 1 启动
