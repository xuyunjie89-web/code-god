# RAG 多模态电商导购

AI 电商导购系统：图片 + 文本混合检索 → LLM 生成导购文案。字节跳动 AI 全栈挑战赛课题 4。

## 环境安装

### 方式一：Conda（推荐，含 GPU 加速）

```bash
conda env create -f environment.yml
conda activate pytorch_gpu
```

### 方式二：pip

```bash
pip install -r requirements.txt
```

### 国内加速（华为云镜像）

如果 HuggingFace 下载模型慢，设置镜像：

```bash
set HF_ENDPOINT=https://hf-mirror.com
```

macOS / Linux：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 启动服务

```bash
set DEEPSEEK_API_KEY=sk-your-key
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

浏览器打开 http://localhost:8000/docs 查看 API 文档。

## 项目结构

```
code-god/
├── main.py          # FastAPI 后端
├── scripts/         # 离线脚本（入库、评测等）
├── data/            # 商品数据
├── chroma_db/       # 向量库（本地，不提交）
└── requirements.txt
```
