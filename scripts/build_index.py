"""
build_index.py — 离线商品入库脚本
===============================
读取 products.jsonl → CLIP 编码图 + BGE-M3 编码文 → 融合 → 写入 ChromaDB

用法: python scripts/build_index.py [--jsonl data/products.jsonl] [--reset]
      --reset  清空旧数据重新导入
"""

import argparse, json, os, sys

import torch
import chromadb
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer, AutoModel

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.6


def load_models():
    """加载 Chinese-CLIP + BGE-M3"""
    print("加载 Chinese-CLIP...")
    clip_processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    clip_model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    clip_model.eval()

    print("加载 BGE-M3...")
    bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    bge_model = AutoModel.from_pretrained("BAAI/bge-m3")
    bge_model.eval()

    return clip_processor, clip_model, bge_tokenizer, bge_model


def img_embed(img_path, clip_processor, clip_model):
    """图片 → 512维向量"""
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            return clip_model.get_image_features(**inputs)[0].tolist()
    except Exception as e:
        print(f"  [跳过] 图片读取失败 {img_path}: {e}")
        return None


def text_embed(text, bge_tokenizer, bge_model):
    """文本 → 512维向量 (BGE-M3 [CLS] 前512维)"""
    if not text.strip():
        return [0.0] * 512
    inputs = bge_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        return bge_model(**inputs).last_hidden_state[:, 0, :][0][:512].tolist()


def build_index(jsonl_path, reset=False):
    """主流程: 逐行读取 jsonl → 编码 → 入库"""
    if not os.path.isfile(jsonl_path):
        print(f"找不到文件: {jsonl_path}")
        sys.exit(1)

    # 加载模型
    clip_processor, clip_model, bge_tokenizer, bge_model = load_models()

    # 连接 ChromaDB
    client = chromadb.PersistentClient(path=os.path.join(MODEL_DIR, "chroma_db"))
    if reset:
        try:
            client.delete_collection("products")
            print("已清空旧数据")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name="products",
        metadata={"hnsw:space": "cosine"},
    )

    # 逐行处理
    total = 0
    skipped = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)

            # 图片编码
            img_path = p.get("image_path", "")
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(jsonl_path), img_path)
            img_emb = img_embed(img_path, clip_processor, clip_model)
            if img_emb is None:
                skipped += 1
                continue

            # 文本编码
            text = f"{p.get('title', '')} {p.get('description', '')}"
            text_emb = text_embed(text, bge_tokenizer, bge_model)

            # 融合
            fused = [ALPHA * i + (1 - ALPHA) * t for i, t in zip(img_emb, text_emb)]

            # 写入
            collection.add(
                ids=[p["id"]],
                embeddings=[fused],
                metadatas=[{
                    "title": p.get("title", ""),
                    "price": p.get("price", 0),
                    "category": p.get("category", ""),
                    "image_path": img_path,
                }],
            )
            total += 1
            print(f"  [{total}] {p['id']} — {p.get('title', '')}")

    print(f"\n入库完成: {total} 条, 跳过 {skipped} 条, 总计 {collection.count()} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=os.path.join(MODEL_DIR, "data", "products.jsonl"))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    build_index(args.jsonl, args.reset)
