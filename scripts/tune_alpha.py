"""
tune_alpha.py — 混合检索权重调优
================================
网格搜索 α ∈ [0.0, 1.0]，用测试 query 评估不同权重下的检索效果

用法: python scripts/tune_alpha.py
"""

import json, os, sys

import torch
import chromadb
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer, AutoModel

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_models():
    clip_processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    clip_model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    clip_model.eval()
    bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    bge_model = AutoModel.from_pretrained("BAAI/bge-m3")
    bge_model.eval()
    return clip_processor, clip_model, bge_tokenizer, bge_model


def img_embed(img_path, clip_processor, clip_model):
    img = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        return clip_model.get_image_features(**inputs)[0].tolist()


def text_embed(text, bge_tokenizer, bge_model):
    inputs = bge_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        return bge_model(**inputs).last_hidden_state[:, 0, :][0][:512].tolist()


def evaluate_alpha(test_queries, collection, clip_processor, clip_model, bge_tokenizer, bge_model):
    """对每个 α，检查期望商品是否在 Top-K 中召回"""
    alphas = [round(v, 2) for v in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)]
    results = {}

    for alpha in alphas:
        hit = 0
        total = 0
        for q in test_queries:
            # 编码
            ie = img_embed(q["image"], clip_processor, clip_model) if q.get("image") else [0.0] * 512
            te = text_embed(q.get("text", ""), bge_tokenizer, bge_model)

            # 融合
            fused = [alpha * i + (1 - alpha) * t for i, t in zip(ie, te)]

            # 检索
            res = collection.query(query_embeddings=[fused], n_results=5)
            total += 1
            if q["expect_id"] in res["ids"][0]:
                hit += 1

        results[alpha] = round(hit / total * 100, 1)
        print(f"  α={alpha:.1f}  命中率: {results[alpha]}% ({hit}/{total})")

    best = max(results, key=results.get)
    print(f"\n最优 α = {best} ({results[best]}%)")
    return best, results


def main():
    # 加载模型
    print("加载模型...")
    clip_processor, clip_model, bge_tokenizer, bge_model = load_models()

    # 连接 ChromaDB
    client = chromadb.PersistentClient(path=os.path.join(MODEL_DIR, "chroma_db"))
    collection = client.get_or_create_collection("products")

    # 测试 query 集: {text, image(可选), expect_id}
    # 图片都指向同一张 OIP.jpg 只测文本检索能力
    test_img = os.path.join(MODEL_DIR, "OIP.jpg")
    test_queries = [
        {"text": "连衣裙", "expect_id": "prod_001"},
        {"text": "法式复古裙子", "expect_id": "prod_001"},
        {"text": "便宜的T恤", "expect_id": "prod_002"},
        {"text": "阔腿裤", "expect_id": "prod_003"},
        {"text": "白色鞋子", "expect_id": "prod_004"},
        {"text": "通勤包包", "expect_id": "prod_005"},
    ]

    print(f"\n测试 query: {len(test_queries)} 条")
    print(f"商品库: {collection.count()} 条\n")
    print("=" * 40)

    best_alpha, all_results = evaluate_alpha(
        test_queries, collection,
        clip_processor, clip_model, bge_tokenizer, bge_model
    )

    print("\n" + "=" * 40)
    print("各 α 对比:")
    for a in sorted(all_results):
        marker = " ← 最优" if a == best_alpha else ""
        bar = chr(9608) * int(all_results[a] / 5)
        print(f"  α={a:.1f}  {bar} {all_results[a]}%{marker}")


if __name__ == "__main__":
    main()
