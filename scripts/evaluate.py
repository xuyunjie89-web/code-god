"""
evaluate.py — 检索评测 + 消融实验
==================================
四种策略对比: 纯图搜 / 纯文搜 / 混合检索 / 混合+LLM Rerank
三类 query 分组统计: text / image / image_text

用法: python scripts/evaluate.py [--jsonl data/test_queries.jsonl]
"""

import json, os, sys, time

import torch
import chromadb
from PIL import Image
from openai import OpenAI
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, AutoTokenizer, AutoModel

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.6


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
    if not text.strip():
        return [0.0] * 512
    inputs = bge_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        return bge_model(**inputs).last_hidden_state[:, 0, :][0][:512].tolist()


def fuse(img_emb, text_emb, alpha):
    return [alpha * i + (1 - alpha) * t for i, t in zip(img_emb, text_emb)]


def recall_at_k(retrieved_ids, expect_id, k):
    return 1 if expect_id in retrieved_ids[:k] else 0


def mrr(retrieved_ids, expect_id):
    for i, rid in enumerate(retrieved_ids):
        if rid == expect_id:
            return 1.0 / (i + 1)
    return 0.0


def metrics_by_type(queries_with_results, qtype):
    """按 query 类型统计指标"""
    subset = [(ret_ids, exp_id) for ret_ids, exp_id, qt in queries_with_results if qt == qtype]
    if not subset:
        return None
    n = len(subset)
    r1 = sum(recall_at_k(rids, eid, 1) for rids, eid in subset) / n * 100
    r5 = sum(recall_at_k(rids, eid, 5) for rids, eid in subset) / n * 100
    m = sum(mrr(rids, eid) for rids, eid in subset) / n
    return {"n": n, "recall@1": round(r1, 1), "recall@5": round(r5, 1), "mrr": round(m, 3)}


def evaluate_strategy(name, test_queries, collection, clip_processor, clip_model,
                      bge_tokenizer, bge_model, alpha=None, llm_client=None):
    """评测一种策略，返回 (overall_metrics, per_query_results, avg_ms)"""
    all_results = []  # [(retrieved_ids, expect_id, qtype), ...]
    times = []

    for q in test_queries:
        t0 = time.time()

        ie = img_embed(q["image"], clip_processor, clip_model) if q.get("image") else [0.0] * 512
        te = text_embed(q.get("text", ""), bge_tokenizer, bge_model)

        if name == "纯图搜":
            query_emb = ie
        elif name == "纯文搜":
            query_emb = te
        else:
            query_emb = fuse(ie, te, alpha)

        n = 5 if name != "混合+LLM" else 20
        results = collection.query(query_embeddings=[query_emb], n_results=n)
        ids = results["ids"][0]

        if name == "混合+LLM" and llm_client and llm_client.api_key:
            candidates = []
            for i in range(len(ids)):
                candidates.append({
                    "id": ids[i],
                    "title": results["metadatas"][0][i]["title"],
                    "price": results["metadatas"][0][i]["price"],
                })
            prompt = (
                f"用户搜索: {q.get('text', '')}。候选商品: {json.dumps(candidates, ensure_ascii=False)}。"
                f"请按相关性从高到低排序，只返回 id 数组 JSON: {{\"ranked_ids\": [\"prod_001\", ...]}}"
            )
            try:
                resp = llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                ranked = json.loads(resp.choices[0].message.content)
                ids = ranked.get("ranked_ids", ids)
            except Exception:
                pass

        t = round((time.time() - t0) * 1000)
        times.append(t)

        qtype = q.get("type", "text")
        all_results.append((ids, q["expect_id"], qtype))

    overall = {
        "recall@1": round(sum(recall_at_k(r, e, 1) for r, e, _ in all_results) / len(all_results) * 100, 1),
        "recall@5": round(sum(recall_at_k(r, e, 5) for r, e, _ in all_results) / len(all_results) * 100, 1),
        "mrr": round(sum(mrr(r, e) for r, e, _ in all_results) / len(all_results), 3),
        "avg_ms": round(sum(times) / len(times)),
    }

    by_type = {}
    for qt in ["text", "image", "image_text"]:
        m = metrics_by_type(all_results, qt)
        if m:
            by_type[qt] = m

    return overall, by_type


def main():
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(MODEL_DIR, "data", "test_queries.jsonl")

    test_queries = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                if q.get("image") and not os.path.isabs(q["image"]):
                    q["image"] = os.path.join(MODEL_DIR, q["image"])
                test_queries.append(q)

    # 统计各类数量
    from collections import Counter
    type_counts = Counter(q.get("type", "text") for q in test_queries)
    print(f"测试 query: {len(test_queries)} 条 (text={type_counts.get('text',0)}, image={type_counts.get('image',0)}, image_text={type_counts.get('image_text',0)})")

    print("加载模型...")
    clip_processor, clip_model, bge_tokenizer, bge_model = load_models()
    client = chromadb.PersistentClient(path=os.path.join(MODEL_DIR, "chroma_db"))
    collection = client.get_or_create_collection("products")
    print(f"商品库: {collection.count()} 条\n")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        llm_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        print("LLM 客户端已就绪")
    else:
        llm_client = None
        print("[WARN] DEEPSEEK_API_KEY 未设置，跳过 混合+LLM 策略")

    strategies = [
        ("纯图搜", None),
        ("纯文搜", None),
        ("混合检索", ALPHA),
    ]
    if llm_client:
        strategies.append(("混合+LLM", ALPHA))

    all_overall = {}
    all_by_type = {}

    for name, alpha in strategies:
        overall, by_type = evaluate_strategy(name, test_queries, collection,
                                             clip_processor, clip_model,
                                             bge_tokenizer, bge_model,
                                             alpha=alpha, llm_client=llm_client)
        all_overall[name] = overall
        all_by_type[name] = by_type

    # ===== 打印汇总表 =====
    qtypes = [t for t in ["text", "image", "image_text"] if any(t in bt for bt in all_by_type.values())]
    col_w = 11

    # 表头
    header = f"{'策略':<10}"
    for qt in qtypes:
        n = all_by_type[strategies[0][0]][qt]["n"]
        header += f" {'':->{col_w}} {qt}(n={n}) {'':->{col_w}}"
    header += f" {'':->{col_w}} 综合(n={len(test_queries)}) {'':->{col_w}}"
    print(header)

    sub_h = f"{'':<10}"
    for _ in qtypes:
        sub_h += f" {'R@1':<6} {'R@5':<6} {'MRR':<6}"
    sub_h += f" {'R@1':<6} {'R@5':<6} {'MRR':<6} {'耗时':<6}"
    print(sub_h)
    print("-" * (10 + (3*col_w+2)*len(qtypes) + 4*col_w + 2))

    best_name = None
    best_r5 = -1

    for name, alpha in strategies:
        row = f"{name:<10}"
        for qt in qtypes:
            m = all_by_type[name].get(qt)
            if m:
                row += f" {m['recall@1']:<5}% {m['recall@5']:<5}% {m['mrr']:<6}"
            else:
                row += f" {'-':<6} {'-':<6} {'-':<6}"
        ov = all_overall[name]
        row += f" {ov['recall@1']:<5}% {ov['recall@5']:<5}% {ov['mrr']:<6} {ov['avg_ms']}ms"
        print(row)

        if ov["recall@5"] > best_r5:
            best_r5 = ov["recall@5"]
            best_name = name

    print(f"\n最佳策略: {best_name} (综合 R@5={best_r5}%)")

    # 输出完整 JSON
    print("\n--- 完整结果 (JSON) ---")
    print(json.dumps({"overall": all_overall, "by_type": all_by_type}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
