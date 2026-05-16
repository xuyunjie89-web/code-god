"""
fetch_fashion200k.py — 从 HuggingFace 拉 Fashion200k 多类目子集

用法: python scripts/fetch_fashion200k.py [--count 150] [--per_category 17]
"""

import argparse, json, os, sys, random
from collections import defaultdict

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 9 个类目
# Fashion200k "data" split 实际只有 5 个类目
ALL_CATEGORIES = ["dresses", "jackets", "pants", "skirts", "tops"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--per_category", type=int, default=17)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"从 HuggingFace 加载 Fashion200k (streaming, 无shuffle)")
    print(f"目标 {args.count} 条, 每类最多 {args.per_category} 条")
    print("streaming 顺序遍历，数据按类目排列，会跳过已满类目...")
    sys.stdout.flush()

    dataset = load_dataset("Marqo/fashion200k", split="data", streaming=True)

    img_dir = os.path.join(MODEL_DIR, "data", "images")
    os.makedirs(img_dir, exist_ok=True)

    # 加载已有 products.jsonl，避免重复下载
    jsonl_path = os.path.join(MODEL_DIR, "data", "products.jsonl")
    seen = set()
    products = []
    category_counts = defaultdict(int)

    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                base_id = p["id"].replace("fk_", "")
                seen.add(base_id)
                products.append(p)
                cat = p["category"].split("/")[0]
                category_counts[cat] += 1
        print(f"从已有 jsonl 加载 {len(products)} 条记录 ({dict(category_counts)})")

    per_cat = args.per_category
    total_target = args.count

    # 计算还需多少
    needed = total_target - len(products)
    if needed <= 0:
        print(f"已有 {len(products)} 条 >= 目标 {total_target} 条，无需继续")
        print("如需重新采集请删除 data/products.jsonl 后重试")
        return
    print(f"目标 {total_target} 条，已有 {len(products)} 条，还需 {needed} 条")

    scanned = 0
    skipped_category = 0
    skipped_dup = 0
    filled_categories = set(c for c in ALL_CATEGORIES if category_counts[c] >= per_cat)

    for item in dataset:
        scanned += 1

        # 进度打印
        if scanned % 200 == 0:
            cats_status = ", ".join(f"{c}={category_counts[c]}" for c in ALL_CATEGORIES)
            print(f"  [已扫描 {scanned} 条] 收集 {len(products)}/{total_target} | 类目: {cats_status}")
            sys.stdout.flush()

        cat = item["category1"]
        if cat not in ALL_CATEGORIES:
            continue
        if category_counts[cat] >= per_cat:
            skipped_category += 1
            if cat not in filled_categories:
                filled_categories.add(cat)
                print(f"  >>> {cat} 已满 ({per_cat}条), 跳过后续同类目")
                sys.stdout.flush()
            continue

        base_id = item["item_ID"].rsplit("_", 1)[0]
        if base_id in seen:
            skipped_dup += 1
            continue
        seen.add(base_id)

        filename = f"{base_id}.jpg"
        img_path = os.path.join(img_dir, filename)
        item["image"].save(img_path, quality=85)

        price = round(random.uniform(29, 599), 2)
        products.append({
            "id": f"fk_{base_id}",
            "image_path": f"images/{filename}",
            "title": item["category3"],
            "description": item["text"][:200],
            "price": price,
            "category": f"{item['category1']}/{item['category2']}",
        })
        category_counts[cat] += 1

        print(f"  + [{len(products)}/{total_target}] {cat}/{item['category2']}: {filename}")
        sys.stdout.flush()

        # 所有类目都满了 或 达到总数目标
        if len(products) >= total_target:
            break
        if all(category_counts[c] >= per_cat for c in ALL_CATEGORIES):
            print("所有类目均已满额，提前结束")
            break

    # 写 products.jsonl
    jsonl_path = os.path.join(MODEL_DIR, "data", "products.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in products:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n===== 完成 =====")
    print(f"共扫描 {scanned} 条, 收集 {len(products)} 条商品")
    print(f"跳过: 类目已满 {skipped_category}, 去重 {skipped_dup}")
    print(f"\n各类目分布:")
    for cat in ALL_CATEGORIES:
        print(f"  {cat}: {category_counts[cat]} 件")
    print(f"\n图片存于: {img_dir}")
    print(f"jsonl: {jsonl_path}")


if __name__ == "__main__":
    main()
