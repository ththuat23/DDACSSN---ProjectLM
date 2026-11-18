import json
from collections import Counter

with open("data/prompts.json", "r", encoding="utf-8") as file:
    prompts = json.load(file)

print("Tổng số prompt:", len(prompts))

categories = Counter(
    item["category"] for item in prompts
)

for category, count in categories.items():
    print(category, ":", count)

print("\nPrompt đầu tiên:")
print(prompts[0]["prompt"])