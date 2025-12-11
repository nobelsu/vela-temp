import time
import asyncio
import csv

import agents.main.agent as MainAgent
import agents.prediction.agent as PredictionAgent
from agents.prediction.helper import row_to_formatted_string

async def train():
    start = time.time()
    with open("data/train.csv", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = []
        actual = []
        for row in reader:
            rows.append(row_to_formatted_string(row))
            actual.append("1" == int(row.get("success")))
            if len(rows) >= 10:
                results = await PredictionAgent.predictSuccess(rows)
                blocks = []
                cnt = 0
                for i, r in enumerate(results, 1):
                    cnt += int(r.prediction == actual[i-1])
                    block = [
                        f"Prediction {i}:",
                        f"Agent answer: {r.prediction}",
                        f"Correct answer: {actual[i-1]}",
                        f"Reasoning: {r.reason}",
                        f"Thought process:\n {'\n'.join(r.thought_process) if r.thought_process else '-'}",
                        ""
                    ]
                    blocks.append("\n".join(block))
                instructions = (f"REPORT:\n\nAccuracy: {cnt}/{len(results)}\n\n") + ("\n".join(blocks))
                await MainAgent.central(instructions)
                rows = []
                actual = []
    end = time.time()
    t = end - start
    print(f"\n\nTotal run time: {t:.2f}s")

if __name__ == "__main__":
    asyncio.run(train())