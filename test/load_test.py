import requests
import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGE_DIR = "/Users/ratchanonkhongsawi/Desktop/CMKL/3rd/Scalable/assessment/assessment_2/dataset/images"
url = "http://localhost:8001/infer"

# -------------------------
# READ IMAGES ONCE
# -------------------------
print("=== START READING IMAGES (ONE BATCH) ===")
start_time = time.time()

one_batch = []
for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith(".png"):

        path = os.path.join(IMAGE_DIR, filename)

        img = cv2.imread(path)
        if img is None:
            print("Could not read:", path)
            continue

        _, enc = cv2.imencode(".png", img)
        img_bytes = enc.tobytes()

        one_batch.append(("payloads", (filename, img_bytes, "image/png")))

read_elapsed = time.time() - start_time

print("=== FINISHED READING IMAGES ===")
print("Images per batch:", len(one_batch))
print("Total batch size:", sum(len(f[1][1]) for f in one_batch), "bytes")
print("Read time:", f"{read_elapsed:.4f}", "seconds")
print("\n")


# -------------------------
# FUNCTION TO SEND ONE BATCH
# -------------------------
def send_batch(batch_id):
    start = time.perf_counter()
    res = requests.post(url, files=one_batch)
    end = time.perf_counter()
    latency = end - start

    return {
        "batch": batch_id,
        "latency": latency,
        "status": res.status_code,
        "response": res.text
    }


# -------------------------
# PARALLEL EXECUTION
# -------------------------
print("=== START SENDING 10 BATCHES IN PARALLEL ===")

latencies = []

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(send_batch, i): i for i in range(1, 11)}

    for future in as_completed(futures):
        result = future.result()
        latencies.append(result["latency"])

        print(f"\n--- Batch {result['batch']} completed ---")
        print(f"Latency: {result['latency']:.4f} seconds")
        print("Status:", result["status"])
        print("Response:", result["response"])


# -------------------------
# SUMMARY
# -------------------------
print("\n=== ALL BATCHES COMPLETED ===")

for i, l in enumerate(latencies, 1):
    print(f"Batch {i}: {l:.4f} sec")

print("\nAverage latency:", f"{sum(latencies)/len(latencies):.4f}", "seconds")
print("Fastest:", f"{min(latencies):.4f}", "seconds")
print("Slowest:", f"{max(latencies):.4f}", "seconds")