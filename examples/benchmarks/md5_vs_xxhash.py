import time
import xxhash
from hashlib import md5
from tqdm import tqdm
import numpy as np


def xxhash_ids(data: list[str]) -> np.ndarray:
    return np.fromiter(
        (xxhash.xxh32_intdigest(d.encode()) for d in data),
        dtype=np.uint32,
        count=len(data)
    )


def md5_ids(data: list[str]) -> np.ndarray:
    return np.fromiter(
        (int(md5(d.encode()).hexdigest(), 16) & 0xFFFFFFFF for d in data),
        dtype=np.uint32,
        count=len(data)
    )


if __name__ == "__main__":
    num_ids = 1000000
    num_iterations = 100
    xxhash_times = []
    md5_times = []

    for i in tqdm(range(num_iterations)):
        test_data = [f"{i}_{j}" for j in range(num_ids)]
        
        start_time = time.time()
        xxhash_result = xxhash_ids(test_data)
        xxhash_times.append(time.time() - start_time)
        
        start_time = time.time()
        md5_result = md5_ids(test_data)
        md5_times.append(time.time() - start_time)
        
        assert len(xxhash_result) == len(md5_result) == num_ids
        assert not np.array_equal(xxhash_result, md5_result)

    avg_xxhash_time = np.mean(xxhash_times)
    avg_md5_time = np.mean(md5_times)
    std_xxhash_time = np.std(xxhash_times)
    std_md5_time = np.std(md5_times)

    print(f"num_ids: {num_ids} | num_iterations: {num_iterations}")
    print(f"\nAverage xxhash time: {avg_xxhash_time:.4f} seconds")
    print(f"Average MD5 time: {avg_md5_time:.4f} seconds")
    print(f"xxhash is {avg_md5_time / avg_xxhash_time:.2f}x faster than MD5 on average")
    print(f"\nxxhash time standard deviation: {std_xxhash_time:.4f} seconds")
    print(f"MD5 time standard deviation: {std_md5_time:.4f} seconds")