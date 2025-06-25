import time
from pathlib import Path

import faiss


def build_faiss_index_and_train(
    train_data,
    index_type: str = "ivfpq",
    n_probe: int = 64,
    search: str = "l2",
    gpu: bool = True,
    index_dir: Path = None,
):

    assert n_probe > 0, f"n_probe must be > 0: {n_probe}"

    _, d = train_data.shape

    # Build a flat (CPU) index
    if search.lower() == "l2":
        coarse_quantizer = faiss.IndexFlatL2(d)
        metric = faiss.METRIC_L2
    elif search.lower() in {"ip"}:
        coarse_quantizer = faiss.IndexFlatIP(d)
        metric = faiss.METRIC_INNER_PRODUCT
    else:
        raise ValueError(f"{search!r} only 'l2' and'ip' are valid.")

    print(f"Initializing a {index_type} index...")
    if index_type.lower() == "flat":
        # Using a flat index
        index = coarse_quantizer
    elif index_type.lower() == "ivf":
        nlist = 512  # number of clusters
        index = faiss.IndexIVFFlat(coarse_quantizer, d, nlist, metric)
    elif index_type.lower() == "ivfpq":
        nlist = 256  # number of clusters
        code_sz = 64  # power of 2
        nbits = 8  # nbits must be 8, 12 or 16
        index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, code_sz, nbits, metric)
    else:
        raise ValueError(f"{index_type!r} is not a valid index type.")

    t0 = time.monotonic()
    print("Training the index...")
    index.train(train_data)
    print(
        f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.monotonic() - t0))}"
    )

    t0 = time.monotonic()
    print("Populating the index with database embeddings...")
    index.add(train_data)
    print(
        f"{index.ntotal:>10,} embeddings are added in total to the index in {time.strftime('%H:%M:%S', time.gmtime(time.monotonic() - t0))}"
    )

    if index_dir is not None:
        save_faiss_index(index, index_dir, on_gpu=False)

    if gpu:
        print("Moving the index to \033[93mGPU\033[0m.")
        GPU_RESOURCES = faiss.StandardGpuResources()
        GPU_OPTIONS = faiss.GpuClonerOptions()
        # use float16 table to avoid https://github.com/facebookresearch/faiss/issues/1178
        GPU_OPTIONS.useFloat16 = True
        index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, index, GPU_OPTIONS)
    else:
        print("Using \033[93mCPU\033[0m index.")

    # Number of neighboring cells to visit during the approximate search
    index.nprobe = n_probe

    return index


def load_faiss_index(index_path: Path, gpu: bool = True):

    cpu_index = faiss.read_index(str(index_path))
    print(f"Index with {cpu_index.ntotal:,} embeddings loaded from {str(index_path)}")

    if gpu:
        print("Moving index to GPU…")
        res = faiss.StandardGpuResources()
        gpu_opts = faiss.GpuClonerOptions()
        gpu_opts.useFloat16 = True
        return faiss.index_cpu_to_gpu(res, 0, cpu_index, gpu_opts)
    else:
        return cpu_index


def save_faiss_index(index, db_dir: Path, on_gpu: bool = True):

    if on_gpu:
        # Convert GPU index back to CPU before writing
        cpu_index = faiss.index_gpu_to_cpu(index)
    else:
        cpu_index = index

    db_dir.mkdir(parents=True, exist_ok=True)
    index_path = db_dir / "database.index"
    if index_path.exists:
        print(f"Overwriting existing index.")
    faiss.write_index(cpu_index, str(index_path))
    print(f"Index written to {str(index_path)}")
