import time
import faiss
import numpy as np


def get_index(
    train_data,
    index_type,
    max_nitem_train: int = None,
    n_probe: int = 40,
    use_gpu: bool = True,
):
    """
    • Create FAISS index
    • Train index using all the data
    • Return index

    Since we store L2 normalized fingerprints, L2, dot product, and cosine
    similarity are equivalent. Therefore, we can use L2 index for all the
    similarity search methods.

    Parameters
    ----------
    train_data : (float32)
        numpy.memmap or numpy.ndarray
    index_type : (str)
        Index type must be one of {'L2', 'IVF', 'IVFPQ', 'IVFPQ-RR',
                                   'IVFPQ-ONDISK', HNSW'}
    max_nitem_train : (int)
        Max number of items to be used for training index. Default is 1e7.
    n_probe : (int)
        Number of neighboring cells to visit during search. Default is 40.
    use_gpu: (bool)
        If False, use CPU. Default is True.

    Returns
    -------
    index : (faiss.swigfaiss_avx2.GpuIndex***)
        Trained FAISS index.

    References:

        https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

    """

    # GPU Setup
    if use_gpu:
        GPU_RESOURCES = faiss.StandardGpuResources()
        GPU_OPTIONS = faiss.GpuClonerOptions()
        # use float16 table to avoid https://github.com/facebookresearch/faiss/issues/1178
        GPU_OPTIONS.useFloat16 = True

        # TODO: understand the following options
        # GPU_OPTIONS.usePrecomputed = False
        # GPU_OPTIONS.indicesOptions = faiss.INDICES_CPU

    # Number of fingerprints and fingerprint dimension
    N, d = train_data.shape

    # Build a flat (CPU) index
    index = faiss.IndexFlatL2(d)

    mode = index_type.lower()
    print(f"Creating index: \033[93m{mode}\033[0m")
    if mode == "l2":
        # Using L2 index
        pass
    elif mode == "ivf":
        # Using IVF index
        nlist = 400
        index = faiss.IndexIVFFlat(index, d, nlist)
    elif mode == "ivfpq":
        # Using IVF-PQ index
        code_sz = 64  # power of 2
        n_centroids = 256  #
        nbits = (
            8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
        )
        index = faiss.IndexIVFPQ(index, d, n_centroids, code_sz, nbits)
    elif mode == "ivfpq-rr":
        # Using IVF-PQ index + Re-rank
        code_sz = 64
        n_centroids = 256  # 10:1.92ms, 30:1.29ms, 100: 0.625ms
        nbits = (
            8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
        )
        M_refine = 4
        nbits_refine = 4
        index = faiss.IndexIVFPQR(
            index, d, n_centroids, code_sz, nbits, M_refine, nbits_refine
        )
    elif mode == "ivfpq-ondisk":
        if use_gpu:
            raise NotImplementedError(f"{mode} is only available in CPU.")
        raise NotImplementedError(mode)
    elif mode == "hnsw":
        if use_gpu:
            raise NotImplementedError(f"{mode} is only available in CPU.")
        else:
            M = 16
            index = faiss.IndexHNSWFlat(d, M)
            index.hnsw.efConstruction = 80
            index.verbose = True
            index.hnsw.search_bounded_queue = True
    else:
        raise ValueError(mode.lower())

    # From CPU index to GPU index
    if use_gpu:
        print("Copy index to \033[93mGPU\033[0m.")
        index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, index, GPU_OPTIONS)

    # Train index
    start_time = time.time()
    print("Training Index using ", end="")
    if max_nitem_train is not None:
        print(f"{100*max_nitem_train/N:>5.2f}% of randomly selected data...")
        # shuffle and reduce training data
        sel_tr_idx = np.random.permutation(len(train_data))
        sel_tr_idx = sel_tr_idx[:max_nitem_train]
        index.train(train_data[sel_tr_idx, :])
    else:
        print("all the data...")
        index.train(train_data)  # Actually do nothing for {'l2', 'hnsw'}
    print("Elapsed time: ", end="")
    print(f'{time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}.')

    # N probe
    index.nprobe = n_probe

    # Add data to index
    start_time = time.time()
    print("Adding the Fingerprints to the Index...")
    index.add(train_data)
    print(
        f"{index.ntotal:>10,} fingerprints are added in total to the Index in ", end=""
    )
    print(f'{time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

    return index
