import argparse
import os
import numpy as np
import faiss
import time


parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--do_train_only', action='store_true')
parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore_fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
parser.add_argument('--num_keys_to_add_at_a_time', default=1000000, type=int,
                    help='can only load a certain amount of data to memory at a time.')
parser.add_argument('--starting_point', type=int, help='index to start adding keys at')
parser.add_argument('--dist', type=str, default='l2')
args = parser.parse_args()

if args.dstore_fp16:
    keys = np.memmap(args.dstore_mmap+'-fp16_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))
else:
    keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

args.faiss_index = args.faiss_index + '.' + args.dist

if not os.path.exists(args.faiss_index+".trained"):
    # Initialize faiss index
    if args.dist in ['l2']:
        quantizer = faiss.IndexFlatL2(args.dimension)
        index = faiss.IndexIVFPQ(quantizer, args.dimension,
            args.ncentroids, args.code_size, 8, faiss.METRIC_L2)
    elif args.dist in ['ip', 'cos']:
        quantizer = faiss.IndexFlatIP(args.dimension)
        index = faiss.IndexIVFPQ(quantizer, args.dimension,
            args.ncentroids, args.code_size, 8, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = args.probe

    print('Training Index')
    np.random.seed(args.seed)
    random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
    start = time.time()
    # Faiss does not handle adding keys in fp16 as of writing this.
    to_train = keys[random_sample].astype(np.float32)
    print('Finished loading train vecs')
    if args.dist == 'cos':
        faiss.normalize_L2(to_train)
    index.train(to_train)
    print('Training took {} s'.format(time.time() - start))

    print('Writing index after training')
    start = time.time()
    faiss.write_index(index, args.faiss_index+".trained")
    print('Writing index took {} s'.format(time.time()-start))

if args.do_train_only:
    print('Only do training')
    exit(0)

print('Adding Keys')
index = faiss.read_index(args.faiss_index+".trained")

if args.use_gpu:
    print('Moving to GPUs')
    res = faiss.StandardGpuResources()

    # co = faiss.GpuMultipleClonerOptions()
    # co.useFloat16 = True
    # co.shard = True
    # index = faiss.index_cpu_to_all_gpus(index, co=co)

    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    faiss.index_cpu_to_gpu(res, 0, index, co)

    print('Now index is on GPUs')

start = args.starting_point
start_time = time.time()
while start < args.dstore_size:
    end = min(args.dstore_size, start+args.num_keys_to_add_at_a_time)
    to_add = keys[start:end].copy().astype(np.float32)
    if args.dist == 'cos':
        faiss.normalize_L2(to_add)
    index.add_with_ids(to_add, np.arange(start, end))
    start += args.num_keys_to_add_at_a_time

    if (start % 1000000) == 0:
        print('Added %d tokens so far (took %f s)' % (start, time.time() - start_time))
        print('Writing Index', start)
        faiss.write_index(index, args.faiss_index)

print("Adding total %d keys" % start)
print('Adding took {} s'.format(time.time() - start_time))
print('Writing Index')
start = time.time()
faiss.write_index(index, args.faiss_index)
print('Writing index took {} s'.format(time.time()-start))
