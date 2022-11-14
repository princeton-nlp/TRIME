import argparse
import os
import numpy as np
import faiss
import time

# the implementation refers to knnlm

parser = argparse.ArgumentParser()
parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore-fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=32, help='number of clusters to query')
parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
parser.add_argument('--num_keys_to_add_at_a_time', default=500000, type=int,
                    help='can only load a certain amount of data to memory at a time.')
parser.add_argument('--starting_point', type=int, default=0, help='index to start adding keys at')
parser.add_argument('--load-multiple-files', default=False, action='store_true')
parser.add_argument('--multiple-key-files', type=str, default=None)
parser.add_argument('--multiple-val-files', type=str, default=None)
parser.add_argument('--multiple-files-size', type=str, default=None)
parser.add_argument('--concat-file-path', type=str, default=None)
parser.add_argument('--dist', type=str, default='l2')

args = parser.parse_args()

print(args)

res = faiss.StandardGpuResources()

# load the saved keys and values
if args.dstore_fp16:
    if args.load_multiple_files:
        assert args.multiple_key_files is not None and args.multiple_val_files is not None
        key_files = args.multiple_key_files.split(':')
        val_files = args.multiple_val_files.split(':')
        sizes = [int(size) for size in args.multiple_files_size.split(':')]
        print(sizes)
        key_list = [np.memmap(key_file, dtype=np.float16, mode='r', shape=(sizes[idx], args.dimension)) for
                    idx, key_file in enumerate(key_files)]
        val_list = [np.memmap(val_file, dtype=np.int, mode='r', shape=(sizes[idx], 1)) for idx, val_file in
                    enumerate(val_files)]
        concat_size = np.sum(sizes)

        keys = np.memmap(args.concat_file_path + '/keys.npy', dtype=np.float16, mode='w+',
                         shape=(concat_size, args.dimension))
        vals = np.memmap(args.concat_file_path + '/vals.npy', dtype=np.int, mode='w+', shape=(concat_size, 1))

        cur_size = 0
        for idx, size in enumerate(sizes):
            print('write {} to {}'.format(cur_size, cur_size + size))
            keys[cur_size: cur_size + size, :] = key_list[idx][:, :]
            vals[cur_size: cur_size + size, :] = val_list[idx][:, :]
            cur_size += size

        exit()

if args.dstore_fp16:
    print('load dstore fp16', args.dstore_size, args.dimension)
    keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='r',
                         shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))
else:
    keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float32, mode='r',
                     shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

print('done.')

if not os.path.exists(args.faiss_index + ".trained"):
    # Initialize faiss index
    if args.dist == 'l2':
        quantizer = faiss.IndexFlatL2(args.dimension)
        index = faiss.IndexIVFPQ(quantizer, args.dimension,
                                 args.ncentroids, args.code_size, 8, faiss.METRIC_L2)
    elif args.dist == 'ip':
        quantizer = faiss.IndexFlatIP(args.dimension)
        index = faiss.IndexIVFPQ(quantizer, args.dimension,
                                 args.ncentroids, args.code_size, 8, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = args.probe

    # TODO, we may remove useFloat16 when the GPU satisfy the condition
    print('Start put index to gpu')
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)

    print('Training Index')
    np.random.seed(args.seed)
    random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
    start = time.time()
    # Faiss does not handle adding keys in fp16 as of writing this.
    # gpu_index.train(keys[random_sample].astype(np.float32))
    print(random_sample[:10])
    print(keys[random_sample][:10])
    gpu_index.train(keys[random_sample].astype(np.float32))
    print('Training took {} s'.format(time.time() - start))

    print('Writing index after training')
    start = time.time()
    faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), args.faiss_index + ".trained")
    print('Writing index took {} s'.format(time.time() - start))

print('Adding Keys')
index = faiss.read_index(args.faiss_index + ".trained")
co = faiss.GpuClonerOptions()
co.useFloat16 = True
gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
cpu_index = index

start = args.starting_point
start_time = time.time()
while start < args.dstore_size:
    end = min(args.dstore_size, start + args.num_keys_to_add_at_a_time)
    to_add = keys[start:end].copy()
    cpu_index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
    start += args.num_keys_to_add_at_a_time

    if (start % 1000000) == 0:
        print('Added %d tokens so far' % start)
        print('Writing Index', start)
        faiss.write_index(cpu_index, args.faiss_index)

print("Adding total %d keys" % end)
print('Adding took {} s'.format(time.time() - start_time))
print('Writing Index')
start = time.time()
faiss.write_index(cpu_index, args.faiss_index)
print('Writing index took {} s'.format(time.time() - start))
