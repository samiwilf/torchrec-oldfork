import numpy as np
import matplotlib.pyplot as plt

import os
dir = 'Plots'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))


stats_pre_hash = np.load("stats_pre_hash.npy", allow_pickle=True).item()
stats_post_hash = np.load("stats_post_hash.npy", allow_pickle=True).item()

for scale_type in ["Most_Access_Embeddings", "All_Embeddings"]:
    for k, (pre, post) in enumerate(zip(stats_pre_hash, stats_post_hash)):

        pre = stats_pre_hash[str(k)]
        post = stats_post_hash[str(k)]

        # Skip if table didn't use 1-hot to N vec hashing
        if np.amax(pre) <= 0.0:
            continue

        pre = np.sort(pre)[::-1]
        post = np.sort(post)[::-1]

        #x_limit = int(min(len(pre)*0.20, 10000))
        x_limit = int(len(pre))
        if scale_type == "Most_Access_Embeddings":
            try:
                x_limit = np.where(pre < pre[0]*0.002)[0][0]
            except:
                pass

        plt.title(f"Table {k} Pre Hash Reads")
        plt.xlabel("Embeddings, sorted by reads.")
        plt.ylabel("Reads")
        plt.ylim(0, int(1.1*np.amax(pre)))
        plt.xlim(0, x_limit)
        plt.plot(pre, color ="blue")
        plt.savefig(f"Plots/{scale_type}_Table_{k}_pre_hash.png")
        plt.clf()

        plt.ylim(0, int(1.1*np.amax(post)))
        plt.xlim(0, x_limit)
        plt.title(f"Table {k} Post Hash Reads")
        plt.xlabel("Embeddings, sorted by reads.")
        plt.ylabel("Reads")
        plt.plot(post, color ="red")
        plt.savefig(f"Plots/{scale_type}_Table_{k}_post_hash.png")
        plt.clf()
