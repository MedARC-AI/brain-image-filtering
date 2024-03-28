import os
import numpy as np
import pandas as pd
import torch
import time
import pickle
import random
from tqdm import tqdm

class SemDeDup:
    def __init__(self, args):
        self.args = args
        random.seed(args.seed)

    def _contains_duplicates(self, arr):
        return len(np.unique(arr)) != len(arr)

    def semdedup(self, cluster, cluster_reps):
        st = time.time()
        cluster_reps.to('cpu')
        pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
        del cluster_reps
        pair_w_sim_matrix.fill_diagonal_(0.0)
        assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]
        image_urls = cluster[:, 0]
        assert not self._contains_duplicates(image_urls)
        triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)
        M = torch.max(triu_sim_mat, dim=0)[0].cpu()
        print(f"Step time: {time.time()-st}(s)")
        return M

    def process_clusters(self):
        st = time.time()
        embs = np.load(self.args.embs_memory_loc)
        step_time = []
        for cluster_id in tqdm(range(self.args.num_clusters)):
            step_st = time.time()
            df_file_loc = os.path.join(
                self.args.save_loc, f"./clusters/dataframes/cluster_{cluster_id}.pkl"
            )
            print(df_file_loc)
            os.makedirs(os.path.dirname(df_file_loc), exist_ok=True)
            if os.path.exists(df_file_loc):
                print(f"{df_file_loc} exists, moving on")
                continue
            cluster_i = np.load(
                os.path.join(
                    self.args.sorted_clusters_path, f"sorted_cluster_{cluster_id}.npy"
                )
            )
            cluster_size = cluster_i.shape[0]
            print("cluster_size: ", cluster_size)
            if cluster_size == 0:
                print("cluster_size is 0")
                continue
            if cluster_size == 1:
                points_to_remove_df = pd.DataFrame()
                points_to_remove_df["indices"] = [0]
                for eps in self.args.eps_list:
                    points_to_remove_df[f"eps={eps}"] = [False]
                if self.args.save_loc != "":
                    with open(df_file_loc, "wb") as file:
                        pickle.dump(points_to_remove_df, file)
                print("DONE cluster_id ", cluster_id)
                continue
            clutser_items_indices = list(range(cluster_size))
            if self.args.which_to_keep.lower() == "random":
                random.shuffle(clutser_items_indices)
                cluster_i = cluster_i[clutser_items_indices]
            if self.args.which_to_keep.lower() == "easy":
                clutser_items_indices = clutser_items_indices[::-1]
                cluster_i = cluster_i[clutser_items_indices]
            cluster_ids = cluster_i[:, 1].astype("int32")
            cluster_reps = embs[cluster_ids]
            cluster_reps = torch.tensor(cluster_reps)
            M = self.semdedup(cluster_i, cluster_reps)
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = clutser_items_indices
            for eps in self.args.eps_list:
                eps_points_to_remove = M > 1 - eps
                points_to_remove_df[f"eps={eps}"] = eps_points_to_remove
            if self.args.save_loc != "":
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)
            step_time.append(time.time() - step_st)
            print("DONE cluster: ", cluster_id)
        print(f"DONE in {((time.time()-st)/60):.2f} minutes")
        return