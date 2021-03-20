import numpy as np


class RetMetric(object):
    def __init__(self, feats, labels):

        if len(feats) == 2 and type(feats) == list:
            """
            feats = [gallery_feats, query_feats]
            labels = [gallery_labels, query_labels]
            """
            self.is_equal_query = False

            self.gallery_feats, self.query_feats = feats
            self.gallery_labels, self.query_labels = labels

        else:
            self.is_equal_query = True
            self.gallery_feats = self.query_feats = feats
            self.gallery_labels = self.query_labels = labels

        self.sim_mat = np.matmul(self.query_feats, np.transpose(self.gallery_feats))
        self.sim_mat_s = np.matmul(self.query_feats, np.transpose(self.query_feats))

    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0

        for i in range(m):
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]

            thresh = np.sort(pos_sim)[-2] if self.is_equal_query else np.max(pos_sim)

            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m

    def re_map(self, k = 1):

         m = len(self.sim_mat)

         match_counter = np.zeros((k,))

         for k_i in range(k):
             s = 0
             for i_i in range(m):
                 ind_sort = np.argsort(self.sim_mat[i_i,:]) #each query's distance
                 ind_sim =self.gallery_labels == self.query_labels[i_i]
                 s += np.sum(ind_sim[ind_sort[1:k_i + 2]])/(k_i+1)
             match_counter[k_i] = s/(m)

         return match_counter