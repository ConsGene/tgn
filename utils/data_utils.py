import numpy as np
import random
import pandas as pd
import torch
import sys
import argparse
from tqdm import tqdm
from sklearn import preprocessing

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels, raw_labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.raw_labels = raw_labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def get_data(dataset_name, scale_label, device, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) 
    
  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values
  scaleUtil = ScaleUtil(scale_label, device)
  labels, raw_labels = scaleUtil.transform_df(graph_df, valid_train_flag)

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels, raw_labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)


  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  # train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
  train_mask = (timestamps <= val_time)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask], raw_labels[train_mask])

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  # validation and test with all edges
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask], raw_labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask], raw_labels[test_mask])


  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))

  return node_features, edge_features, full_data, train_data, val_data, test_data, scaleUtil


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

class ScaleUtil:
    def __init__(self, scale_label, device):
        self.scale_label = scale_label
        if scale_label == 'MinMax':
            self.sscaler = preprocessing.MinMaxScaler
        elif scale_label == 'Quantile':
            self.scaler = preprocessing.QuantileTransformer
        elif scale_label == 'Log':
            self.scaler = preprocessing.StandardScaler
        elif scale_label == 'Cbrt':
            self.scaler = preprocessing.StandardScaler
        
        self.device = device
        self.i2cat = None
        self.scalers_dict = None
    
    def transform_df(self, original_graph_df, valid_train_flag):
        graph_df = original_graph_df.copy()
        graph_df['raw_label'] = graph_df.label.copy()
        
        self.i2cat = graph_df.groupby('i').first().reset_index().set_index('i')['cat'].to_dict()
        for cat in self.i2cat.values():
            orig_labels = graph_df.loc[(graph_df.cat == cat) & valid_train_flag, 'label'].values
            lower = np.quantile(orig_labels, 0.001)
            upper = np.quantile(orig_labels, 0.999)
            graph_df.loc[(graph_df.cat == cat) & valid_train_flag, 'label'] = np.clip(orig_labels, lower, upper)
        
        if self.scale_label == 'none':
            label_l = graph_df.label.values
        else:
            # train_df['abs_label'] = train_df['label'].abs()
            
            # i_maxes = train_df.groupby('i')['abs_label'].max().reset_index().set_index('i')['abs_label'].to_dict()
            # # normalize labels with the max value from training set
            # for i, max_label in i_maxes.items():
            #     g_df.loc[g_df.i == i, 'label'] /= max_label
            # label_l = g_df.label.values
            # def convert_to_raw_label_scale(dst_l_cut, preds):
            #     if isinstance(preds, np.ndarray):
            #         scale = np.array([i_maxes[dst] for dst in dst_l_cut])
            #     else:
            #         scale = torch.tensor([i_maxes[dst] for dst in dst_l_cut], dtype=float, device=device)
            #     return preds * scale, labels * scale
            train_df = graph_df[valid_train_flag]


        self.scalers_dict = {}
        if self.scale_label == 'Cbrt':
            graph_df.label = np.cbrt(graph_df.label.values)
        else:
            for cat in self.i2cat.values():
                cat_train_df = train_df[train_df.cat==cat]
                self.scalers_dict[cat] = self.scaler()
                train_label_vals = self.prepare_transform(cat_train_df.label.values)
                self.scalers_dict[cat].fit(train_label_vals.reshape(-1, 1))
                label_vals = self.prepare_transform(graph_df.loc[graph_df.cat == cat]['label'].values)        
                graph_df.loc[graph_df.cat == cat, 'label'] = self.scalers_dict[cat].transform(label_vals.reshape(-1, 1))
        
        label_l = graph_df.label.values
        raw_label_l = graph_df.raw_label.values
        return label_l, raw_label_l
        
    def prepare_transform(self, label_vals):
        if self.scale_label == 'Log':
            label_vals = np.sign(label_vals) * np.log(np.abs(label_vals)+1)
        elif self.scale_label == 'Cbrt':
            label_vals = np.cbrt(label_vals)
        return label_vals

    def convert_to_raw_label_scale(self, dst_l_cut, preds):
        if (self.i2cat is None) or (self.scalers_dict is None):
            raise RuntimeError("self.i2cat or self.scalers_dict is None. Run ScaleUtil.transform_df function first")
        raw_preds = []
        for dst, pred in zip(dst_l_cut, preds):
            cat = self.i2cat[dst]
            if self.scale_label == 'Cbrt':
                raw = np.power(pred, 3)
                raw_preds.append(raw)
            else:
                raw = self.scalers_dict[cat].inverse_transform(pred.reshape(1, -1))
                if self.scale_label == 'Log':
                    raw = np.sign(raw) * (np.exp(np.abs(raw))-1)
                elif self.scale_label == 'Cbrt':
                    raw = np.power(raw, 3)
                raw_preds.append(raw[0, 0])
        if isinstance(preds, np.ndarray):
            return np.array(raw_preds)
        else:
            return torch.tensor(raw_preds, dtype=float, device=self.device)
