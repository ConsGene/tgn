import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

WEEK_IN_SECS = 7*24*60*60

def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  raw_df = pd.read_csv(data_name)
  
  raw_df['ts'] = raw_df['optimized_date'].apply(pd.to_datetime).astype('int64') // 10**9 // (1*WEEK_IN_SECS)
  df = raw_df.groupby(['member_id','merchant_format_name', 'ts']).agg({'transaction_amount':'sum', 'category':'first', 'member_home_state':'first'}).reset_index()
  
  df = df.sort_values('ts')
  result_df = pd.DataFrame({'u': df['member_id'].astype('category').cat.codes.tolist(),
                            'i': df['merchant_format_name'].astype('category').cat.codes.tolist(),
                            'ts': df['ts'].tolist(),
                            'label': df['transaction_amount'].tolist(),
                            'idx': df.index.values.tolist()})
  # return result_df, np.array([df['category'].astype('category').cat.codes.tolist(), df['member_home_state'].astype('category').cat.codes.tolist()]).T
  return result_df, pd.get_dummies(df['member_home_state'], prefix='_state'), pd.get_dummies(df['category'], prefix='_cat')


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  df, u_feat, i_feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  ## TODO: empty node feature?
  max_idx = max(new_df.u.max(), new_df.i.max())
  node_feat_dim = max(u_feat.shape[1], i_feat.shape[1])
  if node_feat_dim % 2 != 0:
    node_feat_dim += 1
      
  feat = np.zeros((len(new_df), node_feat_dim))
  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  #max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, feat.shape[1]))

  new_df.to_csv(OUT_DF)
  # edge features
  np.save(OUT_FEAT, feat)
  # node features
  np.save(OUT_NODE_FEAT, rand_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='u2k_i200')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)
