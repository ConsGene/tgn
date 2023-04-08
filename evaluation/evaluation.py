import math

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, scaleUtil, scale_label, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_ap_raw, val_auc, val_auc_raw = [], [], [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

      sources_batch = data.sources[s_idx:e_idx]           #src_l_cut
      destinations_batch = data.destinations[s_idx:e_idx] #dst_l_cut
      timestamps_batch = data.timestamps[s_idx:e_idx]     #ts_l_cut
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)  #dst_l_fake

      pos_prob, neg_prob = model.compute_edge_values(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([data.labels[s_idx: e_idx], np.zeros(size)])
      true_label_raw = np.concatenate([data.raw_labels[s_idx: e_idx], np.zeros(size)])

      val_ap.append(mean_absolute_error(true_label, pred_score))
      val_auc.append(r2_score(true_label, pred_score))
      if scale_label != 'none':
        pred_score_raw = scaleUtil.convert_to_raw_label_score(np.concatenate([destinations_batch, negative_samples]), pred_score)
        val_ap_raw.append(mean_absolute_error(true_label_raw, pred_score_raw))
        val_auc_raw.append(r2_score(true_label_raw, pred_score_raw))

  return np.mean(val_ap), np.mean(val_ap_raw), np.mean(val_auc), np.mean(val_auc_raw)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = r2(data.labels, pred_prob)
  return auc_roc
