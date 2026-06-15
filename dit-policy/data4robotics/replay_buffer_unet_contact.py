# Replay buffer variant where the contact anchor is concatenated to the
# observation vector instead of being returned as a separate 4th batch element.
#
# obs returned: [qpos(8), contact_norm(3)]  → total 11-dim
# The rest of the pipeline (trainer, agent) sees a standard 3-tuple batch
# and needs zero architectural changes.

import json
import os
import pickle as pkl
import random
import shutil

import numpy as np
import torch
import tqdm
from robobuf import ReplayBuffer as RB
try:
    from tensorflow.io import gfile
except ImportError:
    class _GFileFallback:
        @staticmethod
        def GFile(path, mode):
            return open(path, mode)
    gfile = _GFileFallback()
from torch.utils.data import Dataset, IterableDataset

buf_cache = dict()
BUF_SHUFFLE_RNG = 3904767649

_img_to_tensor = (
    lambda x: torch.from_numpy(x.copy()).permute((0, 3, 1, 2)).float() / 255
)
_to_tensor = lambda x: torch.from_numpy(x).float()


def _cached_load(path):
    global buf_cache
    if path in buf_cache:
        return buf_cache[path]
    with gfile.GFile(path, "rb") as f:
        buf = RB.load_traj_list(pkl.load(f))
    buf_cache[path] = buf
    return buf


def _get_imgs(t, cam_idx, past_frames):
    imgs = []
    while len(imgs) < past_frames + 1:
        imgs.append(t.obs.image(cam_idx)[None])
        if t.prev is not None:
            t = t.prev
    return np.concatenate(imgs, axis=0)


class IterableWrapper(IterableDataset):
    def __init__(self, wrapped_dataset, max_count=float("inf")):
        self.wrapped = wrapped_dataset
        self.ctr, self.max_count = 0, max_count

    def __iter__(self):
        self.ctr = 0
        return self

    def __next__(self):
        if self.ctr > self.max_count:
            raise StopIteration
        self.ctr += 1
        idx = int(np.random.choice(len(self.wrapped)))
        return self.wrapped[idx]


class RobobufReplayBuffer(Dataset):
    """
    Like replay_buffer_contact.RobobufReplayBuffer but the contact anchor
    (3-dim normalized) is appended to the obs vector instead of being
    returned as a separate 4th element.

    Batch format returned:  (imgs_dict, obs_11d), actions, mask
      obs_11d = [qpos(8), contact_norm(3)]

    Requires buf.pkl produced by convert_to_robobuf_contact_hindsight.py
    (i.e. each transition must have a contact_anchor stored in obs.obs).
    """

    def __init__(
        self,
        buffer_path,
        transform=None,
        n_test_trans=500,
        mode="train",
        ac_chunk=1,
        cam_indexes=[0],
        goal_indexes=[],
        goal_geom_prob=0.01,
        past_frames=0,
        ac_dim=7,
    ):
        assert mode in ("train", "test"), "Mode must be train/test"
        buf = _cached_load(buffer_path)
        assert len(buf) > n_test_trans, "Not enough transitions!"

        norm_file = os.path.join(os.path.dirname(buffer_path), "ac_norm.json")
        if os.path.exists(norm_file):
            shutil.copyfile(norm_file, "./ac_norm.json")

        contact_norm_file = os.path.join(os.path.dirname(buffer_path), "contact_norm.json")
        if os.path.exists(contact_norm_file):
            shutil.copyfile(contact_norm_file, "./contact_norm.json")
        print("[RobobufReplayBuffer-UnetContact] contact anchor appended to obs")

        rng = random.Random(BUF_SHUFFLE_RNG)
        index_list = list(range(len(buf)))
        rng.shuffle(index_list)
        index_list = (
            index_list[n_test_trans:] if mode == "train" else index_list[:n_test_trans]
        )

        self.transform = transform
        self.s_a_mask = []
        self.cam_indexes = cam_indexes = list(cam_indexes)
        self.past_frames = past_frames
        self.goal_geom_prob = goal_geom_prob
        self.goal_indexes = set(goal_indexes)
        assert all(g in self.cam_indexes for g in self.goal_indexes)
        print(f"Building {mode} buffer with cam_indexes={cam_indexes}")

        missing_contact = 0
        for idx in tqdm.tqdm(index_list):
            t = buf[idx]

            loop_t, chunked_actions, loss_mask = t, [], []
            for _ in range(ac_chunk):
                chunked_actions.append(loop_t.action[None])
                loss_mask.append(1.0)
                if loop_t.next is None:
                    break
                loop_t = loop_t.next

            if len(chunked_actions) < ac_chunk:
                for _ in range(ac_chunk - len(chunked_actions)):
                    chunked_actions.append(chunked_actions[-1])
                    loss_mask.append(0.0)

            a_t = np.concatenate(chunked_actions, 0).astype(np.float32)
            assert ac_dim == a_t.shape[-1]
            loss_mask = np.array(loss_mask, dtype=np.float32)

            contact_anchor = t.obs.obs.get("contact_anchor", None)
            if contact_anchor is None:
                missing_contact += 1
            self.s_a_mask.append((t, a_t, loss_mask, loop_t, contact_anchor))

        if missing_contact > 0:
            raise RuntimeError(
                f"{missing_contact} transitions have no contact_anchor. "
                "This buffer requires data converted with convert_to_robobuf_contact_hindsight.py."
            )

    def __len__(self):
        return len(self.s_a_mask)

    def __getitem__(self, idx):
        step, a_t, loss_mask, goal, contact_anchor = self.s_a_mask[idx]

        if self.goal_indexes:
            while np.random.uniform() > self.goal_geom_prob and goal.next is not None:
                goal = goal.next

        i_t, o_t = dict(), step.obs.state
        for cam_i, cam_idx in enumerate(self.cam_indexes):
            i_c = _get_imgs(step, cam_idx, self.past_frames)
            if self.goal_indexes:
                g_c = (
                    _get_imgs(goal, cam_idx, 0)
                    if cam_idx in self.goal_indexes
                    else np.zeros_like(i_c[:1])
                )
                i_c = np.concatenate((g_c, i_c), axis=0)
            i_c = _img_to_tensor(i_c)
            if self.transform is not None:
                i_c = self.transform(i_c)
            i_t[f"cam{cam_i}"] = i_c

        # Concatenate contact_anchor (3-dim, already normalized) to qpos (8-dim)
        o_t_with_contact = np.concatenate(
            [o_t.astype(np.float32), contact_anchor.astype(np.float32)]
        )

        o_t_tensor   = _to_tensor(o_t_with_contact)
        a_t_tensor   = _to_tensor(a_t)
        mask_tensor  = _to_tensor(loss_mask)[:, None].repeat((1, a_t.shape[-1]))

        assert mask_tensor.shape[0] == a_t_tensor.shape[0]
        return (i_t, o_t_tensor), a_t_tensor, mask_tensor