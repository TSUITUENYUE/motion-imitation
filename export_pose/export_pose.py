#!/usr/bin/env python3
"""
Export D-SMAL *.npz → 91-value CSV compatible with DogWalkParser / Go2 retargeter.
All joint positions are world-space; root (x,y) is normalised so frame-0 = (0,0).
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ------------------------------------------------------------------ #
# 1.  SMAL joint list and DogWalk column order                        #
# ------------------------------------------------------------------ #
DEFAULT_MODEL_JOINT_NAMES = [
    "global_orient","pelvis","spine_01","spine_02","spine_03","neck_01","head_01",
    "r_up_shoulder","l_low_shoulder","l_elbow","l_wrist","l_up_shoulder",
    "r_low_shoulder","r_elbow","r_wrist","xxxx","xxxx","xxxx",
    "l_hip","l_knee","l_ankle","xxxx","r_hip","r_knee","r_ankle","xxxx",
    "tail_01","tail_02","tail_03","tail_04","tail_05","tail_06","tail_07",
    "tail_08","tail_09",
]
if len(DEFAULT_MODEL_JOINT_NAMES) != 35:
    raise ValueError("DEFAULT_MODEL_JOINT_NAMES must contain 35 names")

LEG_NAMES_FOR_CSV = [
    "r_low_shoulder","r_elbow","r_wrist",
    "l_low_shoulder","l_elbow","l_wrist",
    "r_hip","r_knee","r_ankle",
    "l_hip","l_knee","l_ankle",
]

# ------------------------------------------------------------------ #
# 2.  Helpers                                                        #
# ------------------------------------------------------------------ #
def quat_wxyz_from_rotmat(rot_mats: np.ndarray) -> np.ndarray:
    """(N,3,3) → (N,4) in w,x,y,z order."""
    if rot_mats.ndim == 4: rot_mats = rot_mats.squeeze(0)
    if rot_mats.ndim == 2: rot_mats = rot_mats[None]
    q_xyzw      = R.from_matrix(rot_mats).as_quat()
    q_wxyz      = np.empty_like(q_xyzw)
    q_wxyz[:,0] = q_xyzw[:,3]
    q_wxyz[:,1:] = q_xyzw[:,:3]
    return q_wxyz.squeeze(0) if q_wxyz.shape[0]==1 else q_wxyz

# ------------------------------------------------------------------ #
# 3.  Main routine                                                   #
# ------------------------------------------------------------------ #
def convert_sequence(src_dir: Path, out_txt: Path) -> None:
    # --- import SMAL regardless of where we run --------------------
    proj_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, proj_root.as_posix())
    from smal.smal_tpg import SMAL      # noqa: E402

    npzs = sorted(src_dir.glob("*.npz"))
    if not npzs:
        sys.exit(f"No *.npz in {src_dir}")

    device                  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smal                    = SMAL().to(device)
    num_shape_betas         = smal.num_betas
    num_limb_betas          = smal.num_betas_logscale

    root_idx                = DEFAULT_MODEL_JOINT_NAMES.index("global_orient")
    leg_indices             = [DEFAULT_MODEL_JOINT_NAMES.index(n) for n in LEG_NAMES_FOR_CSV]

    first_xyz_offset         = None           # ← captured from very first frame

    with open(out_txt, "w") as fout:
        for npz in tqdm(npzs, unit="frame"):
            data           = np.load(npz, allow_pickle=True)

            # ------------- pose matrices -----------------------------------
            poseR_np       = data["pred_pose"]
            if poseR_np.ndim == 3: poseR_np = poseR_np[None]
            pose_t         = torch.from_numpy(poseR_np).float().to(device)

            # ------------- root translation (world) ------------------------
            root           = data["pred_trans"].squeeze()
            if first_xyz_offset is None:
                first_xyz_offset = root.copy()
            root_norm      = root.copy()
            root_norm -= first_xyz_offset

            # ------------- feed through SMAL to get world joints -----------
            trans_t        = torch.from_numpy(root_norm[None]).float().to(device)

            betas          = data.get("pred_betas", np.zeros(num_shape_betas)).squeeze()
            betas          = np.pad(betas, (0,max(0,num_shape_betas-len(betas))))[:num_shape_betas]
            betas_t        = torch.from_numpy(betas[None]).float().to(device)

            betas_limbs    = data.get("pred_betas_limbs", np.zeros(num_limb_betas)).squeeze()
            betas_limbs    = np.pad(betas_limbs,(0,max(0,num_limb_betas-len(betas_limbs))))[:num_limb_betas]
            betas_limbs_t  = torch.from_numpy(betas_limbs[None]).float().to(device)

            with torch.no_grad():
                _, outs    = smal(beta=betas_t, betas_limbs=betas_limbs_t,
                                   pose=pose_t, trans=trans_t)

            j3d_world      = outs[3][0].cpu().numpy()             # (35,3)

            # ------------- quaternions (local) ----------------------------
            q_all          = quat_wxyz_from_rotmat(poseR_np)

            # ------------- compose one CSV row ----------------------------
            row  = list(root_norm) + list(q_all[root_idx])
            for idx in leg_indices:
                row += j3d_world[idx].tolist() + q_all[idx].tolist()

            fout.write(",".join(f"{v:.6f}" for v in row) + "\n")

    exp_cols = 3+4+len(LEG_NAMES_FOR_CSV)*(3+4)
    print(f"✅  {out_txt} written  |  {len(npzs)} frames, {exp_cols} numbers/line")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="D-SMAL *.npz → 91-col CSV for DogWalkParser.")
    ap.add_argument("breed")
    args = ap.parse_args()
    src_dir = Path("./" + args.breed + "/pred")
    output_txt = Path(args.breed + "_joint_pos.txt")
    convert_sequence(src_dir, output_txt)

