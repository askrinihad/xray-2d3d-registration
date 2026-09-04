"""
Compare the classical (iterative gradient-descent) registration approach
against the learned PoseCNN -- and, critically, test the learned model's
generalization to a held-out subject it never saw during training.

Requires a trained model at results/pose_cnn.pt -- run train_pose_cnn.py
first.

Run:
    python src/registration/compare_classical_vs_learned.py
"""
import time
import torch
from tqdm import tqdm

from nanodrr.drr import render
from nanodrr.metrics import NormalizedCrossCorrelation2d
from nanodrr.registration import Registration

from common import (
    load_subject, build_camera, sample_random_poses,
    pose_to_rt_inv, denormalize_pose, HEIGHT, WIDTH,
)
from pose_cnn import PoseCNN

TRAIN_SUBJECT_ID = 1   # one of the subjects the model WAS trained on
TEST_SUBJECT_ID = 6    # held out entirely -- never seen during training


def classical_register(subject, k_inv, sdd_t, true_rt_inv, init_rt_inv,
                        max_iters=500, convergence=0.999):
    reg = Registration(subject, init_rt_inv, k_inv, sdd_t, HEIGHT, WIDTH)
    opt = torch.optim.Adam(
        [{"params": [reg._rot], "lr": 5e-2}, {"params": [reg._xyz], "lr": 1e1}],
        maximize=True,
    )
    ncc = NormalizedCrossCorrelation2d()
    with torch.no_grad():
        target = render(subject, k_inv, true_rt_inv, sdd_t, HEIGHT, WIDTH).sum(dim=1, keepdim=True)

    start = time.time()
    n_used = max_iters
    for i in range(max_iters):
        opt.zero_grad()
        pred = reg()
        loss = ncc(target, pred)
        loss.backward()
        opt.step()
        if loss > convergence:
            n_used = i + 1
            break
    elapsed = time.time() - start
    return reg.pose.detach(), n_used, elapsed


def evaluate_on_subject(subject, k_inv, sdd_t, model, device, n_test=10, label=""):
    torch.manual_seed(0)  # same test poses used for both subjects, for a fair comparison
    true_rot, true_trans = sample_random_poses(n_test, device)

    init_rot = torch.zeros(1, 3, device=device)
    init_trans = torch.tensor([[0.0, 850.0, 0.0]], device=device)

    rows = []
    for i in tqdm(range(n_test), desc=label):
        rot_i = true_rot[i:i + 1]
        trans_i = true_trans[i:i + 1]
        true_rt_inv = pose_to_rt_inv(subject, rot_i, trans_i)
        init_rt_inv = pose_to_rt_inv(subject, init_rot, init_trans)

        _, n_iters, elapsed_classical = classical_register(
            subject, k_inv, sdd_t, true_rt_inv, init_rt_inv
        )

        target_img = render(subject, k_inv, true_rt_inv, sdd_t, HEIGHT, WIDTH).sum(dim=1, keepdim=True)
        start = time.time()
        with torch.no_grad():
            pred_norm = model(target_img)
        pred_rot, pred_trans = denormalize_pose(pred_norm)
        elapsed_learned = time.time() - start

        rows.append({
            "classical_iters": n_iters,
            "classical_time_s": elapsed_classical,
            "learned_time_s": elapsed_learned,
            "learned_rot_err_deg": (pred_rot - rot_i).abs().mean().item(),
            "learned_trans_err_mm": (pred_trans - trans_i).abs().mean().item(),
        })

    print(f"\n[{label}] {'iters':>6} {'classical_s':>12} {'learned_s':>10} {'rot_err_deg':>12} {'trans_err_mm':>13}")
    for r in rows:
        print(f"{r['classical_iters']:>6} {r['classical_time_s']:>12.3f} "
              f"{r['learned_time_s']:>10.5f} {r['learned_rot_err_deg']:>12.2f} "
              f"{r['learned_trans_err_mm']:>13.2f}")

    avg_rot_err = sum(r["learned_rot_err_deg"] for r in rows) / n_test
    avg_trans_err = sum(r["learned_trans_err_mm"] for r in rows) / n_test
    avg_classical = sum(r["classical_time_s"] for r in rows) / n_test
    avg_learned = sum(r["learned_time_s"] for r in rows) / n_test
    print(f"[{label}] avg rot err: {avg_rot_err:.2f} deg | avg trans err: {avg_trans_err:.2f} mm "
          f"| speedup: {avg_classical / avg_learned:.0f}x")
    return avg_rot_err, avg_trans_err


def load_subject_with_retry(sid, device, max_attempts=4):
    for attempt in range(1, max_attempts + 1):
        try:
            subject, _ = load_subject(subject_id=sid, device=device)
            return subject
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait = 5 * attempt
            print(f"  download of subject {sid} failed ({e}); retrying in {wait}s "
                  f"(attempt {attempt}/{max_attempts})...")
            time.sleep(wait)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    k_inv, sdd_t = build_camera(device)

    model = PoseCNN().to(device)
    model.load_state_dict(torch.load("results/pose_cnn.pt", map_location=device))
    model.eval()

    print(f"Loading subject {TRAIN_SUBJECT_ID} (seen during training)...")
    train_subject = load_subject_with_retry(TRAIN_SUBJECT_ID, device)
    print(f"Loading subject {TEST_SUBJECT_ID} (HELD OUT, never seen during training)...")
    test_subject = load_subject_with_retry(TEST_SUBJECT_ID, device)

    seen_err = evaluate_on_subject(
        train_subject, k_inv, sdd_t, model, device, label=f"seen subject {TRAIN_SUBJECT_ID}"
    )
    unseen_err = evaluate_on_subject(
        test_subject, k_inv, sdd_t, model, device, label=f"HELD-OUT subject {TEST_SUBJECT_ID}"
    )

    print("\n=== Generalization summary ===")
    print(f"Seen subject     -> rot err {seen_err[0]:.2f} deg, trans err {seen_err[1]:.2f} mm")
    print(f"Held-out subject -> rot err {unseen_err[0]:.2f} deg, trans err {unseen_err[1]:.2f} mm")
    print("\nIf the held-out numbers are close to the seen-subject numbers, the model")
    print("is genuinely generalizing. If they're much worse, that's evidence of")
    print("overfitting to the training subjects -- an honest, useful result either way.")


if __name__ == "__main__":
    main()
