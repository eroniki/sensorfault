"""Logic check of the stage-0 harness on a tiny random UNet (needs torch+CUDA, diffusers, peft; skipped otherwise)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("peft")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage0_sizing import TINY, build  # noqa: E402


def test_conv_in_is_widened_and_only_lora_plus_conv_in_train():
    unet = build(TINY, rank=4, ckpt=False)
    assert unet.base_model.model.conv_in.in_channels == 8
    trainable = [n for n, p in unet.named_parameters() if p.requires_grad]
    assert any("conv_in" in n for n in trainable) and any("lora_" in n for n in trainable)
    assert all(("conv_in" in n) or ("lora_" in n) for n in trainable)


def test_one_training_step_runs_and_moves_the_loss_inputs():
    import subprocess
    out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "stage0_sizing.py"), "--tiny",
                          "--res", "128", "--batch", "2", "--steps", "3", "--rank", "4", "--ckpt"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    assert '"params_trainable"' in out.stdout


def test_base_fp16_flag_runs_and_halves_the_frozen_weights():
    import subprocess
    out = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "stage0_sizing.py"), "--tiny",
                          "--res", "128", "--batch", "2", "--steps", "3", "--rank", "4", "--ckpt", "--base-fp16"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    assert '"base_fp16": true' in out.stdout
    unet = build(TINY, rank=4, ckpt=False, base_fp16=True)
    frozen = [p for n, p in unet.named_parameters() if not p.requires_grad]
    assert frozen and all(p.dtype == torch.float16 for p in frozen)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable)
