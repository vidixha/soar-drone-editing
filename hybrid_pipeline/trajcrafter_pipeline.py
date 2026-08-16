"""Modal app: test TrajectoryCrafter (github.com/TrajectoryCrafter/TrajectoryCrafter)
as an alternative to ReCamMaster for the Stage 1 re-rendering step, since
ReCamMaster hallucinates a fictional building for cam_type 1's ~20deg rotation
(see modal/DEBUG_LOG.md and trace_replication/README.md). TrajectoryCrafter is a
hybrid depth-reprojection + diffusion-inpainting approach (point cloud warp via
DepthCrafter's depth, diffusion only fills genuine occlusion holes), which
should hallucinate less than ReCamMaster's pure end-to-end diffusion synthesis,
and supports dynamic (moving-object) video unlike the static-scene ViewCrafter.

Cost note: this pipeline needs >=28GB VRAM (CogVideoX-Fun-V1.1-5b backbone,
~4x ReCamMaster's Wan2.1-1.3B), which doesn't fit our L4 (24GB) -- needs A100-40GB
at ~$0.000583/s (~$2.10/hr), vs L4's ~$0.80/hr. This is a real cost step-up from
every other experiment in this project so far; the download_weights step below
is CPU-billed and cheap, but do NOT run test_render on GPU without confirming
cost with the user first.

Usage:
  modal run modal/trajcrafter_pipeline.py::download_weights
  modal run modal/trajcrafter_pipeline.py::test_render
"""
import pathlib

import modal

app = modal.App("trajcrafter-test")

weights_volume = modal.Volume.from_name("trajcrafter-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "ffmpeg", "wget")
    .pip_install("setuptools", "wheel")
    # The entire torch-2.4.x cu121 wheel series hardcodes
    # nvidia-cudnn-cu12==9.1.0.70, which has been pulled from the index
    # (confirmed: same failure at both 2.4.0 and 2.4.1). Unpinning entirely
    # let pip resolve a fully-compatible set on its own -- this is exactly
    # what fixed the identical error in trace_anything_pipeline.py, which
    # landed on torch==2.3.1+cu121. The downstream xformers==0.0.27.post2 pin
    # and the is_flash_attention_available patch below were written for
    # torch 2.4.0 specifically -- if this resolves to something else, both
    # may need re-checking.
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu121")
    .run_commands(
        "git clone --depth 1 https://github.com/TrajectoryCrafter/TrajectoryCrafter.git /opt/TrajectoryCrafter",
        # .gitmodules points at git@github.com (SSH), which fails with no key in
        # a fresh container -- clone DepthCrafter over HTTPS instead and place
        # it at the path the submodule would have used.
        "git clone --depth 1 https://github.com/Tencent/DepthCrafter.git /opt/TrajectoryCrafter/DepthCrafter",
    )
    .pip_install(
        "Pillow", "einops", "safetensors", "timm", "tomesd", "torchdiffeq", "torchsde",
        "decord", "datasets", "numpy", "scikit-image", "opencv-python-headless",
        "omegaconf", "SentencePiece", "albumentations", "imageio[ffmpeg]", "imageio[pyav]",
        "tensorboard", "beautifulsoup4", "ftfy", "func_timeout", "deepspeed",
        "accelerate>=0.25.0", "diffusers==0.30.1", "transformers==4.47", "av==12.0.0",
        "gradio", "huggingface_hub<1.0", "hf_transfer",
    )
    # gradio/deepspeed/accelerate transitively pulled a newer, incompatible
    # torchvision (0.28.0+cu130) that silently overrode our torch-paired
    # pin above and dropped torchvision.io.write_video -- re-pin last so it wins.
    .pip_install("torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu121")
    # xformers was unpinned above and resolved to a wheel built for a much
    # newer torch/cuda/python combo (2.10.0+cu128), hard-failing on our
    # torch==2.4.0 (missing torch.distributed.distributed_c10d.GroupName,
    # added later). Pin to the release actually paired with torch 2.4.0.
    .pip_install("xformers==0.0.27.post2", index_url="https://download.pytorch.org/whl/cu121")
    # diffusers==0.30.1 calls torch.backends.cuda.is_flash_attention_available(),
    # which doesn't exist in torch==2.4.0 (added in a later torch release).
    # A sitecustomize.py shim was tried first but Modal's own runtime injects
    # its own /pkg/sitecustomize.py earlier in sys.path, silently shadowing
    # ours -- so patch inference.py directly at build time instead.
    .run_commands(
        'printf \'import torch\\nif not hasattr(torch.backends.cuda, "is_flash_attention_available"):\\n    torch.backends.cuda.is_flash_attention_available = lambda: False\\n\' '
        '| cat - /opt/TrajectoryCrafter/inference.py > /tmp/inference_patched.py '
        '&& mv /tmp/inference_patched.py /opt/TrajectoryCrafter/inference.py '
        '&& head -5 /opt/TrajectoryCrafter/inference.py'
    )
    .add_local_dir(
        str(pathlib.Path(__file__).parent.parent / "trace_replication" / "src"),
        remote_path="/opt/trace_src",
    )
    .add_local_file(
        "/home/akshata/projects/soar_drone_editing/trace_replication/data/mavrec_raw/stage1_source/scene_1.mp4",
        remote_path="/opt/scene_1.mp4",
    )
    .add_local_file("/tmp/vecteezy_car_crop.mp4", remote_path="/opt/car_microvideo.mp4")
    .add_local_file("/tmp/triplet_clips/clip_s4_car.mp4", remote_path="/opt/harbor.mp4")
)


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, timeout=1800, cpu=2)
def download_weights():
    """CPU-only: downloads all 5 checkpoints (TrajectoryCrafter, DepthCrafter,
    SVD-img2vid-xt, CogVideoX-Fun-V1.1-5b-InP, BLIP2-opt-2.7b) into the volume.
    This is a MUCH bigger download than ReCamMaster's (~30-40GB vs ~20.5GB,
    dominated by the 5B-parameter CogVideoX-Fun backbone) -- still CPU-billed
    and cheap in dollar terms, but will take longer wall-clock time even with
    hf_transfer enabled.
    """
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download

    targets = [
        ("TrajectoryCrafter/TrajectoryCrafter", "TrajectoryCrafter"),
        ("tencent/DepthCrafter", "DepthCrafter"),
        ("stabilityai/stable-video-diffusion-img2vid-xt", "stable-video-diffusion-img2vid-xt"),
        ("alibaba-pai/CogVideoX-Fun-V1.1-5b-InP", "CogVideoX-Fun-V1.1-5b-InP"),
        ("Salesforce/blip2-opt-2.7b", "blip2-opt-2.7b"),
    ]
    for repo_id, subdir in targets:
        local_dir = pathlib.Path(WEIGHTS_DIR) / subdir
        marker = local_dir / ".download_complete"
        if marker.exists():
            print(f"{repo_id} already cached")
            continue
        print(f"downloading {repo_id} ...")
        snapshot_download(repo_id, local_dir=str(local_dir))
        marker.touch()
        weights_volume.commit()
        print(f"done: {repo_id}")

    print("all TrajectoryCrafter-stack weights persisted to volume")


@app.function(image=image, cpu=1, timeout=120)
def check_torchvision_video():
    """CPU-only, no volume needed: debug why torchvision.io.write_video is
    missing -- cheaper to iterate here than re-running the full A100 pipeline
    each time to hit the same save_video() call."""
    import torchvision
    import av
    print("torchvision version:", torchvision.__version__)
    print("av version:", av.__version__)
    print("has write_video:", hasattr(torchvision.io, "write_video"))
    print("torchvision.io contents:", [x for x in dir(torchvision.io) if "video" in x.lower()])
    try:
        import torchvision.io as tio
        print("_HAS_VIDEO_OPT:", getattr(tio, "_HAS_VIDEO_OPT", "N/A"))
    except Exception as e:
        print("error inspecting:", e)

    import site
    print("sitepackages:", site.getsitepackages())
    import subprocess as sp
    print("sitecustomize.py exists:", sp.run(["find", "/", "-name", "sitecustomize.py"], capture_output=True, text=True).stdout)
    import sys
    print("sys.path:", sys.path)

    import torch
    print("has is_flash_attention_available:", hasattr(torch.backends.cuda, "is_flash_attention_available"))
    try:
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel  # noqa: F401
        print("OK: diffusers.models.unets.unet_2d_condition imports")
    except Exception as e:
        print("FAILED diffusers unet_2d_condition import:", e)


@app.function(image=image, cpu=1, timeout=120)
def check_inference_patch():
    """CPU-only: confirm the is_flash_attention_available patch prepended to
    inference.py actually takes effect before diffusers' unet_2d_condition
    import chain runs, without spending any GPU time."""
    import subprocess
    result = subprocess.run(
        ["python", "inference.py", "--help"],
        cwd="/opt/TrajectoryCrafter", capture_output=True, text=True,
    )
    print("returncode:", result.returncode)
    print("--- stdout (last 20 lines) ---")
    print("\n".join(result.stdout.splitlines()[-20:]))
    print("--- stderr (last 30 lines) ---")
    print("\n".join(result.stderr.splitlines()[-30:]))


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, cpu=2, timeout=300)
def sanity_check():
    """CPU-only: verify the image imports cleanly and all checkpoint dirs are
    non-empty, WITHOUT touching the GPU -- catches broken deps / bad downloads
    before spending on the expensive A100 test."""
    import sys
    sys.path.insert(0, "/opt/TrajectoryCrafter")
    sys.path.insert(0, "/opt/TrajectoryCrafter/DepthCrafter")

    errors = []
    try:
        from models.utils import Warper  # noqa: F401
        print("OK: models.utils.Warper imports")
    except Exception as e:
        errors.append(f"models.utils import failed: {e}")

    try:
        from models.infer import DepthCrafterDemo  # noqa: F401
        print("OK: models.infer.DepthCrafterDemo imports")
    except Exception as e:
        errors.append(f"models.infer import failed: {e}")

    try:
        from models.pipeline_trajectorycrafter import TrajCrafter_Pipeline  # noqa: F401
        print("OK: models.pipeline_trajectorycrafter.TrajCrafter_Pipeline imports")
    except Exception as e:
        errors.append(f"pipeline_trajectorycrafter import failed: {e}")

    for repo_id, subdir in [
        ("TrajectoryCrafter/TrajectoryCrafter", "TrajectoryCrafter"),
        ("tencent/DepthCrafter", "DepthCrafter"),
        ("stabilityai/stable-video-diffusion-img2vid-xt", "stable-video-diffusion-img2vid-xt"),
        ("alibaba-pai/CogVideoX-Fun-V1.1-5b-InP", "CogVideoX-Fun-V1.1-5b-InP"),
        ("Salesforce/blip2-opt-2.7b", "blip2-opt-2.7b"),
    ]:
        d = pathlib.Path(WEIGHTS_DIR) / subdir
        if not d.exists() or not any(d.iterdir()):
            errors.append(f"checkpoint dir missing/empty: {d} (for {repo_id})")
        else:
            size_gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
            print(f"OK: {subdir} present, {size_gb:.1f} GB")

    if errors:
        print("\nSANITY CHECK FAILED:")
        for e in errors:
            print(" -", e)
        raise RuntimeError(f"{len(errors)} sanity check failure(s), see above")
    print("\nAll sanity checks passed. Safe to run the GPU test.")


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_volume},
    # >=28GB VRAM recommended (CogVideoX-Fun-V1.1-5b backbone) -- does not fit
    # L4 (24GB); needs A100-40GB. ~2.6x L4's hourly rate. DO NOT run without
    # explicit confirmation -- see module docstring.
    gpu="A100-40GB",
    timeout=3600,
)
def test_render(theta: float = 10.0, phi: float = 0.0, r: float = 0.0, video_length: int = 49) -> dict:
    """Milder-than-ReCamMaster test: a small pure-tilt/pan camera pose change
    (matching the same spirit as our cam_type 3 milder-rotation ReCamMaster
    test) on scene_1's real clip, using TrajectoryCrafter's 'target' camera
    mode. theta/phi are in degrees per docs/config_help.md (theta<60 tilts up,
    phi<60 pans right); kept small here since our whole investigation started
    from ReCamMaster hallucinating badly on a large (~20deg) rotation.
    r is forward/backward dolly motion (docs: "+r (r<0.6) moves camera
    forward") -- unlike pure rotation, this creates real parallax (near
    objects shift more than far ones), which stresses the depth-estimation +
    point-cloud warp in a way theta/phi alone don't.
    """
    import shutil
    import subprocess
    import sys
    import time

    sys.path.insert(0, "/opt/trace_src")

    ckpt = pathlib.Path(WEIGHTS_DIR)
    work_dir = pathlib.Path("/opt/TrajectoryCrafter")
    video_dir = work_dir / "test" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("/opt/scene_1.mp4", video_dir / "scene_1.mp4")

    out_dir = pathlib.Path("/tmp/trajcrafter_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "inference.py",
        "--video_path", str(video_dir / "scene_1.mp4"),
        "--out_dir", str(out_dir),
        "--camera", "target",
        "--mode", "gradual",
        "--mask",
        "--target_pose", str(theta), str(phi), str(r), "0", "0",
        "--video_length", str(video_length),
        "--model_name", str(ckpt / "CogVideoX-Fun-V1.1-5b-InP"),
        "--transformer_path", str(ckpt / "TrajectoryCrafter"),  # weights are at repo root, not under crosstransformer/ (verified via HF API listing)
        "--unet_path", str(ckpt / "DepthCrafter"),
        "--pre_train_path", str(ckpt / "stable-video-diffusion-img2vid-xt"),
        "--blip_path", str(ckpt / "blip2-opt-2.7b"),
    ]
    print("running TrajectoryCrafter:", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(work_dir))
    t_done = time.time()
    print(f"TrajectoryCrafter render done in {t_done - t0:.1f}s")

    rendered = list(out_dir.rglob("*.mp4"))
    print("rendered clips:", rendered)

    persisted_dir = pathlib.Path(WEIGHTS_DIR) / "outputs" / f"theta{theta}_phi{phi}_r{r}"
    persisted_dir.mkdir(parents=True, exist_ok=True)
    persisted = []
    for clip in rendered:
        dest = persisted_dir / clip.name
        shutil.copy(clip, dest)
        weights_volume.commit()
        persisted.append(str(dest))

    return {"theta": theta, "phi": phi, "r": r, "video_length": video_length,
            "render_time_s": t_done - t0, "rendered_clips": persisted}


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_volume},
    gpu="A100-40GB",
    timeout=3600,
)
def test_render_object(theta: float = 0.0, phi: float = 0.0, r: float = 0.15, x: float = 0.0, y: float = 0.0, video_length: int = 49) -> dict:
    """Object-level test: instead of running TrajectoryCrafter on the full
    scene, run it on car_microvideo.mp4 (now the full native-resolution
    frame, see #18 in DEBUG_LOG.md for why). The idea: preserve the car's
    real pose via depth-based reprojection instead of asking a generative
    model (VACE) to invent it from a mask.

    IMPORTANT, per the honest correction after the first two runs: theta=0,
    phi=0, r=0.15 alone only applies a mild forward dolly -- it does NOT
    move the car to a new screen position, and does not test whether pose
    holds up after an actual repositioning. x/y (docs: "pans camera right"
    / "pans camera upward", recommended magnitude <4, units undocumented)
    are the actual lever for that. This call is the first real test of
    that -- unlike the two before it, a genuine attempt to reposition, not
    just re-render roughly the same view.
    """
    import shutil
    import subprocess
    import sys
    import time

    sys.path.insert(0, "/opt/trace_src")

    ckpt = pathlib.Path(WEIGHTS_DIR)
    work_dir = pathlib.Path("/opt/TrajectoryCrafter")
    video_dir = work_dir / "test" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("/opt/car_microvideo.mp4", video_dir / "car_microvideo.mp4")

    out_dir = pathlib.Path("/tmp/trajcrafter_object_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "inference.py",
        "--video_path", str(video_dir / "car_microvideo.mp4"),
        "--out_dir", str(out_dir),
        "--camera", "target",
        "--mode", "gradual",
        "--mask",
        "--target_pose", str(theta), str(phi), str(r), str(x), str(y),
        "--video_length", str(video_length),
        "--model_name", str(ckpt / "CogVideoX-Fun-V1.1-5b-InP"),
        "--transformer_path", str(ckpt / "TrajectoryCrafter"),
        "--unet_path", str(ckpt / "DepthCrafter"),
        "--pre_train_path", str(ckpt / "stable-video-diffusion-img2vid-xt"),
        "--blip_path", str(ckpt / "blip2-opt-2.7b"),
    ]
    print("running TrajectoryCrafter on car micro-video:", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(work_dir))
    t_done = time.time()
    print(f"object-level render done in {t_done - t0:.1f}s")

    rendered = list(out_dir.rglob("*.mp4"))
    print("rendered clips:", rendered)

    persisted_dir = pathlib.Path(WEIGHTS_DIR) / "object_outputs" / f"theta{theta}_phi{phi}_r{r}_x{x}_y{y}"
    persisted_dir.mkdir(parents=True, exist_ok=True)
    persisted = []
    for clip in rendered:
        dest = persisted_dir / clip.name
        shutil.copy(clip, dest)
        weights_volume.commit()
        persisted.append(str(dest))

    return {"theta": theta, "phi": phi, "r": r, "video_length": video_length,
            "render_time_s": t_done - t0, "rendered_clips": persisted}


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_volume},
    gpu="A100-40GB",
    timeout=3600,
)
def render_harbor(theta: float = 0.0, phi: float = 20.0, r: float = 0.3, x: float = 0.0, y: float = 0.0, video_length: int = 49) -> dict:
    """Trajectory-edit sample on the clean MAVREC harbor clip (clip_s4_car) --
    the same static-hover clip that produced the clean removal pairs. A moderate
    pan (phi) + forward dolly (r) that creates genuine parallax, so the
    depth-reprojection is actually exercised (not just a trivial re-render).
    """
    import shutil, subprocess, sys, time
    sys.path.insert(0, "/opt/trace_src")
    ckpt = pathlib.Path(WEIGHTS_DIR)
    work_dir = pathlib.Path("/opt/TrajectoryCrafter")
    video_dir = work_dir / "test" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("/opt/harbor.mp4", video_dir / "harbor.mp4")
    out_dir = pathlib.Path("/tmp/trajcrafter_harbor_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "inference.py",
        "--video_path", str(video_dir / "harbor.mp4"),
        "--out_dir", str(out_dir),
        "--camera", "target", "--mode", "gradual", "--mask",
        "--target_pose", str(theta), str(phi), str(r), str(x), str(y),
        "--video_length", str(video_length),
        "--model_name", str(ckpt / "CogVideoX-Fun-V1.1-5b-InP"),
        "--transformer_path", str(ckpt / "TrajectoryCrafter"),
        "--unet_path", str(ckpt / "DepthCrafter"),
        "--pre_train_path", str(ckpt / "stable-video-diffusion-img2vid-xt"),
        "--blip_path", str(ckpt / "blip2-opt-2.7b"),
    ]
    print("running TrajectoryCrafter on harbor clip:", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(work_dir))
    dt = time.time() - t0
    print(f"harbor render done in {dt:.1f}s")
    rendered = list(out_dir.rglob("*.mp4"))
    print("rendered:", rendered)
    out = {}
    for clip in rendered:
        data = clip.read_bytes()
        out[clip.name] = data
    # return the largest mp4 (the final result) as bytes for local saving
    result_bytes = max(out.values(), key=len) if out else b""
    persisted_dir = pathlib.Path(WEIGHTS_DIR) / "harbor_outputs" / f"phi{phi}_r{r}"
    persisted_dir.mkdir(parents=True, exist_ok=True)
    for clip in rendered:
        shutil.copy(clip, persisted_dir / clip.name)
    weights_volume.commit()
    return {"phi": phi, "r": r, "render_time_s": dt,
            "files": [c.name for c in rendered], "result_bytes": result_bytes}


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_volume},
    # A100-40GB confirmed insufficient for ANY meaningful resolution bump above
    # inference.py's own tiny silent default -- OOM'd at both 720x1280 (needed
    # 30GB more than the 23.9GB free) and even the modest 480x848 "480p" tier
    # (needed 11.75GB more than the 8.9GB free, 30.56GB already allocated).
    # A100-80GB gives the headroom this specific attention op needs.
    gpu="A100-80GB",
    timeout=3600,
)
def render_traj(video_bytes: bytes, theta: float = 0.0, phi: float = 0.0, r: float = 0.0,
                x: float = 0.0, y: float = 0.0, video_length: int = 49,
                sample_h: int = None, sample_w: int = None) -> bytes:
    """Router-callable: take an arbitrary clip (mp4 bytes) + a target pose, run
    TrajectoryCrafter, and return the generated clip (gen.mp4) bytes. Lets the NL
    router fire a real trajectory edit mid-pipeline.

    sample_h/sample_w: DO NOT SET for --mode gradual. Explicitly passing
    --sample_size (tried 720x1280, then 480x848) reliably crashes
    TrajectoryCrafter's own gradual-mode compositing in demo.py -- it warps
    real left/right content at --sample_size but internally generates the
    fill "interval" at a hardcoded ~384px reference, then torch.cat's the two
    without resizing: "Expected size 480 but got size 384" (confirmed twice,
    same root cause both times, not fixable via a different --sample_size
    value). Leaving this unset falls back to inference.py's own working
    default -- softer output (documented limitation) but does not crash.
    """
    import shutil, subprocess, sys, time
    sys.path.insert(0, "/opt/trace_src")
    ckpt = pathlib.Path(WEIGHTS_DIR)
    work_dir = pathlib.Path("/opt/TrajectoryCrafter")
    video_dir = work_dir / "test" / "videos"; video_dir.mkdir(parents=True, exist_ok=True)
    inp = video_dir / "router_in.mp4"; inp.write_bytes(video_bytes)
    out_dir = pathlib.Path("/tmp/traj_router_out"); out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "inference.py", "--video_path", str(inp), "--out_dir", str(out_dir),
        "--camera", "target", "--mode", "gradual", "--mask",
        "--target_pose", str(theta), str(phi), str(r), str(x), str(y),
        "--video_length", str(video_length),
    ]
    if sample_h is not None and sample_w is not None:
        cmd += ["--sample_size", str(sample_h), str(sample_w)]
    cmd += [
        "--model_name", str(ckpt / "CogVideoX-Fun-V1.1-5b-InP"),
        "--transformer_path", str(ckpt / "TrajectoryCrafter"),
        "--unet_path", str(ckpt / "DepthCrafter"),
        "--pre_train_path", str(ckpt / "stable-video-diffusion-img2vid-xt"),
        "--blip_path", str(ckpt / "blip2-opt-2.7b"),
    ]
    print("router trajectory render:", " ".join(cmd)); t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(work_dir))
    print(f"done in {time.time()-t0:.1f}s")
    gens = list(out_dir.rglob("gen.mp4")) or list(out_dir.rglob("*.mp4"))
    return gens[0].read_bytes()


@app.local_entrypoint()
def main(step: str = "download", theta: float = 10.0, phi: float = 0.0, r: float = 0.0, x: float = 0.0, y: float = 0.0, video_length: int = 49):
    if step == "render_harbor":
        res = render_harbor.remote(theta=theta, phi=phi, r=r, x=x, y=y, video_length=video_length)
        rb = res.pop("result_bytes", b"")
        if rb:
            open("/tmp/harbor_traj_result.mp4", "wb").write(rb)
            print("saved /tmp/harbor_traj_result.mp4", len(rb), "bytes")
        print(res)
        return
    if step == "download":
        download_weights.remote()
    elif step == "sanity":
        sanity_check.remote()
    elif step == "debug_video":
        check_torchvision_video.remote()
    elif step == "debug_patch":
        check_inference_patch.remote()
    elif step == "render":
        result = test_render.remote(theta=theta, phi=phi, r=r, video_length=video_length)
        print(result)
    elif step == "render_object":
        result = test_render_object.remote(theta=theta, phi=phi, r=r, x=x, y=y, video_length=video_length)
        print(result)
    else:
        raise ValueError(f"unknown step: {step}")
