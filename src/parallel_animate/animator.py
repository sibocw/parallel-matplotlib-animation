import matplotlib

matplotlib.use("Agg")

import tempfile
import logging
import time
import multiprocessing as mp
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import av
from PIL import Image
from tqdm import tqdm, trange


class Animator(ABC):
    """
    Base class for creating matplotlib animations with efficient parallel rendering.
    """

    def __init__(self):
        """Initialize the animator."""
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def setup(self):
        """
        Set up the figure, axes, and any artists. Store whatever you need as instance
        attributes (self.ax, self.line, etc.); you will need them in update().

        This method MUST not accept any argument other than self (if you need to pass
        data to setup, store it as instance attributes in __init__).

        This method MUST return the plt.Figure object, and that alone.

        This method is called once before the animation loop (once per worker if
        parallelized), unless reuse_figure_object is False (basically never).
        """
        pass

    @abstractmethod
    def update(self, frame_idx: int, params: Any):
        """
        Update the plot for the given frame.

        This is called for each frame in the animation. Modify any figure, axes, and
        artists that you created in setup() and stored as instance attributes.

        Args:
            frame_idx (int): Current frame index (0 to num_frames-1)
            params (Any): Parameters for this frame from param_by_frame[frame_idx]. Can
                be any object you want as long as it's pickleable (required to
                distribute jobs to distributed workers).
        """
        pass

    def make_video(
        self,
        output_file: Path | str,
        param_by_frame: list[Any],
        fps: int,
        num_workers: int = -1,
        disable_progress_bar: bool | None = None,
        plotting_log_interval: int | None = None,
        saving_log_interval: int | None = None,
        savefig_params: dict[str, Any] = {},
        video_codec: str = "libx264",
        video_params: dict[str, Any] = {"pix_fmt": "yuv420p"},
        reuse_figure_object: bool = True,
    ) -> None:
        """
        Render the animation to a video file.

        Args:
            output_file (Path | str): Path to output video file
            param_by_frame (list[Any]): List of parameters, one per frame
            fps (int): Frames per second for output video
            num_workers (int): Number of parallel workers. 1 for serial processing
                (in the main thread), -1 for all CPU cores, -2 for all but one CPU core,
                etc.
            disable_progress_bar (bool | None): Same behavior as tqdm: if True, disable
                progress bar. If None, auto-detect based on whether output is a TTY.
            plotting_log_interval (int | None): Log progress every N frames in each
                worker (None = no interval logging).
            saving_log_interval (int | None): Log progress every N frames when merging
                frames into video (None = no interval logging).
            savefig_params (dict[str, Any]): Additional keyword arguments to
                pass to plt.Figure.savefig() when saving frames (default: {}).
            video_codec (str): Codec to use for video encoding (default: "libx264").
            video_params (dict[str, Any]): Additional parameters to set on the video
                stream (default: {"pix_fmt": "yuv420p"}).
            reuse_figure_object (bool): If False, the figure will be re-created for each
                frame (i.e. setup() called every frame). There is basically no reason to
                set this to False. Use only for testing and benchmarking.
        """
        # Try to convert param_by_frame to list
        try:
            params_list = list(param_by_frame)
        except Exception as e:
            self.logger.critical(
                "param_by_frame must be convertible to a list. Ensure it is a list-like object."
            )
            raise e

        num_frames = len(params_list)

        # Determine number of workers
        if num_workers == -1:
            num_workers = mp.cpu_count()
        elif num_workers < -1:
            num_workers = max(1, mp.cpu_count() + num_workers + 1)

        self.logger.info(f"Rendering {num_frames} frames at {fps} fps")
        with tempfile.TemporaryDirectory(prefix="animator_frames_") as frames_dir:
            self.logger.info(f"Using temporary directory: {frames_dir}")

            if num_workers == 1:
                self.logger.info("Running in serial mode")
                self._render_serial(
                    params_list,
                    frames_dir,
                    disable_progress_bar,
                    plotting_log_interval,
                    savefig_params,
                    reuse_figure_object,
                )
            else:
                self.logger.info(f"Running in parallel mode with {num_workers} workers")
                self._render_parallel(
                    params_list,
                    frames_dir,
                    num_workers,
                    disable_progress_bar,
                    plotting_log_interval,
                    savefig_params,
                    reuse_figure_object,
                )

            self.logger.info("Creating video with PyAV")
            _merge_frames_into_video(
                frames_dir,
                output_file,
                fps,
                video_codec,
                video_params,
                disable_progress_bar,
                self.logger,
                log_interval=saving_log_interval,
            )

        self.logger.info(f"Animation complete: {output_file}")

    def _setup_and_check(self) -> plt.Figure:
        """Call setup() and validate its return type."""
        fig = self.setup()
        if not isinstance(fig, plt.Figure):
            raise TypeError(
                f"`.setup()` must return a matplotlib Figure object, got {type(fig).__name__}"
            )
        return fig

    def _render_serial(
        self,
        params_list: list[Any],
        frames_dir: Path | str,
        disable_progress_bar: bool | None,
        log_interval: int | None,
        savefig_params: dict[str, Any],
        reuse_figure_object: bool,
    ) -> None:
        """Render frames serially."""
        self.logger.info("Serial rendering")

        if reuse_figure_object:
            # Setup once and get figure
            fig = self._setup_and_check()

        # Render all frames with progress bar
        num_frames = len(params_list)
        for frame_idx in trange(
            num_frames, desc="Rendering", disable=disable_progress_bar
        ):
            if not reuse_figure_object:
                fig = self._setup_and_check()

            params = params_list[frame_idx]
            self.update(frame_idx, params)
            fig.canvas.draw()

            frame_path = Path(frames_dir) / f"frame_{frame_idx:09d}.png"
            fig.savefig(frame_path, **savefig_params)
            if not reuse_figure_object:
                plt.close(fig)

            # Optional interval logging
            if log_interval and (frame_idx + 1) % log_interval == 0:
                self.logger.info(f"Frame {frame_idx + 1}/{num_frames}")

        plt.close(fig)

    def _render_parallel(
        self,
        params_list: list[Any],
        frames_dir: Path | str,
        num_workers: int,
        disable_progress_bar: bool | None,
        log_interval: int | None,
        savefig_params: dict[str, Any],
        reuse_figure_object: bool,
    ) -> None:
        """Render frames in parallel using dynamic work distribution."""
        num_frames = len(params_list)

        self.logger.info(f"Using dynamic work distribution with {num_workers} workers")

        # Create queues for task distribution and atomic counter for progress
        task_queue = mp.Queue()
        num_frames_completed = mp.Value("i", 0)  # atomic integer counter

        # Start worker processes
        workers = []
        for worker_id in range(num_workers):
            p = mp.Process(
                target=_worker_process,
                args=(
                    self,
                    worker_id,
                    task_queue,
                    num_frames_completed,
                    frames_dir,
                    log_interval,
                    savefig_params,
                    reuse_figure_object,
                ),
            )
            p.start()
            workers.append(p)

        # Populate task queue with individual frames (batch_size = 1)
        for frame_idx in range(num_frames):
            task_queue.put((frame_idx, params_list[frame_idx]))

        # Send sentinel values to signal workers to exit
        for _ in range(num_workers):
            task_queue.put(None)

        # Monitor progress using atomic counter
        pbar = tqdm(total=num_frames, desc="Rendering", disable=disable_progress_bar)
        while num_frames_completed.value < num_frames:
            current_progress = num_frames_completed.value
            if current_progress > pbar.n:
                pbar.update(current_progress - pbar.n)
            time.sleep(0.1)
        pbar.update(num_frames_completed.value - pbar.n)
        pbar.close()

        # Wait for all workers to finish
        for p in workers:
            p.join()

        self.logger.info("All workers completed")


def _worker_process(
    animator: Animator,
    worker_id: int,
    task_queue: mp.Queue,
    progress_counter,
    frames_dir: Path | str,
    log_interval: int | None,
    savefig_params: dict[str, Any],
    reuse_figure_object: bool,
) -> None:
    """
    Worker process that renders frames.

    Each worker:
    1. Calls setup() once to initialize the figure (unless reuse_figure_object is False)
    2. Repeatedly pulls individual frames from the task queue
    3. Renders each frame
    4. Atomically increments the progress counter
    """
    # Setup once per worker
    if reuse_figure_object:
        fig = animator._setup_and_check()

    # Process frames until we get a sentinel value (None)
    frames_processed = 0
    while True:
        task = task_queue.get()
        if task is None:
            break  # sentinel value - exit
        frame_idx, params = task

        # Render the frame
        if not reuse_figure_object:
            fig = animator._setup_and_check()
        animator.update(frame_idx, params)
        fig.canvas.draw()
        frame_path = Path(frames_dir) / f"frame_{frame_idx:09d}.png"
        fig.savefig(frame_path, **savefig_params)
        frames_processed += 1
        if not reuse_figure_object:
            plt.close(fig)

        # Logging
        if log_interval and frames_processed % log_interval == 0:
            animator.logger.info(
                f"Worker {worker_id}: processed {frames_processed} frames"
            )

        # Atomically increment progress counter
        with progress_counter.get_lock():
            progress_counter.value += 1

    plt.close(fig)
    animator.logger.info(f"Worker {worker_id}: completed {frames_processed} frames")


def _merge_frames_into_video(
    frames_dir: Path | str,
    output_file: Path | str,
    fps: int,
    video_codec: str,
    video_params: dict[str, Any],
    disable_progress_bar: bool | None,
    logger: logging.Logger,
    log_interval: int | None,
) -> None:
    """Use PyAV to merge frames into video."""
    # Gather frame files in sorted order
    frame_files = sorted(Path(frames_dir).glob("frame_*.png"))

    if not frame_files:
        raise RuntimeError("No frames found in temporary directory")

    # Open first image to determine size and ensure even dimensions
    with Image.open(frame_files[0]) as first_img:
        width, height = first_img.size
    # Make width/height even (required by many codecs)
    width = (width // 2) * 2
    height = (height // 2) * 2

    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(output_file), mode="w")
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = width
        stream.height = height
        for key, value in video_params.items():
            setattr(stream, key, value)

        # Encode each frame
        for i, frame_path in tqdm(
            enumerate(frame_files),
            total=len(frame_files),
            desc="Merging frames",
            disable=disable_progress_bar,
        ):
            img = Image.open(frame_path).convert("RGBA")
            # Ensure image has the right size
            if img.size != (width, height):
                img = img.resize((width, height))

            video_frame = av.VideoFrame.from_image(img)
            for packet in stream.encode(video_frame):
                container.mux(packet)

            # Optional interval logging
            if log_interval and (i + 1) % log_interval == 0:
                logger.info(f"Frame written {i + 1}/{len(frame_files)} to video")

        # Flush encoder
        for packet in stream.encode(None):
            container.mux(packet)

        container.close()

        size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        logger.info(f"Video created: {output_file} ({size_mb:.2f} MB)")

    except Exception as e:
        logger.critical(f"PyAV failed to create video: {e}")
        raise e
