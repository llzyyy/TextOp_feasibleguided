import rclpy
from rclpy.node import Node
import logging
from pathlib import Path
from omegaconf import OmegaConf
from functools import partial
from pathlib import Path
from typing import Deque, Dict, Any, Optional, List, Union
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch_tensorrt

import torch._dynamo

torch._dynamo.config.suppress_errors = True

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from textop_ctrl.msg import MotionBlock

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
import time
import threading

import robotmdar
from robotmdar.dtype import seed
from robotmdar.dtype.motion import AbsolutePose, MotionDict_, get_zero_abs_pose, get_zero_feature, motion_feature_dim, motion_feature_to_dict, motion_dict_to_feature, MotionKeys
from robotmdar.dtype.abc import Diffusion, VAE, Denoiser, Dataset, SSampler
from robotmdar.train.manager import DARManager
from robotmdar.skeleton.robot import RobotSkeleton
from robotmdar.skeleton.forward_kinematics import ForwardKinematics
from robotmdar.eval.generate_dar import ClassifierFreeWrapper, generate_next_motion
from robotmdar.model.clip import load_and_freeze_clip, encode_text
from robotmdar.guidance import TaskContext, build_geoguide, make_vae_decoder


def warmup(vae_trt, cfg_denoiser, diffusion, val_data, clip_model, future_len,
           history_len, cfg):

    history_motion = val_data.normalize(get_zero_feature().unsqueeze(0).expand(
        1, history_len, -1).to(cfg.device))
    abs_pose = get_zero_abs_pose((1, ), device=cfg.device)

    def get_text_embedding(text: str, clip_model, device: str) -> torch.Tensor:
        """Encode text using CLIP model."""
        try:
            with torch.no_grad():
                import clip
                text_tokens = clip.tokenize([text]).to(device)
                text_embedding = clip_model.encode_text(text_tokens)
                # text_embedding = text_embedding / text_embedding.norm(
                #     dim=-1, keepdim=True)
            return text_embedding.float()
        except Exception as e:
            print(f"Warning: Failed to encode text '{text}': {e}")
            return torch.zeros(1, 512, device=device, dtype=torch.float32)

    text_embedding = get_text_embedding("stand", clip_model, cfg.device)
    future_motion, motion_dict, abs_pose = generate_next_motion(
        vae=vae_trt,
        denoiser=cfg_denoiser,
        diffusion=diffusion,
        val_data=val_data,
        text_embedding=text_embedding,
        history_motion=history_motion,
        abs_pose=abs_pose,
        future_len=future_len,
        # cfg=cfg,
        use_full_sample=cfg.use_full_sample,
        guidance_scale=cfg.guidance_scale,
    )
    # 因为第一次和第二次, history_motion的stride内存布局不一样, 会重新触发编译

    # print(f"First history_motion strides: {history_motion.stride()}")
    # print(
    #     f"Second history_motion strides: {future_motion[:, -history_len:, :].stride()}"
    # )
    history_motion = future_motion[:, -history_len:, :]
    future_motion, motion_dict, abs_pose = generate_next_motion(
        vae=vae_trt,
        denoiser=cfg_denoiser,
        diffusion=diffusion,
        val_data=val_data,
        text_embedding=text_embedding,
        history_motion=history_motion,
        abs_pose=abs_pose,
        future_len=future_len,
        # cfg=cfg,
        use_full_sample=cfg.use_full_sample,
        guidance_scale=cfg.guidance_scale,
    )


mujoco_to_isaaclab_reindex = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24,
    18, 25, 19, 26, 20, 27, 21, 28
]


def expand_dof_23_to_29(v: np.ndarray) -> np.ndarray:
    """
    v: shape [T, 23]
    return: shape [T, 29]
    """
    T = v.shape[0]
    out = np.zeros((T, 29), dtype=v.dtype)

    out[:, :19] = v[:, :19]
    out[:, 22:26] = v[:, 19:23]

    return out


def jetson_compatible_torch_compile(vae, denoiser, dar_cfg):
    print("DEBUG: jetson_compatible_torch_compile")
    future_len = dar_cfg.data.future_len

    example_timestep = 9
    example_y = {'uncond': F}
    example_latent = torch.randn(1, 1, 128).cuda().to(torch.float32)
    example_history = torch.randn(1, 2, 57).cuda().to(torch.float32)

    class WrappedVAEDecode(torch.nn.Module):

        def __init__(self, vae_model, nfuture):
            super().__init__()
            self.vae = vae_model
            self.nfuture = nfuture

        def forward(self, motion_latent: torch.Tensor,
                    history_motion: torch.Tensor):
            return self.vae.decode(motion_latent,
                                   history_motion,
                                   nfuture=self.nfuture)

    class WrappedTRTVAE:

        def __init__(self):
            vae_decode_trace = torch.jit.trace(
                # vae.decode,
                WrappedVAEDecode(vae, future_len),
                (example_latent, example_history))
            # vae_decode_trace = torch.jit.trace_module(
            #     vae,                     # 追踪的 module
            #     {"decode": (example_latent, example_history, future_len)}  # 需要追踪的方法和示例输入
            # )

            self.fn = torch_tensorrt.compile(
                # WrappedVAEDecode(vae, future_len),
                vae_decode_trace,
                inputs=[
                    torch_tensorrt.Input(example_latent.shape),
                    torch_tensorrt.Input(example_history.shape),
                ],
                enabled_precisions={torch.float32},
                truncate_long_and_double=True,
                workspace_size=1 << 22)

        def decode(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    # vae_trt = WrappedTRTVAE()
    vae_trt = None

    # ret =  vae_trt.decode(example_latent, example_history)
    # print(f"DEBUG: vae_trt.decode output shape: {ret.shape} dtype: {ret.dtype}")

    denoiser_trt = None
    raise NotImplementedError("denoiser_trt not implemented yet")
    return vae_trt, denoiser_trt
    ...


class MotionDAR(Node):
    dt = 0.02
    timer = None
    _init_done = False
    _start_infer = False
    _counter = 0
    _gen_counter = -1
    _toggle_time = -1.0
    _block_index = 0
    _current_block_size = 0

    _text_prompt: str = "stand"
    _text_embedding: torch.Tensor = None
    _ref_motion_dict: Dict[str, Any]
    history_motion: torch.Tensor = None
    history_abs_pose: AbsolutePose = None

    # Thread control
    _keyboard_thread: threading.Thread = None
    _shutdown_event: threading.Event = None

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__("motion_dar")
        # self.get_logger().set_level(logging.DEBUG)

        # Store configuration
        self.config = cfg

        # Extract configuration parameters
        self.ckpt = cfg.dar.ckpt
        self.use_full_sample = cfg.dar.use_full_sample
        self.guidance_scale = cfg.dar.guidance_scale
        self._device = cfg.dar.device
        self._robot_world_position = None
        self._robot_world_yaw = None
        self._task_start_position = None
        self._task_start_yaw = None

        # Motion block parameters
        self.block_size = cfg.motion_block.block_size
        self.total_steps = cfg.motion_block.total_steps
        self.num_joints = cfg.motion_block.num_joints

        self.dt = cfg.control_dt

        self.load_dar()

        # Create publisher for MotionBlock
        self.motion_publisher = self.create_publisher(MotionBlock,
                                                      cfg.topics.motion_output,
                                                      10)

        self.odom_subscriber = self.create_subscription(
            Odometry,
            cfg.topics.odometry,
            self._odometry_callback,
            10,
        )

        self.reset_subscriber = self.create_subscription(
            Time,
            cfg.topics.toggle,
            self._toggle_callback,
            10,
        )
        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info("MotionDAR Created with Hydra configuration")
        self.get_logger().info(
            f"  - block_size: {self.block_size}, total_steps: {self.total_steps}, num_joints: {self.num_joints}"
        )

        # Initialize and start keyboard input thread
        self._shutdown_event = threading.Event()
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_input_loop, daemon=True)
        self._keyboard_thread.start()
        self.get_logger().info(
            "Keyboard input thread started. Type text prompts and press Enter."
        )

        self.get_logger().info(
            "Waiting for /dar/toggle to start generating...")

    def _odometry_callback(self, msg: Odometry):
        """Cache simulation world pose as non-differentiable TaskContext."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self._robot_world_position = torch.tensor(
            [[position.x, position.y, position.z]],
            device=self._device,
            dtype=torch.float32,
        )
        yaw = np.arctan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        self._robot_world_yaw = torch.tensor([yaw], device=self._device, dtype=torch.float32)

    def _create_msg_array(self, data, dimensions_info):
        """Helper function to create Float32MultiArray with proper dimension info"""
        msg = Float32MultiArray()

        # Set data
        if isinstance(data, np.ndarray):
            msg.data = data.flatten().astype(np.float32).tolist()
        else:
            msg.data = [float(x) for x in data]

        # Set dimensions
        msg.layout.dim = []
        for label, size, stride in dimensions_info:
            dim = MultiArrayDimension()
            dim.label = label
            dim.size = size
            dim.stride = stride
            msg.layout.dim.append(dim)

        msg.layout.data_offset = 0
        return msg

    def _convert_motion_to_msg_instance(self):
        """Convert motion dictionary to complete MotionBlock message instance and cache it"""
        if not hasattr(self,
                       '_ref_motion_dict') or self._ref_motion_dict is None:
            self.get_logger().warning(
                "Motion buffer not initialized, cannot convert to message instance"
            )
            return

        # Extract data from motion dict (remove batch dimension [1, T, ...] -> [T, ...])
        joint_pos_block_mjc_ord = expand_dof_23_to_29(
            self._ref_motion_dict['dof_pos']
            [0].cpu().numpy())  # [T, 23] -> [T, 29]
        joint_pos_block = joint_pos_block_mjc_ord[:,
                                                  mujoco_to_isaaclab_reindex]
        joint_vel_block_mjc_ord = expand_dof_23_to_29(
            self._ref_motion_dict['dof_vel']
            [0].cpu().numpy())  # [T, 23] -> [T, 29]
        joint_vel_block = joint_vel_block_mjc_ord[:,
                                                  mujoco_to_isaaclab_reindex]
        anchor_pos_block = self._ref_motion_dict['root_trans_offset'][0].cpu(
        ).numpy()  # [T, 3]
        anchor_ori_block = self._ref_motion_dict['root_rot'][0].cpu().numpy(
        )[:, [3, 0, 1, 2]]  # [T, 4], xyzw -> wxyz

        T = joint_pos_block.shape[0]  # Number of time steps in buffer
        Nq = joint_pos_block.shape[1]  # Number of joints

        # Create complete MotionBlock message instance
        self._cached_motion_msg = MotionBlock()
        self._cached_motion_msg.index = self._block_index

        # Fill message fields
        self._cached_motion_msg.joint_positions = self._create_msg_array(
            joint_pos_block, [("time_steps", T, T * Nq), ("joints", Nq, Nq)])
        self._cached_motion_msg.joint_velocities = self._create_msg_array(
            joint_vel_block, [("time_steps", T, T * Nq), ("joints", Nq, Nq)])
        self._cached_motion_msg.anchor_body_ori = self._create_msg_array(
            anchor_ori_block, [("time_steps", T, T * 4), ("quaternion", 4, 4)])
        self._cached_motion_msg.anchor_body_pos = self._create_msg_array(
            anchor_pos_block, [("time_steps", T, T * 3), ("position", 3, 3)])

        self._cached_buffer_size = T

        self.get_logger().debug(
            f"Converted motion to message instance: {T} steps x {Nq} joints")

    def _publish_motion_block(self):
        """Publish all time steps from the current buffer using cached message instance"""

        t0 = time.time()

        # Check if cached message instance is available
        if not hasattr(self, '_cached_motion_msg'):
            self.get_logger().warning(
                "Cached message instance not available, skipping publish")
            return

        # Update only the index and timestamp in the cached message
        self._cached_motion_msg.timestamp = self.get_clock().now().to_msg()

        # Publish the cached message directly
        self.motion_publisher.publish(self._cached_motion_msg)

        t1 = time.time()
        T = self._cached_buffer_size
        # Log based on configuration
        self.get_logger().debug(
            f"Published MotionBlock #{self._cached_motion_msg.index} with {T} steps "
            f"in {(t1 - t0) * 1000:.2f} ms")

    def load_dar(self):
        ckpt = Path(self.ckpt)
        dar_cfg_path = ckpt.parent / '.hydra' / 'config.yaml'
        # dar_cfg_path = ckpt.parent / 'cfg.yaml'
        self.dar_cfg = OmegaConf.load(dar_cfg_path)

        cfg = self.dar_cfg
        cfg.device = str(self._device)
        cfg.ckpt.dar = str(ckpt)
        cfg.train.manager.device = str(self._device)
        cfg.train.manager.platform._target_ = 'robotmdar.train.train_platforms.NoPlatform'
        cfg.data.datadir = self.config.dar.datadir
        cfg.skeleton.asset.assetRoot = self.config.dar.skeleton_assetRoot
        cfg.data.val.split = 'none'
        cfg.data.val.batch_size = 1
        cfg.use_full_sample = self.use_full_sample
        cfg.guidance_scale = self.guidance_scale

        seed.set(cfg.seed)
        self.clip_model = load_and_freeze_clip("ViT-B/32", device=self._device)
        val_data: Dataset = instantiate(cfg.data.val)
        vae: VAE = instantiate(cfg.vae)
        denoiser: Denoiser = instantiate(cfg.denoiser)

        schedule_sampler: SSampler = instantiate(
            cfg.diffusion.schedule_sampler)
        diffusion: Diffusion = schedule_sampler.diffusion

        vae.eval()
        denoiser.eval()

        # Load checkpoints
        manager: DARManager = instantiate(cfg.train.manager)
        manager.hold_model(vae, denoiser, None, val_data)

        # vae_trt = vae
        # denoiser_trt = denoiser
        try:
            vae_trt = torch.compile(vae, backend='tensorrt')
            denoiser_trt = torch.compile(denoiser, backend='tensorrt')
        except KeyError as e:
            error_key = e.args[0]
            if error_key == 'torch_dynamo_backends':
                # Now we are in jetson env, which does not support torch.compile
                vae_trt, denoiser_trt = jetson_compatible_torch_compile(
                    vae, denoiser, cfg)
            else:
                raise e

        # vae_trt = torch.compile(vae, backend='inductor')
        # denoiser_trt = torch.compile(denoiser, backend='inductor')

        cfg_denoiser = ClassifierFreeWrapper(denoiser_trt)

        self.dataset = val_data
        self.dataiter = iter(val_data)
        self._motion_dt = 1 / val_data.fps
        self.future_len = cfg.data.future_len
        self.history_len = cfg.data.history_len
        self.gen_len = self.history_len + self.future_len
        self.geoguide = None
        if self.config.get("geoguide") is not None and bool(self.config.geoguide.get("enabled", False)):
            project_root = Path(__file__).resolve().parents[4]
            self.geoguide = build_geoguide(
                self.config.geoguide,
                make_vae_decoder(vae, self.future_len),
                project_root=project_root,
                feature_mean=val_data.mean,
                feature_std=val_data.std,
            )
            self.get_logger().info("GeoGuide enabled; PyTorch VAE retained for task gradients/JVP")
        self.dar_gen_fn = partial(generate_next_motion,
                                  vae=vae_trt,
                                  denoiser=cfg_denoiser,
                                  diffusion=diffusion,
                                  val_data=val_data,
                                  future_len=self.future_len,
                                  use_full_sample=self.use_full_sample,
                                  guidance_scale=self.guidance_scale,
                                  geoguide=self.geoguide,
                                  ret_fk=True,
                                  ret_fk_full=False)

        self.warmup(vae_trt, cfg_denoiser, diffusion, val_data,
                    self.clip_model, self.future_len, self.history_len, cfg)

        self._reset_motion_buffer()
        self._update_text_embedding()

        self.get_logger().info(f"Finish loading DAR from {ckpt}")
        self._init_done = True

    def warmup(self, vae_trt, cfg_denoiser, diffusion, val_data, clip_model,
               future_len, history_len, cfg):
        t0 = time.time()
        self.get_logger().info("Warming up...")
        for i in range(3):
            warmup(vae_trt, cfg_denoiser, diffusion, val_data, clip_model,
                   future_len, history_len, cfg)
            self._reset_motion_buffer()
            self._update_text_embedding()
            self._gen_motion()
        t1 = time.time()
        self.get_logger().info(f"Warming up done in {(t1 - t0) * 1000:.2f} ms")

    def _toggle_callback(self, msg: Time):
        msg_time = msg.sec + msg.nanosec / 1e9
        self.get_logger().info(f"Toggle callback: {msg_time}")
        self._toggle_time = msg_time
        self._counter = 0

        if not self._start_infer:
            self.get_logger().info("Start inference")
            self._reset_motion_buffer()
            self._task_start_position = (
                None if self._robot_world_position is None else self._robot_world_position.detach().clone()
            )
            self._task_start_yaw = None if self._robot_world_yaw is None else self._robot_world_yaw.detach().clone()
        else:
            self.get_logger().info("Stop inference")

        self._start_infer = not self._start_infer

    def loop(self):
        if not self._start_infer:
            return

        self.get_logger().debug(f"Loop: {self._counter}")

        # Publish motion block data
        if self._gen_counter - self._current_block_size - self._counter <= self.future_len:
            self._publish_motion_block()
            self._gen_motion()

        # Calculate counter based on absolute time since toggle started
        if self._toggle_time >= 0:
            time0 = self.get_clock().now()
            # current_time = time0.seconds + time0.nanoseconds / 1e9   # deprecated usage
            current_time = time0.seconds_nanoseconds()[0] + time0.seconds_nanoseconds()[1] / 1e9
            absolute_elapsed = current_time - self._toggle_time
            
            self._counter = int(np.floor(absolute_elapsed / self.dt))

    def _gen_motion(self):
        t0 = time.time()

        task_context = None
        if self.geoguide is not None:
            reference_position = self.history_abs_pose['root_trans_offset']
            reference_yaw_quaternion = self.history_abs_pose['root_rot']
            reference_context = TaskContext.from_reference_pose(
                reference_position,
                reference_yaw_quaternion,
                motion_fps=float(self.dataset.fps),
            )
            robot_position = self._robot_world_position
            robot_yaw = self._robot_world_yaw
            if robot_position is None or robot_yaw is None:
                robot_position = reference_context.robot_world_position
                robot_yaw = reference_context.robot_world_yaw
            task_context = TaskContext(
                robot_world_position=robot_position,
                robot_world_yaw=robot_yaw,
                generator_position=reference_context.generator_position,
                generator_yaw=reference_context.generator_yaw,
                robot_start_position=(self._task_start_position if self._task_start_position is not None else robot_position),
                robot_start_yaw=(self._task_start_yaw if self._task_start_yaw is not None else robot_yaw),
                motion_fps=float(self.dataset.fps),
            )

        future_motion, gen_motion_dict, abs_pose = self.dar_gen_fn(
            text_embedding=self._text_embedding,
            history_motion=self.history_motion,
            abs_pose=self.history_abs_pose,
            task_context=task_context)

        t02 = time.time()
        self.get_logger().debug(
            f"dar_gen_fn: generate motion in {(t02 - t0) * 1000:.2f} ms")

        self.history_motion = future_motion[:, -self.history_len:, :]
        self.history_abs_pose = abs_pose
        for k in self._ref_motion_dict:
            if isinstance(self._ref_motion_dict[k], torch.Tensor):
                self._ref_motion_dict[k] = gen_motion_dict[k][:, -self.
                                                              future_len:]
                # self.get_logger().debug(f"{k}: {self._ref_motion_dict[k].shape}")
            # (Pdb) gen_motion_dict['root_trans_offset'].shape
            # torch.Size([1, 10, 3])

        self._current_block_size = self.future_len
        self._block_index = self._gen_counter
        self._gen_counter += self._current_block_size
        t1 = time.time()
        self.get_logger().debug(
            f"Current | self._counter: {self._counter} | self._gen_counter: {self._gen_counter} | self._block_index: {self._block_index}"
        )

        self.get_logger().debug(
            f"Generate motion in {(t1 - t0) * 1000:.2f} ms")
        # Convert motion to message instance immediately after generation
        self._convert_motion_to_msg_instance()
        t2 = time.time()
        self.get_logger().debug(
            f"Convert motion to message instance in {(t2 - t1) * 1000:.2f} ms")

    def _reset_motion_buffer(self):
        _reset_block_size = self.gen_len + self.future_len  # 18
        self._block_index = 0
        self._gen_counter = _reset_block_size
        self._current_block_size = _reset_block_size

        motion_feat = get_zero_feature().to(self._device).reshape(
            1, 1, -1).repeat(1, _reset_block_size, 1)  # [1, 18, 57]
        self.history_motion = self.dataset.normalize(
            motion_feat)[:, -self.history_len:, :]  # [1, 18, 57] -> [1, 2, 57]
        self.history_abs_pose = get_zero_abs_pose((1, ), device=self._device)
        self._ref_motion_dict = self.dataset.reconstruct_motion(
            motion_feat.to(self._device),
            need_denormalize=False,
            ret_fk=True,
            ret_fk_full=False)  # type:ignore

        # (Pdb) motion_feat.shape
        # torch.Size([1, 18, 57])
        # (Pdb) self._ref_motion_dict['dof'].shape
        # torch.Size([1, 18, 23])

        # Convert motion to message instance immediately after reset
        self._convert_motion_to_msg_instance()

    def _update_text_embedding(self):
        with torch.no_grad():
            text_embedding = encode_text(self.clip_model, [self._text_prompt])
            # text_embedding = text_embedding / text_embedding.norm(
            #     dim=-1, keepdim=True)  # FIXME: why use norm?
            self._text_embedding = text_embedding.float()
            # self.get_logger().debug(f"Text prompt: {self._text_prompt}")
            self.get_logger().debug(
                f"Text embedding: {self._text_embedding.shape}")

    def _keyboard_input_loop(self):
        """Background thread loop to handle keyboard input for text prompts"""

        while not self._shutdown_event.is_set():
            try:
                # Use input() to get text from keyboard
                user_input = input(
                    "Enter text prompt (or 'q' to exit): \n").strip()

                if user_input.lower() in ['exit', 'q']:
                    self._shutdown_event.set()
                    # Shutdown the entire ROS2 node
                    rclpy.shutdown()
                    break

                if user_input:  # Only update if non-empty input
                    self._text_prompt = user_input
                    self.get_logger().info(
                        f"Text prompt updated: '{self._text_prompt}'")

                    self._update_text_embedding()

            except EOFError:
                # Handle Ctrl+D or EOF
                self.get_logger().info(
                    "EOF received, stopping keyboard input thread")
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                self.get_logger().info("KeyboardInterrupt in input thread")
                break
            except Exception as e:
                self.get_logger().error(f"Error in keyboard input thread: {e}")

        self.get_logger().info("Keyboard input thread exiting")

    def destroy_node(self):
        """Override destroy_node to clean up threads"""
        self.get_logger().info("Shutting down MotionDAR node...")

        # Signal thread to shutdown
        if self._shutdown_event:
            self._shutdown_event.set()

        # Wait for thread to finish (with timeout)
        if self._keyboard_thread and self._keyboard_thread.is_alive():
            self._keyboard_thread.join(timeout=1.0)
            if self._keyboard_thread.is_alive():
                print(
                    "Warning: Keyboard input thread did not shutdown gracefully"
                )

        super().destroy_node()




@hydra.main(version_base=None,
            config_path=f"../config",
            config_name="rmdar_config")
def main(cfg: DictConfig) -> None:
    """Main function to run MotionDAR node with Hydra configuration"""
    print("Starting MotionDAR with configuration:")
    print(f"\n{OmegaConf.to_yaml(cfg)}")

    rclpy.init()
    node = None
    try:
        node = MotionDAR(cfg)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down MotionDAR...")
    finally:
        if node:
            node.destroy_node()
        # Only shutdown if it wasn't already shutdown by keyboard thread
        if not (node and node._shutdown_event.is_set()):
            rclpy.shutdown()


if __name__ == "__main__":
    main()
