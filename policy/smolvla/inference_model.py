from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import numpy as np

# LeRobot imports for robust loading and normalization
try:
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("WARNING: LeRobot not installed. Defaulting to raw transformers (normalization may be incorrect).")

class SMOLVLA:
    def __init__(self, model_path, dataset_repo_id=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.use_lerobot = LEROBOT_AVAILABLE and (dataset_repo_id is not None)
        self.instruction = "pick up object"
        self.observation = None

        if self.use_lerobot:
            print(f"Loading SmolVLA via LeRobot from {model_path}...")
            # Load Config
            self.cfg = PreTrainedConfig.from_pretrained(model_path)
            
            # Load Metadata (required for normalization stats)
            print(f"Loading dataset metadata from {dataset_repo_id}...")
            # Ideally avoid full download, LeRobotDataset usually handles this gracefully or checks cache
            self.dataset = LeRobotDataset(repo_id=dataset_repo_id)
            ds_meta = self.dataset.meta

            # Make Policy
            self.policy = make_policy(cfg=self.cfg, ds_meta=ds_meta)
            self.policy = self.policy.from_pretrained(model_path)
            self.policy.to(self.device)
            self.policy.eval()
            
            # Processors
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=self.cfg,
                pretrained_path=model_path,
            )
        else:
            print(f"Loading SmolVLA via Transformers from {model_path}...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.vla = AutoModelForVision2Seq.from_pretrained(
                model_path, 
                attn_implementation="flash_attention_2", 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            ).to(self.device)

    def random_set_language(self, instruction=None):
        if instruction:
            self.instruction = instruction
        print(f"Instruction set to: {self.instruction}")

    def update_observation_window(self, observation_input, state=None):
        """
        observation_input: 
            If using LeRobot: dict matching policy keys (e.g. {"observation.images.cam_head": ...})
            If using Transformers: (img_head, img_wrist) tuple/list + state
        """
        if self.use_lerobot:
            # Expecting a dict directly or constructing it?
            # Ideally the caller passes the correct dict.
            # But to maintain compat with the example script's calling convention:
            if isinstance(observation_input, (list, tuple)) and state is not None:
                # Convert list of images to what LeRobot typically expects
                # WARNING: Key names must match what the policy was trained with!
                # Assuming "img_head" -> "observation.images.cam_head"
                # Assuming "img_wrist" -> "observation.images.cam_wrist"
                # This depends on the specific dataset linkage.
                self.observation = {
                    "observation.images.cam_head": observation_input[0],
                    "observation.images.cam_wrist": observation_input[1],
                    "observation.state": state
                }
            elif isinstance(observation_input, dict):
                 self.observation = observation_input
        else:
            # Legacy/Transformers mode
            if isinstance(observation_input, (list, tuple)):
                self.observation = {
                    "image_head": observation_input[0],
                    "image_wrist": observation_input[1],
                    "state": state
                }

    def get_action(self):
        if self.observation is None:
            raise ValueError("Observation not set. Call update_observation_window first.")
            
        if self.use_lerobot:
            # Prepare Batch (Add batch dim)
            batch = {}
            for k, v in self.observation.items():
                if isinstance(v, np.ndarray):
                    # Check dim. LeRobot expects [B, C, H, W] for images usually? 
                    # preprocessor handles formatting.
                    val = torch.from_numpy(v).to(self.device).float() # Ensure float for input
                    
                    if val.ndim > 0:
                        val = val.unsqueeze(0)
                        
                    # Handle Image permutation if needed: HWC (Piper) -> CHW (LeRobot Dataset)
                    if "image" in k and val.shape[-1] == 3: # HWC detected
                         val = val.permute(0, 3, 1, 2) # B, H, W, C -> B, C, H, W
                    
                    batch[k] = val
                else:
                    batch[k] = v
                    
            if "task" not in batch:
                batch["task"] = [self.instruction]

            with torch.no_grad():
                batch = self.preprocessor(batch)
                action = self.policy.select_action(batch)
                action = self.postprocessor(action)
            
            return action.squeeze(0).cpu().numpy()

        else:
            # Fallback (Original Logic)
            image_head = Image.fromarray(self.observation["image_head"])
            image_wrist = Image.fromarray(self.observation["image_wrist"])
            prompt = self.instruction
            
            combined_image = Image.new('RGB', (image_head.width + image_wrist.width, image_head.height))
            combined_image.paste(image_head, (0, 0))
            combined_image.paste(image_wrist, (image_head.width, 0))

            inputs = self.processor(text=prompt, images=combined_image, return_tensors="pt").to(self.device, torch.bfloat16)
            
            # Note: "bridge_orig" unnorm key is dangerous if stats don't match
            action = self.vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            return action

    def reset_obsrvationwindows(self):
        self.observation = None
