import os
from trl import GRPOConfig, ModelConfig, TrlParser, get_peft_config
from projects.open_r1.constants import reward_funcs_registry
from projects.open_r1.utils import save_args_to_txt

from projects.open_r1.trainer.trainer_videorl_cot import Qwen3VLGRPOTrainer
from projects.open_r1.arguments import GRPOScriptArguments
from projects.open_r1.dataset import ReferVideoSegDataset, RobustMixedDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(script_args, training_args, model_args):
    if any('+' in reward for reward in script_args.reward_funcs):
        script_args.reward_funcs = script_args.reward_funcs[0].split('+')
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.kl_approximator == 'fullkimi':
        script_args.use_kl = True
        training_args.sync_ref_model = True
        training_args.ref_model_mixup_alpha = 1.0
        training_args.ref_model_sync_steps = 1

    save_args_to_txt(script_args, os.path.join(training_args.output_dir, 'config', 'script_args.txt'))
    save_args_to_txt(training_args, os.path.join(training_args.output_dir, 'config', 'training_args.txt'))
    save_args_to_txt(model_args, os.path.join(training_args.output_dir, 'config', 'model_args.txt'))

    datasets = []
    if 'mevis' in script_args.dataset_name:
        dataset_mevis = ReferVideoSegDataset(
            script_args,
            base_image_dir="data/video_datas/mevis/train/JPEGImages",
            mask_file="data/video_datas/mevis/train/mask_dict.json",
            expression_file="data/VideoSEG-O3-RL/rl_mevis_data.json"
        )
        datasets.append(dataset_mevis)
    if "revos" in script_args.dataset_name:
        dataset_revos = ReferVideoSegDataset(
            script_args,
            base_image_dir="data/video_datas/revos",
            mask_file="data/video_datas/revos/mask_dict.json",
            expression_file="data/VideoSEG-O3-RL/rl_revos_data.json"
        )
        datasets.append(dataset_revos)
    if "longrvos" in script_args.dataset_name:
        dataset_revos = ReferVideoSegDataset(
            script_args,
            base_image_dir="data/video_datas/Long-RVOS/train/JPEGImages",
            mask_file="data/video_datas/Long-RVOS/train/mask_dict.json",
            expression_file="data/VideoSEG-O3-RL/rl_longrvos_data.json"
        )
        datasets.append(dataset_revos)

    dataset = RobustMixedDataset(datasets)

    trainer_cls = Qwen3VLGRPOTrainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        script_args=script_args,
    )

    trainer.train()

    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    print('training_args:\n', training_args)
    print('script_args:\n', script_args)
    print('model_args:\n', model_args)
    main(script_args, training_args, model_args)
