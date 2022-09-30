from transformers.convert_pytorch_checkpoint_to_tf2 import convert_pt_checkpoint_to_tf


state_dict_path = '/home/eco/zyh/github/TPlinker-joint-extraction/default_log_dir/aSF8mcXh/model_state_dict_0.pt'
convert_pt_checkpoint_to_tf(model_type='bert',pytorch_checkpoint_path=state_dict_path,
                           config_file='/home/eco/zyh/model/huggingface/rbt4/config.json',
                           tf_dump_path='/home/eco/zyh/model/train_fintune/tensorflow/TPlinker/',
                           compare_with_pt_model=True,
    use_cached_models=True)