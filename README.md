# Iterative Introspection based Refinement: Boosting Multi-Document Scientific Summarization with Large Language Models


## Dataset
Our dataset ComRW is available at `dataset` folder.

## Code
1. The code to generate the initial draft:

```bash
python t0_generate_initial_related_work_0shot.py
```

## Predict the ranking results of the test set
```bash
export PYTHONPATH=.

python train.py  --mode test --cuda  --data_dir <path-to-test-dataset-folder> --batch_size 8 --valid_batch_size 8 --seed 666   --visible_gpus 0 --gpu_ranks 0 --dec_dropout 0.2 --enc_dropout 0.1  --lr 0.2 --label_smoothing 0.1  --log_file ./log/log_full_test_wordlevel_copy001.txt  --inter_layers 6,7 --inter_heads 6 --dec_hidden_size 768 --doc_max_timesteps 50 --use_bert true --report_rouge --alpha 0.4 --max_length 200 --result_path ./resultmx/prompt_ctr_ --prop 3 --test_all false --sep_optim true --use_nucleus_sampling false --min_length 120  --no_repeat_ngram_size 2 --test_from <path-to-saved-reranker-checkpoint>  --bce True 
```

## Training the summarization model based on the root folder
```bash
export PYTHONPATH=.

python train.py  --mode train --cuda  --data_dir <path-to-training-dataset-folder> --batch_size 2 --seed 666 --train_steps 80000 --save_checkpoint_steps 4000  --report_every 1  --visible_gpus 0 --gpu_ranks 0  --world_size 1 --accum_count 2 --dec_dropout 0.1 --enc_dropout 0.1  --model_path  ./trained_model/train_mx_vaeR1  --log_file ./log/train_source.txt  --inter_layers 6,7 --inter_heads 8 --hier --doc_max_timesteps 50 --use_bert false --prop 3 --sep_optim false --num_workers 5 --warmup_steps 8000 --lr 0.005 --enc_layers 6  --dec_layers 6 --use_nucleus_sampling false --label_smoothing 0.1  --candidate_type 0shot  --loss_kl 0.001  --loss_bow 0.01 --kl_annealing_steps 100000 --cand_num 3  --rank_type Rouge1  
```

## Test the summarization model
```bash
export PYTHONPATH=.

python train.py  --mode test --cuda  --data_dir <path-to-test-dataset-folder> --batch_size 8 --valid_batch_size 8 --seed 666   --visible_gpus 0 --gpu_ranks 0 --dec_dropout 0.1 --enc_dropout 0.1  --lr 0.2 --label_smoothing 0.0  --log_file ./log/log_full_test_wordlevel_copy001.txt  --inter_layers 6,7 --inter_heads 8 --doc_max_timesteps 50 --use_bert false --report_rouge --alpha 0.4 --max_length 200 --result_path ./result/model_ --prop 3 --test_all false --sep_optim false   --use_bert false  --use_nucleus_sampling false --min_length 100  --no_repeat_ngram_size 2 --test_from <path-to-saved-summarization-model-checkpoint> --predrank_path  <path-to-predicted-test-ranking-result>  --cand_num 3  --enc_layers  6  --dec_layers 6  --rank_type Piratio  --use_z2 true
```

## References
```
@inproceedings{wang2024disentangling,
  title={Disentangling Instructive Information from Ranked Multiple Candidates for Multi-Document Scientific Summarization},
  author={Wang, Pancheng and Li, Shasha and Tang, Jintao and Wang, Ting},
  booktitle={Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval},
  year={2024}
}
```