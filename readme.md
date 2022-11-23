Cross Domain NER
==========

## Paper
[A Label-Aware Autoregressive Framework for Cross-Domain NER](https://aclanthology.org/2022.findings-naacl.171/) (Accepted in NAACL2022-Findings)

## Citations

If you use or extend our work, please cite our paper at NAACL-2022.
```
@inproceedings{hu-etal-2022-label,
    title = "A Label-Aware Autoregressive Framework for Cross-Domain {NER}",
    author = "Hu, Jinpeng  and
      Zhao, He  and
      Guo, Dan  and
      Wan, Xiang  and
      Chang, Tsung-Hui",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022"
}
```


## Requirements

- Python 3 (tested on 3.7)
- PyTorch (tested on 1.7)
- Transformers (tested on 3.0.2)

We use a Linux platform with A100 GPU to train our model.

==============================


## Data
We give an example about the example source domain data in the `ner_data/conll2003` and target domain data in the `ner_data/ai`.

##DAPT
For DAPT, we follow [CrossNER](https://github.com/zliucr/CrossNER)

##Training 

###Train the NER model with DAPT
We give an example train shell file, you just need to run
```
python main.py \
--exp_name ai_experiment \
--exp_id ai_experiment \
--num_tag 29 \
--batch_size 16 \
--ckpt ./CrossNER_pre_trained/ai_spanlevel_integrated/pytorch_model.bin \
--tgt_dm ai \
--target_sequence \
--seed 8888 \
--target_embedding_dim 100 \
--target_type RNN \
--connect_label_background \
--conll
```
`ckpt` is the path to your pre-trained model after DAPT.

###Train the NER model without DAPT
```
python main.py \
--exp_name ai_experiment_wo_DAPT \
--exp_id ai_experiment_wo_DAPT \
--num_tag 29 \
--batch_size 16 \
--model_name=bert-base-cased \
--tgt_dm ai \
--target_sequence \
--seed 8888 \
--target_embedding_dim 100 \
--target_type RNN \
--connect_label_background \
--conll
```
`model_name` is the path to your pre-trained model.
