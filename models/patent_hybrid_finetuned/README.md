---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:30
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Smart drinking bottle with capacitive fluid sensors measuring liquid
    volume and sending drinking reminders
  sentences:
  - Smart Hydration and Vital Sign Monitoring Drinking Vessel with Capacitive Fluid
    Sensor. A smart connected drinking vessel measures consumed fluid volume via capacitive
    sensor strips positioned longitudinally along the container walls. Integrated
    optical photoplethysmography sensors in the container base detect heart rate and
    blood oxygenation when handled, transmitting telemetry to a companion smartphone
    app.
  - Federated Edge Learning with Differential Privacy and Quantized Gradient Synchronization.
    A decentralized federated machine learning framework enables distributed edge
    devices to cooperatively train deep neural networks without transmitting raw user
    data. Local models compute parameter gradients, inject calibrated differential
    privacy noise, and apply adaptive 4-bit quantization before synchronizing with
    a central parameter aggregator.
  - Soft Robotic End-Effector Gripper with Integrated Piezoresistive Tactile Sensor
    Arrays. A pneumatic soft robotic gripper utilizes elastomeric silicone fingers
    embedded with flexible 3D-printed piezoresistive sensor meshes. Real-time contact
    pressure and shear stress distributions are fed back to an adaptive grasp controller,
    allowing delicate handling of fragile agricultural produce and slippery surgical
    instruments without crushing.
- source_sentence: Lipid nanoparticle formulation encapsulating mRNA with ionizable
    lipids and PEG for cellular delivery
  sentences:
  - Solid-State Lithium Battery with Ceramic Sulfide Electrolyte and Silicon Nanowire
    Anode. A rechargeable solid-state lithium-metal battery incorporates a ceramic
    sulfide-based solid electrolyte membrane bonded to a high-capacity silicon nanowire
    anode and a nickel-cobalt-manganese cathode. An in-situ formed elastic polymer
    interlayer accommodates volumetric expansion during high-rate charging and prevents
    lithium dendrite propagation.
  - Federated Edge Learning with Differential Privacy and Quantized Gradient Synchronization.
    A decentralized federated machine learning framework enables distributed edge
    devices to cooperatively train deep neural networks without transmitting raw user
    data. Local models compute parameter gradients, inject calibrated differential
    privacy noise, and apply adaptive 4-bit quantization before synchronizing with
    a central parameter aggregator.
  - Lipid Nanoparticle Formulations for Targeted mRNA Delivery and Cellular Transfection.
    Ionizable cationic lipid nanoparticle compositions encapsulate synthetic messenger
    RNA encoding therapeutic antigens. Formulations incorporate biodegradable ester
    linkers, cholesterol, and PEGylated lipids, achieving high encapsulation efficiency,
    reduced hepatic accumulation, and targeted endosomal escape into dendritic immune
    cells.
- source_sentence: Hardware cryptographic processor accelerating number theoretic
    transform for lattice Kyber post-quantum encryption
  sentences:
  - Federated Edge Learning with Differential Privacy and Quantized Gradient Synchronization.
    A decentralized federated machine learning framework enables distributed edge
    devices to cooperatively train deep neural networks without transmitting raw user
    data. Local models compute parameter gradients, inject calibrated differential
    privacy noise, and apply adaptive 4-bit quantization before synchronizing with
    a central parameter aggregator.
  - Solid-State Lithium Battery with Ceramic Sulfide Electrolyte and Silicon Nanowire
    Anode. A rechargeable solid-state lithium-metal battery incorporates a ceramic
    sulfide-based solid electrolyte membrane bonded to a high-capacity silicon nanowire
    anode and a nickel-cobalt-manganese cathode. An in-situ formed elastic polymer
    interlayer accommodates volumetric expansion during high-rate charging and prevents
    lithium dendrite propagation.
  - Hardware Cryptographic Accelerator for Post-Quantum Lattice-Based Key Encapsulation
    (Kyber). A dedicated hardware coprocessor accelerates number theoretic transform
    (NTT) polynomial multiplications for post-quantum lattice cryptography (CRYSTALS-Kyber).
    Constant-time modular arithmetic datapaths and randomized side-channel masking
    prevent differential power analysis and timing leakages during private key operations.
- source_sentence: Smart drinking bottle with capacitive fluid sensors measuring liquid
    volume and sending drinking reminders
  sentences:
  - Vision Transformer with Hierarchical Spatial Patch Merging for Real-Time Object
    Detection. A hierarchical vision transformer extracts multi-scale visual features
    from digital images by recursively merging adjacent image patches across successive
    transformer stages. Linear self-attention within shifted windows provides linear
    computational complexity with respect to image resolution, enabling high-fps bounding
    box localization and semantic segmentation.
  - Smart Hydration and Vital Sign Monitoring Drinking Vessel with Capacitive Fluid
    Sensor. A smart connected drinking vessel measures consumed fluid volume via capacitive
    sensor strips positioned longitudinally along the container walls. Integrated
    optical photoplethysmography sensors in the container base detect heart rate and
    blood oxygenation when handled, transmitting telemetry to a companion smartphone
    app.
  - Hardware Cryptographic Accelerator for Post-Quantum Lattice-Based Key Encapsulation
    (Kyber). A dedicated hardware coprocessor accelerates number theoretic transform
    (NTT) polynomial multiplications for post-quantum lattice cryptography (CRYSTALS-Kyber).
    Constant-time modular arithmetic datapaths and randomized side-channel masking
    prevent differential power analysis and timing leakages during private key operations.
- source_sentence: Lipid nanoparticle formulation encapsulating mRNA with ionizable
    lipids and PEG for cellular delivery
  sentences:
  - Hardware Cryptographic Accelerator for Post-Quantum Lattice-Based Key Encapsulation
    (Kyber). A dedicated hardware coprocessor accelerates number theoretic transform
    (NTT) polynomial multiplications for post-quantum lattice cryptography (CRYSTALS-Kyber).
    Constant-time modular arithmetic datapaths and randomized side-channel masking
    prevent differential power analysis and timing leakages during private key operations.
  - Vision Transformer with Hierarchical Spatial Patch Merging for Real-Time Object
    Detection. A hierarchical vision transformer extracts multi-scale visual features
    from digital images by recursively merging adjacent image patches across successive
    transformer stages. Linear self-attention within shifted windows provides linear
    computational complexity with respect to image resolution, enabling high-fps bounding
    box localization and semantic segmentation.
  - Lipid Nanoparticle Formulations for Targeted mRNA Delivery and Cellular Transfection.
    Ionizable cationic lipid nanoparticle compositions encapsulate synthetic messenger
    RNA encoding therapeutic antigens. Formulations incorporate biodegradable ester
    linkers, cholesterol, and PEGylated lipids, achieving high encapsulation efficiency,
    reduced hepatic accumulation, and targeted endosomal escape into dendritic immune
    cells.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps inputs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Lipid nanoparticle formulation encapsulating mRNA with ionizable lipids and PEG for cellular delivery',
    'Lipid Nanoparticle Formulations for Targeted mRNA Delivery and Cellular Transfection. Ionizable cationic lipid nanoparticle compositions encapsulate synthetic messenger RNA encoding therapeutic antigens. Formulations incorporate biodegradable ester linkers, cholesterol, and PEGylated lipids, achieving high encapsulation efficiency, reduced hepatic accumulation, and targeted endosomal escape into dendritic immune cells.',
    'Vision Transformer with Hierarchical Spatial Patch Merging for Real-Time Object Detection. A hierarchical vision transformer extracts multi-scale visual features from digital images by recursively merging adjacent image patches across successive transformer stages. Linear self-attention within shifted windows provides linear computational complexity with respect to image resolution, enabling high-fps bounding box localization and semantic segmentation.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.8675, -0.0936],
#         [ 0.8675,  1.0000, -0.0416],
#         [-0.0936, -0.0416,  1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 30 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 30 samples:
  |          | sentence_0                                                                        | sentence_1                                                                         | sentence_2                                                                          |
  |:---------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type     | string                                                                            | string                                                                             | string                                                                              |
  | modality | text                                                                              | text                                                                               | text                                                                                |
  | details  | <ul><li>min: 17 tokens</li><li>mean: 21.3 tokens</li><li>max: 25 tokens</li></ul> | <ul><li>min: 73 tokens</li><li>mean: 87.7 tokens</li><li>max: 105 tokens</li></ul> | <ul><li>min: 77 tokens</li><li>mean: 83.63 tokens</li><li>max: 101 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence_2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:--------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Hardware cryptographic processor accelerating number theoretic transform for lattice Kyber post-quantum encryption</code> | <code>Hardware Cryptographic Accelerator for Post-Quantum Lattice-Based Key Encapsulation (Kyber). A dedicated hardware coprocessor accelerates number theoretic transform (NTT) polynomial multiplications for post-quantum lattice cryptography (CRYSTALS-Kyber). Constant-time modular arithmetic datapaths and randomized side-channel masking prevent differential power analysis and timing leakages during private key operations.</code>                             | <code>Intelligent Battery Management System with Ultrasonic State-of-Charge and Degradation Monitoring. A battery pack management system transmits high-frequency ultrasonic acoustic pulses through individual lithium-ion cells to detect internal structural phase transitions, electrolyte depletion, and mechanical delamination. Time-of-flight acoustic wave velocity measurements provide non-invasive state-of-charge (SoC) and state-of-health (SoH) diagnostics.</code>                                       |
  | <code>Non-invasive continuous glucose monitor measuring near-infrared laser Raman spectra from skin capillaries</code>          | <code>Non-Invasive Continuous Glucose Monitor Using Multi-Wavelength Optical Raman Spectroscopy. A wrist-worn wearable medical diagnostic device projects polarized multi-wavelength near-infrared lasers into capillary dermis tissue. Inelastically scattered Raman photons are collected by a compact spectrometer, and a machine learning regression model isolates spectral fingerprint signatures of blood glucose from overlapping collagen and albumin noise.</code> | <code>Vision Transformer with Hierarchical Spatial Patch Merging for Real-Time Object Detection. A hierarchical vision transformer extracts multi-scale visual features from digital images by recursively merging adjacent image patches across successive transformer stages. Linear self-attention within shifted windows provides linear computational complexity with respect to image resolution, enabling high-fps bounding box localization and semantic segmentation.</code>                                    |
  | <code>Solid-state lithium battery using ceramic sulfide electrolyte and silicon nanowire anode to stop dendrites</code>         | <code>Solid-State Lithium Battery with Ceramic Sulfide Electrolyte and Silicon Nanowire Anode. A rechargeable solid-state lithium-metal battery incorporates a ceramic sulfide-based solid electrolyte membrane bonded to a high-capacity silicon nanowire anode and a nickel-cobalt-manganese cathode. An in-situ formed elastic polymer interlayer accommodates volumetric expansion during high-rate charging and prevents lithium dendrite propagation.</code>           | <code>Sparse Multi-Head Attention Neural Network for Efficient Sequence Processing. A transformer-based neural network architecture employs sparse factorized multi-head self-attention mechanisms to process long sequence representations with linear computational complexity. Dedicated memory routing gates dynamically route query tokens to partitioned key-value blocks, drastically reducing quadratic quadratic matrix multiplication overhead while preserving semantic cross-context representations.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 1
- `disable_tqdm`: True
- `per_device_eval_batch_size`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: True
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 4
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Time
- **Training**: 14.0 seconds

### Framework Versions
- Python: 3.14.4
- Sentence Transformers: 6.2.0
- Transformers: 5.22.2
- PyTorch: 2.15.0+cpu
- Accelerate: 1.15.0
- Datasets: 5.1.0
- Tokenizers: 0.23.3

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->