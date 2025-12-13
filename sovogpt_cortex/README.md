---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: where is cuttack
- text: identify yourself
- text: tame kie
- text: good morning
- text: what are you doing
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/paraphrase-MiniLM-L3-v2
---

# SetFit with sentence-transformers/paraphrase-MiniLM-L3-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 128 tokens
- **Number of Classes:** 4 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                      |
|:------|:----------------------------------------------------------------------------------------------|
| 0     | <ul><li>'tu kie'</li><li>'tame kie'</li><li>'apana kie'</li></ul>                             |
| 1     | <ul><li>'namaskar'</li><li>'hi'</li><li>'hello'</li></ul>                                     |
| 2     | <ul><li>'weather'</li><li>'weather in bhubaneswar'</li><li>'puri re weather kimiti'</li></ul> |
| 3     | <ul><li>'odisha ra cm kiye'</li><li>'who is the pm'</li><li>'india capital'</li></ul>         |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("tame kie")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
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

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 1   | 2.3864 | 5   |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 9                     |
| 1     | 14                    |
| 2     | 9                     |
| 3     | 12                    |

### Training Hyperparameters
- batch_size: (4, 4)
- num_epochs: (10, 10)
- max_steps: -1
- sampling_strategy: oversampling
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0028 | 1    | 0.168         | -               |
| 0.1393 | 50   | 0.3244        | -               |
| 0.2786 | 100  | 0.2729        | -               |
| 0.4178 | 150  | 0.2511        | -               |
| 0.5571 | 200  | 0.2344        | -               |
| 0.6964 | 250  | 0.2063        | -               |
| 0.8357 | 300  | 0.1904        | -               |
| 0.9749 | 350  | 0.1681        | -               |
| 1.1142 | 400  | 0.1327        | -               |
| 1.2535 | 450  | 0.0831        | -               |
| 1.3928 | 500  | 0.0734        | -               |
| 1.5320 | 550  | 0.0533        | -               |
| 1.6713 | 600  | 0.0401        | -               |
| 1.8106 | 650  | 0.0188        | -               |
| 1.9499 | 700  | 0.0128        | -               |
| 2.0891 | 750  | 0.0114        | -               |
| 2.2284 | 800  | 0.0083        | -               |
| 2.3677 | 850  | 0.0076        | -               |
| 2.5070 | 900  | 0.0062        | -               |
| 2.6462 | 950  | 0.0061        | -               |
| 2.7855 | 1000 | 0.0062        | -               |
| 2.9248 | 1050 | 0.0059        | -               |
| 3.0641 | 1100 | 0.0054        | -               |
| 3.2033 | 1150 | 0.0042        | -               |
| 3.3426 | 1200 | 0.0042        | -               |
| 3.4819 | 1250 | 0.004         | -               |
| 3.6212 | 1300 | 0.0041        | -               |
| 3.7604 | 1350 | 0.0034        | -               |
| 3.8997 | 1400 | 0.0032        | -               |
| 4.0390 | 1450 | 0.0036        | -               |
| 4.1783 | 1500 | 0.0033        | -               |
| 4.3175 | 1550 | 0.0032        | -               |
| 4.4568 | 1600 | 0.0031        | -               |
| 4.5961 | 1650 | 0.0034        | -               |
| 4.7354 | 1700 | 0.003         | -               |
| 4.8747 | 1750 | 0.0025        | -               |
| 5.0139 | 1800 | 0.0027        | -               |
| 5.1532 | 1850 | 0.0025        | -               |
| 5.2925 | 1900 | 0.0022        | -               |
| 5.4318 | 1950 | 0.0027        | -               |
| 5.5710 | 2000 | 0.0027        | -               |
| 5.7103 | 2050 | 0.0027        | -               |
| 5.8496 | 2100 | 0.0022        | -               |
| 5.9889 | 2150 | 0.0025        | -               |
| 6.1281 | 2200 | 0.0025        | -               |
| 6.2674 | 2250 | 0.0026        | -               |
| 6.4067 | 2300 | 0.002         | -               |
| 6.5460 | 2350 | 0.0023        | -               |
| 6.6852 | 2400 | 0.0023        | -               |
| 6.8245 | 2450 | 0.0023        | -               |
| 6.9638 | 2500 | 0.0024        | -               |
| 7.1031 | 2550 | 0.0019        | -               |
| 7.2423 | 2600 | 0.002         | -               |
| 7.3816 | 2650 | 0.0023        | -               |
| 7.5209 | 2700 | 0.0019        | -               |
| 7.6602 | 2750 | 0.0018        | -               |
| 7.7994 | 2800 | 0.0017        | -               |
| 7.9387 | 2850 | 0.0021        | -               |
| 8.0780 | 2900 | 0.002         | -               |
| 8.2173 | 2950 | 0.0021        | -               |
| 8.3565 | 3000 | 0.0018        | -               |
| 8.4958 | 3050 | 0.0019        | -               |
| 8.6351 | 3100 | 0.0017        | -               |
| 8.7744 | 3150 | 0.0019        | -               |
| 8.9136 | 3200 | 0.0018        | -               |
| 9.0529 | 3250 | 0.0021        | -               |
| 9.1922 | 3300 | 0.0017        | -               |
| 9.3315 | 3350 | 0.0017        | -               |
| 9.4708 | 3400 | 0.0017        | -               |
| 9.6100 | 3450 | 0.0019        | -               |
| 9.7493 | 3500 | 0.0016        | -               |
| 9.8886 | 3550 | 0.0017        | -               |

### Framework Versions
- Python: 3.10.19
- SetFit: 1.1.3
- Sentence Transformers: 5.1.2
- Transformers: 4.57.3
- PyTorch: 2.9.1
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
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