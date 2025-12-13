---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: news today
- text: tumara nama kuha
- text: tume kie?
- text: shubha sakala
- text: tume ka'na karucha?
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
| Label | Examples                                                                                           |
|:------|:---------------------------------------------------------------------------------------------------|
| 0     | <ul><li>'who are you'</li><li>'tume kie?'</li><li>'what is your name'</li></ul>                    |
| 1     | <ul><li>'hello'</li><li>'hae , hae'</li><li>'how are you'</li></ul>                                |
| 2     | <ul><li>'weather in london'</li><li>'landanare paga'</li><li>'temperature today'</li></ul>         |
| 3     | <ul><li>'capital of france'</li><li>'phransara rajadhani'</li><li>'who is the president'</li></ul> |

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
preds = model("tume kie?")
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
| Word count   | 1   | 2.6522 | 4   |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 10                    |
| 1     | 14                    |
| 2     | 10                    |
| 3     | 12                    |

### Training Hyperparameters
- batch_size: (4, 4)
- num_epochs: (5, 5)
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
| 0.0025 | 1    | 0.3797        | -               |
| 0.1269 | 50   | 0.3148        | -               |
| 0.2538 | 100  | 0.239         | -               |
| 0.3807 | 150  | 0.2214        | -               |
| 0.5076 | 200  | 0.2092        | -               |
| 0.6345 | 250  | 0.1812        | -               |
| 0.7614 | 300  | 0.1564        | -               |
| 0.8883 | 350  | 0.0902        | -               |
| 1.0152 | 400  | 0.0869        | -               |
| 1.1421 | 450  | 0.0603        | -               |
| 1.2690 | 500  | 0.0418        | -               |
| 1.3959 | 550  | 0.0356        | -               |
| 1.5228 | 600  | 0.0222        | -               |
| 1.6497 | 650  | 0.0159        | -               |
| 1.7766 | 700  | 0.0104        | -               |
| 1.9036 | 750  | 0.0106        | -               |
| 2.0305 | 800  | 0.0078        | -               |
| 2.1574 | 850  | 0.0072        | -               |
| 2.2843 | 900  | 0.0064        | -               |
| 2.4112 | 950  | 0.0053        | -               |
| 2.5381 | 1000 | 0.0055        | -               |
| 2.6650 | 1050 | 0.0054        | -               |
| 2.7919 | 1100 | 0.0046        | -               |
| 2.9188 | 1150 | 0.0042        | -               |
| 3.0457 | 1200 | 0.0046        | -               |
| 3.1726 | 1250 | 0.0041        | -               |
| 3.2995 | 1300 | 0.0043        | -               |
| 3.4264 | 1350 | 0.0034        | -               |
| 3.5533 | 1400 | 0.0039        | -               |
| 3.6802 | 1450 | 0.0043        | -               |
| 3.8071 | 1500 | 0.0034        | -               |
| 3.9340 | 1550 | 0.0031        | -               |
| 4.0609 | 1600 | 0.0034        | -               |
| 4.1878 | 1650 | 0.0036        | -               |
| 4.3147 | 1700 | 0.0028        | -               |
| 4.4416 | 1750 | 0.0033        | -               |
| 4.5685 | 1800 | 0.0031        | -               |
| 4.6954 | 1850 | 0.0035        | -               |
| 4.8223 | 1900 | 0.0035        | -               |
| 4.9492 | 1950 | 0.0022        | -               |

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