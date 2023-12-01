# WebNLG Toolkit

This repo is intended to provide common utilities for WebNLG-related tasks. This includes data loading and processing, model training and evaluation. It also provides access to various baseline models.

```bash
pip install -e .
```

## Loading WebNLG Data

WebNLG XML datasets can be automatically loaded into python using utility functions. 

For example, `load_webnlg_dataset()` will automatically apply standard preprocessing of triples and lexicalisations and return input/output pairs ready to be used for seq2seq model training and inference.
Use the `lang` parameter to filter out lexicalisations of the desired language(s), `multilingual` to add a language-specific prefix, and `task` to specify whether you intend to perform RDF-to-Text or Text-to-RDF.

```python
>>> from webnlg_toolkit.utils.data import load_webnlg_dataset

>>> data = load_webnlg_dataset(test_file, lang="en", task="rdf2text")
>>> data[:3]
[('<S> Aarhus Airport <P> city served <O> Aarhus, Denmark',
  'The Aarhus is the airport of Aarhus, Denmark.'),
 ('<S> Aarhus Airport <P> city served <O> Aarhus, Denmark',
  'Aarhus Airport serves the city of Aarhus, Denmark.'),
 ('<S> Aarhus Airport <P> city served <O> Aarhus',
  'Aarhus airport serves the city of Aarhus.')]

# set `training=False` to get multi-reference examples 
>>> data = load_webnlg_dataset(test_file, lang="en", task="rdf2text", training=False)
>>> data[:3]
[('<S> Aarhus Airport <P> city served <O> Aarhus, Denmark',
  [('The Aarhus is the airport of Aarhus, Denmark.', 'en'),
   ('Aarhus Airport serves the city of Aarhus, Denmark.', 'en')]),
 ('<S> Aarhus Airport <P> city served <O> Aarhus',
  [('Aarhus airport serves the city of Aarhus.', 'en')]),
 ('<S> Aarhus Airport <P> elevation above the sea level <O> 25.0',
  [('Aarhus Airport is 25 metres above sea level.', 'en'),
   ('Aarhus airport is at an elevation of 25 metres above seal level.', 'en'),
   ('Aarhus Airport is 25.0 metres above the sea level.', 'en')])]

# set `multilingual=True` to add task prefix
>>> data = load_webnlg_dataset(test_file, lang="all", task="rdf2text", multilingual=True)
>> data[:3]
[('RDF-to-text in en: <S> Aarhus <P> leader <O> Jacob Bundsgaard',
  'The leader of Aarhus is Jacob Bundsgaard.'),
 ('RDF-to-text in ga: <S> Aarhus <P> leader <O> Jacob Bundsgaard',
  'Is é Jacob Bundsgaard ceannaire Aarhus.'),
 ('RDF-to-text in cy: <S> Aarhus <P> leader <O> Jacob Bundsgaard',
  "Arweinydd Aarhus yw Jacob Bundsgaard.")]
```

It is also possible to load the entire dataset without seq2seq preprocessing via `load_webnlg_xml()`. By default this returns a `List` containing a `dict` for each entry, but a `Benchmark` object (from the original [corpus reader](https://gitlab.com/webnlg/corpus-reader/-/tree/master)) can also be return by setting `return_type="benchmark"`. 
```python
>>> from webnlg_toolkit.utils.data import load_webnlg_xml

>>> b = load_webnlg_xml("cy_dev.xml")
>>> b[:1]
[{'category': 'Airport',
  'size': '1',
  'xml_id': '1',
  'shape': '(X (X))',
  'shape_type': 'NA',
  'originaltriplesets': {'originaltripleset': [[{'subject': 'Aarhus',
      'property': 'leaderName',
      'object': 'Jacob_Bundsgaard'}]]},
  'modifiedtripleset': [{'subject': 'Aarhus',
    'property': 'leader',
    'object': 'Jacob_Bundsgaard'}],
  'lexicalisations': [{'comment': 'good',
    'xml_id': 'Id1',
    'lex': 'The leader of Aarhus is Jacob Bundsgaard.',
    'lang': 'en'},
   {'comment': '',
    'xml_id': 'Id2',
    'lex': 'Arweinydd Aarhus yw Jacob Bundsgaard.',
    'lang': 'cy'}],
  'dbpedialinks': [],
  'links': []}]

>>> b = load_webnlg_xml("cy_dev.xml", return_type="benchmark")
>>> type(b)
webnlg_toolkit.utils.benchmark_reader.Benchmark
```

## Training
We provide code to handle basic training of seq2seq models for WebNLG tasks. Currently this works for any T5-type transformers model (T5, mT5, mT0, etc.). 

This can be called directly from the terminal like the following examples (see `webnlg_toolkit/t5.py` for the full range of possible arguments).

```bash
python webnlg_toolkit/train_mt5.py
  --train_file=en_train.xml
  --val_file=en_dev.xml
  --val_check_interval=0.25
  --base_model=t5-base
  --save_dir=model_ckpts
  --name=en_ft_t5
  --lang=en
  --gpus=2

python webnlg_toolkit/train_mt5.py
  --train_file=all_train.xml
  --val_file=all_dev.xml
  --val_check_interval=0.25
  --base_model=google/mt5-base
  --save_dir=model_ckpts
  --name=all_ft_mt5
  --lang=all
  --gpus=2
  --multilingual # flag to add task prefix

python webnlg_toolkit/train_t5.py
  --train_file=en_train.xml
  --val_file=en_dev.xml
  --val_check_interval=0.25
  --base_model=google/t5-base
  --save_dir=model_ckpts
  --name=en_t2r_t5
  --lang=en
  --gpus=2
  --task=text2rdf # flag for semantic parsing task
```

After training a model, we recommend using `.save_model()` to save in pytorch format prior to inference/evalution.
```python
from webnlg_toolkit.t5 import T5Module

model = T5Module.load_from_checkpoint(model_ckpt, **kwargs)
model.save_model("my_new_model")
```

## Evaluation

Seq2seq model inference and evaluation can be performed via the `inference()` function. Setting `eval=True` will automatically performed evaluation on the generated outputs.

```python
>>> from webnlg_toolkit.t5 import inference

>>> df = inference("webnlg/en_t5base", "en_test.xml", lang="en", eval=True)
   BLEU    chrF++    TER    BERT-SCORE P    BERT-SCORE R    BERT-SCORE F1
-------  --------  -----  --------------  --------------  ---------------
44.163     0.627  0.575           0.942           0.941            0.941

>>> df.columns
Index(['input', 'ref', 'output', 'bleu', 'ter', 'bert_precision',
       'bert_recall', 'bert_f1'],
      dtype='object')
```

Alternatively, the traditional method of evaluation that reads hypothesis and references for individual text files can still be performed from the terminal.

```bash
python webnlg_toolkit/eval/eval.py
  --hypothesis ga_hyps.txt
  --reference ga_test_lexs.txt
  --metrics bleu,chrf++,ter,bert
  --language ga
  --num_refs 1
```

In case of multiple references, they have to be stored in separated files and named reference0, reference1, reference2, etc. like in the following [example](https://github.com/WebNLG/GenerationEval/tree/master/data/en/references). 

```bash
# with multiple references in a series of files `ru_test_lexs0`, `ru_test_lexs1`, `ru_test_lexs2`, etc.
python webnlg_toolkit/eval/eval.py
  --hypothesis ru_hyps.txt
  --reference ru_test_lexs
  --metrics bleu,chrf++,ter,bert
  --language ru
  --num_refs 7
```

## RDF-to-Text Baselines
We provide a number of basic [pretrained models](https://huggingface.co/webnlg) for you to use as baselines or in your own projects. These include multilingual and monolingual models for specific languages.

[>>> See pretrained models](https://huggingface.co/webnlg)

* `en-t5base` - [T5-base](https://huggingface.co/t5-base) fine-tuned on the WebNLG 2020 English data.
* `ru-mt0base` - [mT0-base](https://huggingface.co/bigscience/mt0-base) fine-tuned on the WebNLG 2020/2023 Russian data.
* `all-mt5base` - [mT5-base](https://huggingface.co/google/mt5-base) fine-tuned on the WebNLG 2020/2023 data for the full range of supported languages (en, ru, br, cy, ga, mt)
* `all-mt5large` - [mT5-large](https://huggingface.co/google/mt5-large) fine-tuned on the WebNLG 2020/2023 data for the full range of supported languages (en, ru, br, cy, ga, mt)

_Note: for cy, ga, and mt we use NLLB translations of the English training data instead of the automatic translations originally published in WebNLG 2023 as this is believed to be of higher quality. However, we use the original 2023 training data for br as it is not supported by NLLB._

### English
|                 | BLEU   | chrF++ | TER   | BERT Prec. | BERT Rec. | BERT F1 |
|-----------------|--------|--------|-------|------------|-----------|---------|
| English T5-base | 52.569 | 0.680  | 0.411 | 0.958      | 0.955     | 0.956   |
| All mT5-base    | 44.163 | 0.627  | 0.575 | 0.942      | 0.941     | 0.941   |
| All mT5-large   | 44.019 | 0.634  | 0.558 | 0.942      | 0.942     | 0.941   |

### Russian
|                  | BLEU   | chrF++ | TER   | BERT Prec. | BERT Rec. | BERT F1 |
|------------------|--------|--------|-------|------------|-----------|---------|
| Russian mT0-base | 52.227 | 0.685  | 0.397 | 0.915      | 0.910     | 0.912   |
| All mT5-base     | 51.861 | 0.684  | 0.393 | 0.916      | 0.909     | 0.911   |
| All mT5-large    | 51.954 | 0.686  | 0.401 | 0.914      | 0.908     | 0.910   |

### Breton
|               | BLEU  | chrF++ | TER   | BERT Prec. | BERT Rec. | BERT F1 |
|---------------|-------|--------|-------|------------|-----------|---------|
| All mT5-base  | 7.747 | 0.257  | 0.807 | 0.754      | 0.709     | 0.729   |
| All mT5-large | 8.232 | 0.266  | 0.813 | 0.747      | 0.711     | 0.727   |

### Welsh
|               | BLEU   | chrF++ | TER   | BERT Prec. | BERT Rec. | BERT F1 |
|---------------|--------|--------|-------|------------|-----------|---------|
| All mT5-base  | 15.519 | 0.431  | 0.804 | 0.786      | 0.779     | 0.782   |
| All mT5-large | 15.498 | 0.432  | 0.820 | 0.782      | 0.779     | 0.780   |

### Irish
|               | BLEU   | chrF++ | TER   | BERT Prec. | BERT Rec. | BERT F1 |
|---------------|--------|--------|-------|------------|-----------|---------|
| All mT5-base  | 14.574 | 0.419  | 0.782 | 0.778      | 0.764     | 0.771   |
| All mT5-large | 15.048 | 0.427  | 0.765 | 0.780      | 0.767     | 0.773   |

### Maltese
|               | BLEU   | chrF++ | TER   |
|---------------|--------|--------|-------|
| All mT5-base  | 13.384 | 0.434  | 0.776 |
| All mT5-large | 13.486 | 0.436  | 0.784 |
