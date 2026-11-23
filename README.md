# netmine

Unsupervised intrusion detection on network flows. It clusters traffic into behaviour
profiles and scores every flow for how far it sits from the normal ones. Labels exist in
the dataset but nothing is trained on them. They are only used to check the results
afterwards.

I built this because most write-ups I read on this dataset evaluate on the split as it
ships, which is 55 percent attacks. Real network traffic is not 55 percent attacks, and
precision-recall numbers computed on a split like that look much better than the same
detector would look in production. So netmine rebuilds the evaluation set at 1, 5 and 10
percent attack rates and reports all of them next to the raw split.

## Data

UNSW-NB15, the train and test files published with the paper. 175,341 flows in the
reference file and 82,332 in the evaluation file, 42 features plus a binary label and one
of nine attack categories.

`python main.py download` fetches both files into `data/`. They are not committed.

## How it works

Detectors are fitted on a reference set of 40,000 flows drawn from normal traffic only.
This is the baseline period you would collect on a real network before turning detection
on. Evaluation sets are built separately from the test file, so nothing the detectors saw
during fitting appears in the scores.

Preprocessing is fitted on the reference set alone. Columns with skew above 2 get a
log1p, everything numeric is centred on the median and scaled by the interquartile range,
and scaled values are clipped at 20. Protocol, service and state are one-hot encoded with
a vocabulary learned from the reference set, so anything the baseline period never saw
lands in an `other` column. On the clean reference that leaves 56 features, 32 of them
log transformed.

Four scores are computed for every flow, then rank averaged into an ensemble:

- `lof`, local outlier factor, written from scratch
- `iforest`, isolation forest, written from scratch
- `cluster_distance`, distance to the assigned k-means centroid divided by that cluster's
  median distance in the reference set
- `gmm_nll`, negative log likelihood under a Gaussian mixture fitted on the reference set

## What is written from scratch

LOF and Isolation Forest are mine. Blocked numpy distance computation, k-distance,
reachability distance and local reachability density for LOF. Random isolation trees with
the c(n) path length correction and vectorised level-by-level scoring for the forest.

Both are checked against scikit-learn on the same inputs as part of every run:

| check | result |
|---|---|
| LOF, largest absolute difference in score | 2.5e-06 |
| LOF, largest relative difference in score | 4.3e-07 |
| LOF, Spearman correlation | 0.9999994 |
| LOF, my time vs scikit-learn | 13.6s vs 1.4s |
| Isolation Forest, Spearman correlation | 0.987 |
| Isolation Forest, top 50 overlap | 0.82 |
| Isolation Forest, my time vs scikit-learn | 0.39s vs 0.35s |

LOF comes out numerically identical and about ten times slower, which is what you would
expect from blocked matrix distances against a KD-tree. Isolation Forest picks different
random splits so the scores cannot match exactly, and rank agreement is the thing to look
at there.

One detail that matters for LOF: 7 percent of the reference flows are exact duplicates.
With 20 neighbours a duplicate group makes the k-distance zero and the local density
infinite. The reference set is deduplicated before LOF is fitted, which removes the
problem entirely.

## Results

### Ranking quality, and what the base rate does to it

ROC-AUC is measured on the full test file. PR-AUC is measured on evaluation sets rebuilt
at each attack rate.

| detector | ROC-AUC | PR-AUC at 1% | PR-AUC at 5% | PR-AUC at 10% | PR-AUC at 55% |
|---|---|---|---|---|---|
| gmm_nll | 0.926 | 0.407 | 0.669 | 0.754 | 0.947 |
| ensemble | 0.895 | 0.209 | 0.515 | 0.645 | 0.916 |
| lof | 0.862 | 0.075 | 0.287 | 0.451 | 0.873 |
| iforest | 0.836 | 0.042 | 0.193 | 0.328 | 0.827 |
| cluster_distance | 0.724 | 0.026 | 0.121 | 0.215 | 0.735 |

The detectors are unchanged across those five columns. ROC-AUC moves by less than 0.01
across the four evaluation sets, and PR-AUC for the GMM goes from 0.407 to 0.947. The
whole gap is the base rate.

Precision at the top 100 flows on the 5 percent set: 0.96 for the GMM, 0.93 for the
ensemble, 0.58 for LOF, 0.04 for cluster distance and 0.00 for Isolation Forest.

### Recall by attack category

Thresholds are set on normal traffic to give a 5 percent false positive rate, then recall
is measured per category on the full test file. This does not depend on the base rate.

| category | support | gmm_nll | ensemble | lof | iforest |
|---|---|---|---|---|---|
| Analysis | 677 | 1.000 | 0.517 | 0.290 | 0.065 |
| Backdoor | 583 | 0.947 | 0.667 | 0.345 | 0.093 |
| Generic | 18,871 | 0.987 | 0.891 | 0.614 | 0.601 |
| DoS | 4,089 | 0.884 | 0.403 | 0.293 | 0.090 |
| Exploits | 11,132 | 0.816 | 0.551 | 0.372 | 0.112 |
| Worms | 44 | 0.386 | 0.205 | 0.227 | 0.114 |
| Fuzzers | 6,062 | 0.311 | 0.200 | 0.101 | 0.071 |
| Reconnaissance | 3,496 | 0.244 | 0.149 | 0.162 | 0.015 |
| Shellcode | 378 | 0.082 | 0.074 | 0.106 | 0.000 |
| all attacks | 45,332 | 0.780 | 0.598 | 0.410 | 0.299 |

Everything that shows up as unusual volume or unusual protocol gets caught. Fuzzers,
Reconnaissance and Shellcode do not, because at flow level they look like ordinary
traffic. No detector here gets past 31 percent on Fuzzers.

### When the baseline is not clean

Collecting 40,000 genuinely attack-free flows is the hard part in practice. This run
repeats the fit with attacks mixed into the reference set and evaluates on the same 5
percent set every time.

| detector | clean | 1% dirty | 5% dirty | 10% dirty |
|---|---|---|---|---|
| gmm_nll | 0.931 | 0.669 | 0.567 | 0.478 |
| ensemble | 0.906 | 0.806 | 0.735 | 0.709 |
| lof | 0.859 | 0.759 | 0.752 | 0.692 |
| iforest | 0.835 | 0.828 | 0.814 | 0.771 |
| cluster_distance | 0.722 | 0.618 | 0.499 | 0.601 |

The ordering flips. A Gaussian mixture fits its components around whatever it is given,
so a few thousand attacks in the baseline get modelled as normal and the detector ends up
worse than a coin flip at 10 percent. Isolation Forest loses 0.06 over the same range,
because a small contaminating group still isolates quickly. If you cannot promise a clean
baseline, the ensemble or the forest is the safer choice even though both lose to the GMM
when the baseline is clean.

### Cluster structure

Silhouette picks k = 4 on the 5 percent evaluation set, at 0.491, with purity 0.950
against the attack categories. The clusters split traffic cleanly by attack density even
though nothing told them what an attack is:

| cluster | size | attack rate |
|---|---|---|
| 0 | 6,607 | 0.000 |
| 1 | 7,945 | 0.046 |
| 2 | 3,442 | 0.194 |
| 3 | 3,059 | 0.005 |

DBSCAN on a 20,000 flow sample finds 32 clusters and calls 11.9 percent of flows noise.
Those noise flows are 13.8 percent attacks against a 5.0 percent base rate, and the flows
inside clusters are 3.8 percent attacks. Roughly a 2.7x enrichment from a method that
never saw a label.

## Running it

```
pip install -r requirements.txt
python main.py download
python main.py run
```

The full run takes about seven minutes on my laptop and writes `summary.json` plus four
CSV files into `results/`. Tests:

```
python -m pytest tests -q
```

Every setting lives in `config.toml` and is validated on load, including unknown and
missing keys.

## Layout

```
main.py             command line entry point
config.toml         every tunable
netmine/config.py   loading and validation
netmine/data.py     download and schema checks
netmine/features.py encoding and scaling
netmine/splits.py   reference and evaluation set construction
netmine/lof.py      local outlier factor
netmine/iforest.py  isolation forest
netmine/cluster.py  k-means, DBSCAN, Gaussian mixture
netmine/scoring.py  detector bank and rank ensemble
netmine/evaluate.py metrics
netmine/verify.py   comparison against scikit-learn
netmine/pipeline.py the run
tests/              42 tests
```
