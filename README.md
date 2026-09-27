# Satellite Collision Risk Prediction

**[Explore the results interactively](https://reecemill.github.io/collision/)**: pick any test event and see its risk over time and what the rule and the model predicted.

## The problem
Two objects in orbit can collide, causing anything from slight damage to the end of a mission. Satellite operators receive data about upcoming close approaches and have to decide whether a threat is real and whether to maneuver their satellite out of the way. Maneuvering isn't free: it uses fuel and money, both limited resources. Maneuvering when it isn't needed wastes them, and not maneuvering when it is could cost the satellite.

## The goal
Predict, two days before closest approach, whether a close approach will end dangerous, meaning a final collision risk of −6 or higher (about 1 in a million). The aim is to flag credible threats early and avoid spending fuel on ones that turn out to be harmless. Two days gives operators time to review the threat, brief the people who approve a maneuver, and plan it.

## The data
The data comes from the European Space Agency's Kelvins Collision Avoidance Challenge: real, anonymized records from ESA satellite operations between 2015 and 2019. It covers 13,154 close-approach events. Each event is a series of updates (Conjunction Data Messages) sent over the days before closest approach, about 12 per event on average. In total that's 162,634 updates, each with 103 measurements such as miss distance, relative speed, and position uncertainty.

## What I've found so far
Dangerous events are rare: only 365 of 13,154 (2.8%) end with a risk of −6 or higher. That makes accuracy misleading, since a model that always says "safe" would be 97% accurate and useless. Risk can also change sharply at the end. Event 5 looked dangerous for five days, then dropped to effectively zero. Most dangerous events do look dangerous early (91% did three days out), but so do many harmless ones: of 705 events that looked dangerous three days out, only 276 ended that way. The hard part isn't spotting danger early; it's telling real threats from false alarms.

## Baseline
Before building a model, I tested the simplest possible forecast: take the most recent risk available two days before closest approach and assume it doesn't change. Of 323 dangerous events, it caught 300 (93%) and missed 23. It also raised 269 false alarms, so only about half of its warnings were real. That gives an F2 score of about 0.81; F2 weights catching threats more heavily than avoiding false alarms. 1,212 events had no data from two or more days out and couldn't be forecast at all. Any model I build has to beat this rule, or it isn't adding anything.

## Why the baseline is hard to beat
Some events get no new updates after the two-day mark. For those, the most recent risk two days out is also the final risk, so the baseline is right by definition. That's 3,050 of the 11,942 events that can be forecast, including 230 of the 323 dangerous ones. All of the baseline's mistakes come from the other 8,892 events, where updates keep arriving after the forecast point. There it caught 70 of 93 dangerous events, missed 23, raised 269 false alarms, and scored an F2 of only 0.49. That's where the real forecasting problem is, so I score models on both groups.

## Model
I trained a gradient-boosting classifier (scikit-learn's `HistGradientBoostingClassifier`) on what's known about each event two days before closest approach:

- **The most recent update:** all of its measurements (about 100 columns), such as miss distance, relative speed, and position uncertainty.
- **Trend features from earlier updates:** how much the risk and the miss distance changed since the previous update, the highest and lowest risk so far, and the number of updates so far.

Every feature uses only updates from two or more days out, the information an operator would actually have. Dangerous events are rare, so they count more heavily in training. The model outputs a probability, and the cutoff for flagging an event is chosen to maximize F2 using cross-validated predictions on the training events only.

Events are split 80/20 into training and test sets, keeping the same share of dangerous events in each. One split turned out not to be enough. On the first split I tried, the model beat the baseline (F2 0.821 against 0.800), but it lost on four of five other splits. All results below are averages over those five splits.

## Results
F2, averaged over five train/test splits:

| Features | All events: model | All events: baseline | Risk can still change: model | Risk can still change: baseline |
|---|---|---|---|---|
| Most recent update only | 0.798 | 0.817 | 0.454 | 0.499 |
| Most recent update + trends | 0.812 | 0.817 | 0.502 | 0.499 |

On the events where the risk can still change, the model with trends matches the baseline's F2 while raising about a third fewer false alarms (32 per test split against 48), at the cost of more missed threats (6.4 against 4.8). Overall it's still slightly behind, because on events that get no more updates it sometimes second-guesses a forecast that can't be wrong.

These differences are small enough to depend on the software version. With scikit-learn 1.9 instead of 1.7, the same notebook gives the model 0.485 without trend features and 0.461 with them on the events where the risk can still change, so the trend features' apparent gain isn't reliable. Two things hold in both versions: the model doesn't beat the baseline overall, and it raises far fewer false alarms (26 to 32 per test split against 48) while missing more threats (6.4 to 8 against 4.8). `requirements.txt` pins the versions that produce the table above.

So the model doesn't clearly beat the baseline. It trades a few more misses for many fewer false alarms. Whether that's worth it depends on how an operator weighs a missed threat against a wasted maneuver. F2 counts a miss as the more costly mistake, and by that measure the two are tied. Only 93 dangerous events fall in the group where the risk can still change, about 18 per test split, so telling the two apart would take more data.

## Decisions along the way
- **Which events to label.** The competition only kept events whose final update came within a day of closest approach, so the final risk is trustworthy. Here that rule left just 66 of the 323 dangerous events, because dangerous events tend to stop getting updates earlier than safe ones. I kept every event with data from two or more days out instead. The trade-off is that some final risks come from updates more than a day before closest approach.
- **Checking the result on more than one split.** A single test set has only about 65 dangerous events, so luck can decide the winner. Averaging over five splits showed that the first "win" was luck.

## Next steps
- A sequence model (such as an LSTM) that reads each event's updates in order and learns trends itself, compared with the baseline on the same five splits.
- A closer look at the false alarms that remain: what the events flagged by both the model and the baseline have in common.

## Notebooks
- `01_explore.ipynb`: explores the data. Dataset size, updates per event, how rare dangerous events are, and whether danger shows up early.
- `02_baseline.ipynb`: builds the "assume the risk won't change" forecast and scores it.
- `03_model.ipynb`: labels, the train/test split, the gradient-boosting model with trend features, and the five-split comparison with the baseline.

## How to run
You'll need Python 3.11 or 3.12. The data isn't included in this repository. Download `train_data.zip` from the [Kelvins Collision Avoidance Challenge data page](https://kelvins.esa.int/collision-avoidance-challenge/data/), unzip it, and put `train_data.csv` in a `Data/` folder at the top of the repository. The page also has `test_data.csv`, which isn't used here: it has no final risks, so predictions on it can't be scored.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in `notebooks/` in order. `scripts/build_explorer_data.py` repeats the model notebook's steps and writes the data file for the interactive explorer. `03_model.ipynb` takes a few minutes, most of it spent training the models for the five-split comparison.
