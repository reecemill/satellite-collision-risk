# Satellite Collision Risk Prediction

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

## Next steps
A model can only beat the baseline by catching some of the 23 missed threats or cutting down the 269 false alarms. Next, I'll look at those two groups directly: how do missed threats and false alarms differ from the events the baseline got right? Whatever separates them becomes a feature for the model. Then I'll train a model and compare it against the baseline's F2 of 0.81.

## Notebooks
- `01_explore.ipynb`: explores the data. Dataset size, updates per event, how rare dangerous events are, and whether danger shows up early.
- `02_baseline.ipynb`: builds the "assume the risk won't change" forecast and scores it.
