# For SSD Failure prediction in noise label

Modify to apply Time-Series dataset(SSD SMART log).

(24/4/5~)

[Original code](https://github.com/MacLLL/SELC)

# SELC: Self-Ensemble Label Correction Improves Learning with Noisy Labels
Code for IJCAI2022 [SELC: Self-Ensemble Label Correction Improves Learning with Noisy Labels](https://www.ijcai.org/proceedings/2022/455), SELC is a label correction method, it will automatically correct the noisy labels in training set. 


## Requirements
- Python 3.9.18
- Pytorch 2.1.1 


## Usage
```run
python3 train_SSD_with_SELC.py
```
I modify dataloader and SELCLoss.
