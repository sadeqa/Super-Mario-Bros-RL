## Usage 
For training the A3C model without curiosity 
```bash 
# Works only with Python 3
python3 train-mario.py --reward_type <dense or sparse>
``` 
For training the A3C model with curiosity 
```bash 
# Works only with Python 3
python3 train-mario-curiosity.py --reward_type <dense or sparse>
``` 
For the dense reward setting, with 24 processes the algorithms converges in 2 hours for the first level.








This repository is heavily based on : https://github.com/ikostrikov/pytorch-a3c
