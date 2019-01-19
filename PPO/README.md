This is a PPO application on Super Mario Bros with the addition of the Intrinsic Curiosity Module.

For training the PPO model without curiosity 
```bash 
# Works only with Python 3
python3 main.py --reward_type <dense or sparse> --use_curiosity 0
``` 
For training the A3C model with curiosity 
```bash 
# Works only with Python 3
python3 main.py --reward_type <dense or sparse> --use_curiosity 1
``` 
For the dense reward setting, with 12 processes and 1 Nvidia GPU, the algorithms converges in 6 hours for the first level.

We just experimented with PPO and we encourage you to use the A3C which is more resource friendly.


The PPO code is based on : https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

