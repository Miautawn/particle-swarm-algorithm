# Particle Swarm Algorithm
Swarm Optimisation Algotithm (PSO) - is a stochastic optimisation algorithm which relies on many independent actors (or particles) to find the optimisation goal. Each particle moves acording to it's current, personal best location and global best location directions to iteratively converge.

## Mathematical interpretation
Each particle has its position vector $X$ and velocity vector $V$. As they go, they track their personal best location $P$ and global best location $G$ across all particles. To controll the tradeoff between the exploitation and exploration, we have 3 parameters:
- Inertia weight **w** which controlls the importance of the current particle direction. This weight should linearly decrese.
- cognitive component **c1** which controlls the importance of personal best location direction
- social component **c2** which controlls the importance of global best location direction  

Each iteration, particle position gets updated: 
$$X_{t+1} = X_t  +  V_{t+1}$$
$$V_{t+1} = wV_t + c1 (P_t - X_t) + c2 (G_t - X_t)$$  

We can make the algorithm more stochastic by multiplying each member in the formula above with random number **[0, 1]**.  
If we leave the simulation for some iterations, we can observe such particle behaviour:

<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/24988290/169686446-c74c8be1-7c82-4929-a1e0-4e6dfbaa1f01.gif">
</p>

## How to run
**NOTE:** This project uses [Poetry](https://python-poetry.org/) as the package manager! If you haven't tried it we highly recommend it!  

1.) Pull the repository  
2.) Install the packages via poetry  
3.) Go to `src` directory and run `main.py` with Python  
4.) If you have configured GUI, you should see the rendered simulation animation 
```bash
>> cd particle-swarm-algotithm
>> poetry install
>> cd src/
>> python main.py
```
P.S. If you can't view the animation, there is also a standalone jupyter notebook!

