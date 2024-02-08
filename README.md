# CoLoRA: Continuous low-rank adaptation
CoLoRA: Continuous low-rank adaptation for reduced implicit neural modeling of parameterized partial differential equations

## Setup

First locally install the colora package with

```bash
pip install --editable .
```

Then install jax with the appropriate CPU or GPU support: [here](https://github.com/google/jax#installation)

Install all additional required packages run:

```bash
 pip install -r requirements.txt
```

Then you should be able to run the included notebooks:

- vlasov.ipynb
- burgers.ipynb
- rde.ipynb



## Reduced models discovered via CoLoRA
This work introduces reduced models based on Continuous Low Rank Adaptation (CoLoRA) that pre-train neural networks for a given partial differential equation and then continuously adapt low-rank weights in time to rapidly predict the evolution of solution fields at new physics parameters and new ini- tial conditions. The adaptation can be either purely data-driven or via an equation-driven variational approach that provides Galerkin-optimal approximations. Because CoLoRA approximates solution fields locally in time, the rank of the weights can be kept small, which means that only few training trajectories are required offline so that CoLoRA is well suited for data-scarce regimes. Predictions with CoLoRA are orders of magnitude faster than with classical methods and their accuracy and parameter efficiency is higher compared to other neural network approaches.

## Collisionless charged particles in electric field (Vlasov's Equation)
The Vlasov equation describes the motion of collision- less charged particles under the influence of an electric
field:

<span>
<img src="./img/vlasov.gif" width="400" height="320" />
<img src="./img/vlasov_dynamics.gif" width="400" height="320" />
</span>

<br>

## Burgers’ equation in 2D
Burgers’ equations give a simplified model of fuild mechanics and can form sharp advecting fronts which are difficult for traditional methods:

<span>
<img src="./img/burgers.gif" width="400" height="320" />
<img src="./img/burgers_dynamics.gif" width="400" height="320" />
</span>


## Rotating denotation waves 
We consider a model of rotating detonation waves, which is motivated by space propulsion with rotating detonation engines (RDE)

<span>
<img src="./img/rde.gif" width="400" height="320" />
<img src="./img/rde_dynamics.gif" width="400" height="320" />
</span>


## CoLoRA vs LoRA

LoRA fine-tunes networks to downstream tasks by adapting low-rank matrices AB. Our CoLoRA introduces a scaling α(t,μ) on the low-rank matrix AB to adapt networks continuously to predict PDE solution trajectories.

![Manifold Cartoon](./img/colora_mani.png)

## CoLoRA architecture

The CoLoRA architecture uses a hyper-network h to generate a set of continuous parameters α which are used to scale low rank matrices AiBi which are internal to the reduced order model uˆ. The parameters of ψ and θ are then jointly optimized to fit data from the full order model uF.

![Architecture Cartoon](./img/colora_arch.png)