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



## Reduced models discovered via CoLoRA
<span>
<img src="./img/vlasov.gif" width="400" height="350" />
<img src="./img/vlasov_dynamics.gif" width="400" height="350" />
</span>

<span>
<img src="./img/burgers.gif" width="400" height="350" />
<img src="./img/burgers_dynamics.gif" width="400" height="350" />
</span>

## CoLoRA vs LoRA
![Manifold Cartoon](./img/colora_mani.png)

## CoLoRA architecture
![Architecture Cartoon](./img/colora_arch.png)