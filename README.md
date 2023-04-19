# Exploring Symbolic Deep Learning

**SDL** is a course project of [《Machine learning for physicists》](https://github.com/wangleiphy/ml4p). The goal of this project is to explore the effectiveness of combining traditional **symbolic models** with **deep learning**.  

The content includes sharing literature reading and some code demos, which we will gradually put in the repository.  

Finally, we will initially test our **SDL** on well-known physical models that have established formulas to describe interactions, as well as use our algorithm on physical models that are less certain and require further exploration.

### Contents
*   [Background](#Background)
*   [Example Readmes](#Example-Readmes)
*   [Maintainers](#Maintainers)
*   [Contributing](#Contributing)

## Background
- [Symbolic regression (SR)](https://en.wikipedia.org/wiki/Symbolic_regression) is a type of a supervised machine learning technique that searches the space of mathematical expressions to find the model that best fits a given datase. However, the [genetic algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm) is considered essentially a brute force procedure[[1]](https://www.science.org/doi/10.1126/science.1165893) which scale exponentially with the number of input variables and operators[[2]]().

>Deep learning methods allow efficient training of complex models on highdimensional datasets...

>...train a Networks in a supervised setting, then apply symbolic regression to components of the learned model to extract explicit physical relations.

### Using deep learning to preprocess data for symbolic regression
**e.g.Using GNN to preprocess data**
[Miles Cranmer et al.](https://arxiv.org/abs/2006.11287)

### Using a hybrid optimization algorithms
**e.g.RNN & risk-policy gradient: Using a big model to search the space of small model**     
[Brenden K. Petersen](https://arxiv.org/abs/1912.04871)
**e.g.RNN &GP: Using a big model to search the space of small model**     
[T. Nathan Mundhenk et al.](https://arxiv.org/abs/2111.00053)
**e.g.RNN & units constrains: Using a big model to search the space of small model**    
[Tenachi et al 2023](https://arxiv.org/abs/2303.03192)


## Example Readmes

- We will gradually put in the repository.
  our work is gonna to extend the package of  [Tenachi et al 2023](https://arxiv.org/abs/2303.03192) base on some algorithms of RL. e.g. A2C, Reward shaping...

- We recommend using the following Python packages: [numpy](https://numpy.org/), [scipy](https://scipy.org/), [sklearn](https://scikit-learn.org/stable/index.html), [pandas](https://pandas.pydata.org/), [tensorflow](https://www.tensorflow.org/?hl=zh-cn) and [jax](https://github.com/google/jax).

## Maintainers
@chordc.

## Contributing



