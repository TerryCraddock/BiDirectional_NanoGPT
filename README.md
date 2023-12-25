# BiDirectionalGPT: Efficient and Stable Bidirectional Dual-Head Causal Attention Transformers

This repository contains a modified version of Andrew Karpathy's NanoGPT codebase, introducing significant enhancements to the GPT language model architecture. Our model extends language modeling capabilities by introducing bidirectional dual-head causal attention mechanisms. The aim is to improve context understanding and predictive capabilities while achieving performance comparable to or better than larger GPT-2 models in reduced training time.

## Acknowledgment

The foundation of this project is built upon Andrew Karpathy's NanoGPT repository ([karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)), which provided the groundwork for our modifications and improvements.

## Introduction

We introduce a novel bidirectional variant of the GPT language model, incorporating dual sets of attention mechanisms to process sequences in both forward and backward directions. This approach aims to enhance the model's understanding of context and improve its predictive capabilities.

## Model Architecture

Our core modification lies in the `BiDirectionalGPT` class, an extension of the original NanoGPT architecture. This enhanced model incorporates parallel forward and backward attention mechanisms within each transformer block. These dual attention heads capture dependencies from both directions, providing a more comprehensive contextual understanding.

The bidirectional attention is realized through the `ForwardCausalSelfAttention` and `BackwardCausalSelfAttention` classes, preserving causality by masking future information in the backward attention.

## Implementation Details

The model's configurability is facilitated by the `GPTConfig` data class, allowing easy adjustments to layers, heads, embedding dimensions, and other crucial hyperparameters. Model stability and prevention of overfitting are ensured by employing layer normalization, dropout, and specific weight initialization techniques inspired by the GPT-2 paper.

## Efficiency and Comparative Performance

In training, our model showcases accelerated convergence compared to larger GPT-2 variants. At step 57,600, our model achieved a train loss of 3.3869 and a validation loss of 3.4204, corresponding to train and validation perplexities of 29.5753 and 30.5811, respectively. These results, obtained in significantly fewer training steps, rival or surpass those of larger GPT-2 models.

## Discussion

The bidirectional architecture of our model grants a more nuanced understanding of language, particularly beneficial for tasks requiring context from both directions. We discuss the implications of this architecture and its potential applications across various natural languag

