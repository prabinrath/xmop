# XMoP: Whole-Body Control Policy for Zero-shot Cross-Embodiment Neural Motion Planning
This repository contains the official implementation for XMoP. 

XMoP, ia a novel configuration-space neural motion policy that solves planning problems zero-shot for unseen robotic manipulators which has not been achieved by any prior robot learning algorithm. We formulate C-space control as a link-wise SE(3) pose transformation method, and showcase its scalability for data-driven policy learning. XMoP uses fully synthetic data to train models for motion planning and collision detection while demonstrating strong sim-to-real generalization with a 70% success rate. Our work demonstrates for the first time that C-space behavior cloning policies can be learned without embodiment bias
and that these learned behaviors can be transferred to novel unseen embodiments in a zero-shot manner. This repository contains the implementation, data generation, and evaluation scripts for XMoP.

<div align="center">
  <img src="media/approach.png" alt="approach">
</div>

**Authors**: [Prabin Kumar Rath](https://prabinrath.github.io/), [Nakul Gopalan](https://nakulgopalan.github.io/) 
**Website**: [https://prabinrath.github.io/xmop](https://prabinrath.github.io/xmop)  
**Models**: [https://huggingface.co/prabinrath/xmop](https://huggingface.co/prabinrath/xmop)
**Datasets**: [https://huggingface.co/datasets/prabinrath/xmop](https://huggingface.co/datasets/prabinrath/xmop)

## Real-world Rollouts
All rollouts shown below use XMoP with a fixed set of frozen policy parameters.
<div align="center" style="max-width: 100%;margin: 0; padding: 0;">
  <img src="media/xmop_webpage_banner.gif" alt="realworld" style="width: 100%; height: auto;margin: 0; padding: 0;">
</div>
