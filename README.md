# MultisourceStackingTransferLearning

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)

MultisourceStackingTransferLearning is a Python project that demonstrates the application of multi-source transfer learning through a stacking ensemble approach. The primary focus is on using data from multiple source domains to improve predictions in a target domain using neural networks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Contact](#contact)

## Introduction

This project aims to explore transfer learning techniques where data from different sources (domains) is leveraged to train models that are then combined with the target domain model. Predictions from models trained on different source domains are combined and used as input features for a final model trained on the target domain.

## Features

- Multi-source transfer learning with stacking ensemble.
- Customizable model parameters.
- Outputs performance metrics such as MAE, RMSE, MSE, R², and adjusted R².
- Supports easy addition of new source domains.

## Installation

### Prerequisites

- Python 3.9+
- [pip](https://pip.pypa.io/en/stable/)
