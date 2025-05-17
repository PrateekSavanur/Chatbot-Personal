# Crowdfunding Smart Contract

## Overview

This repository contains a set of Solidity smart contracts for a decentralized crowdfunding platform. The platform allows project creators to raise funds for their projects and distribute ERC20 tokens to contributors. The contracts included are:

1. **Crowdfunding**: Manages the creation, funding, and token distribution of crowdfunding projects.
2. **RewardToken**: ERC20 token that is used as a reward for project contributors.
3. **TokenFactory**: Factory contract to create new instances of `RewardToken`.

## Contracts

### Crowdfunding

The `Crowdfunding` contract is the main contract for managing projects, contributions, and token distribution.

#### Features

- **Create Project**: Project owners can create a new crowdfunding project with a funding goal, duration, and reward token details.
- **Contribute Funds**: Users can contribute ETH to projects they support.
- **Close Project**: Automatically closes a project when its deadline is reached or the funding goal is met.
- **Withdraw Funds**: Project owners can withdraw the funds raised once the project is closed.
- **Start ICO**: Initiates the token distribution process for a project.
- **Distribute Tokens**: Distributes reward tokens to contributors based on their contributions.
- **Withdraw Leftover Tokens**: Allows project owners to withdraw any leftover tokens after the ICO is completed.

### RewardToken

The `RewardToken` contract is an ERC20 token used to reward contributors of a crowdfunding project.

#### Constructor Parameters

- `_name`: The name of the token.
- `_symbol`: The symbol of the token.
- `_initialAmount`: The initial supply of tokens.
- `_owner`: The address that will own the initial supply.

### TokenFactory

The `TokenFactory` contract is used to create new instances of `RewardToken`.

#### Features

- **Create Token**: Creates a new `RewardToken` and assigns the initial supply to the specified owner.

#### Events

- `TokenCreated(address indexed projectOwner, address indexed tokenContract)`: Emitted when a new token is created.

## Deployment

1. Deploy the `TokenFactory` contract.
2. Deploy the `Crowdfunding` contract with the address of the deployed `TokenFactory` contract.

## Usage

### Creating a Project

Call the `createProject` function with the following parameters:

- `_owner`: The address of the project owner.
- `_goalAmount`: The funding goal in wei.
- `_durationDays`: The duration of the crowdfunding campaign in days.
- `_tokenName`: The name of the reward token.
- `_tokenTicker`: The symbol of the reward token.
- `_tokenPerEth`: The amount of tokens per ETH contributed.

### Contributing to a Project

Send ETH to the `contributeFunds` function with the `_projectId` of the project you wish to support.

### Withdrawing Funds

Once a project is closed, the project owner can call the `withdrawFunds` function with the `_projectId` to withdraw the funds raised.

### Starting and Completing ICO

The project owner can call `startICO` to initiate the token distribution process. Once the ICO is started, the `distributeTokens` function can be called to distribute tokens to all contributors.
