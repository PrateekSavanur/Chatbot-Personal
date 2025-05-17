# Event Contract

This is a Solidity contract that allows users to create events, buy tickets, and transfer tickets to other addresses.

## Contract Details

The contract contains the following main features:

- Event Creation: Users can create events by providing the event name, date, ticket price, and the number of available tickets.

- Ticket Purchase: Users can buy tickets for a specific event by providing the event ID and the desired quantity of tickets. The payment should be made in Ether, and the contract ensures that the event is not expired and enough tickets are available.

- Ticket Transfer: Users can transfer their purchased tickets to other addresses. The contract verifies the ownership of the tickets and the availability of the requested quantity before performing the transfer.

## Getting Started

To use this contract, you need to have the following:

- A compatible Ethereum wallet (e.g., MetaMask) to interact with the contract on the Ethereum network.
- Sufficient Ether balance to purchase tickets.

## Usage

1. Deploy the Contract: Deploy the EventContract.sol Solidity contract on an Ethereum network of your choice (e.g., Ethereum Mainnet, Rinkeby, etc.).

2. Interact with the Contract: Once the contract is deployed, you can interact with it using an Ethereum wallet application, such as MetaMask. Connect your wallet to the deployed contract address.

3. Create an Event: Use the `createEvent` function to create an event. Provide the event name, date, ticket price, and the number of available tickets. Make sure the event date is in the future.

4. Buy Tickets: Use the `buyTicket` function to purchase tickets for a specific event. Provide the event ID and the quantity of tickets you want to purchase. Make sure to send the correct amount of Ether with the transaction.

5. Transfer Tickets: Use the `transferTicket` function to transfer your purchased tickets to another address. Provide the event ID, the quantity of tickets to transfer, and the recipient's address. Ensure you have enough tickets to transfer.

