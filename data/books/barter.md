Here is the updated content in .md format for your Barter System project:

````markdown
# Swapify - One swap at a time.

A backend for a Barter System application where users can trade items directly with each other, keeping track of transactions, messaging, and item listings.

## Features

- **Item Listings**: Users can list items they want to barter.
- **Transactions**: Initiate, accept, reject, or complete barter transactions.
- **Messaging**: Users can send messages during transactions.
- **Authentication**: User registration and login using JWT.
- **Security**: Protection against XSS, NoSQL injection, and HTTP parameter pollution.

## Tech Stack

- **Backend**: Node.js, Express.js
- **Database**: MongoDB (MongoDB Atlas)
- **Authentication**: JSON Web Token (JWT)
- **Other Dependencies**:
  - **mongoose** for MongoDB ORM
  - **express-rate-limit** for rate-limiting requests
  - **helmet** for setting security headers
  - **express-mongo-sanitize** for NoSQL injection protection
  - **xss-clean** for XSS protection
  - **bcryptjs** for hashing passwords
  - **dotenv** for environment variables

## Installation

### Prerequisites

Make sure you have [Node.js](https://nodejs.org/) and [npm](https://www.npmjs.com/) installed.

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/barter-system.git
   cd barter-system
   ```
````

2. Install dependencies:

   ```bash
   npm install
   ```

3. Create a `.env` file in the root directory and add the following environment variables:

   ```
   MONGODB_URI=your-mongodb-connection-string
   JWT_SECRET=your-secret-key
   JWT_EXPIRES_IN=1d
   NODE_ENV=development
   ```

4. Run the app in development mode:

   ```bash
   npm run start
   ```

   The app will be running on `http://localhost:8000`.

### Running in Production

1. Set the `NODE_ENV` to production:

   ```bash
   npm run start:prod
   ```

2. Ensure you have your MongoDB cluster set up with the correct IP whitelisting and access control.

## API Endpoints

### 1. **Authentication**

- **POST** `/api/v1/users/signup`: User sign up (register).
- **POST** `/api/v1/users/login`: User login (JWT-based).
- **POST** `/api/v1/users/forgotPassword`: Sends email to reset password using nodemailer.
- **POST** `/api/v1/users/resetPassword/:token`: Link sent to user via email.

### 2. **Items**

- **GET** `/api/v1/items`: Get all items.
- **POST** `/api/v1/items`: Create a new item.
- **GET** `/api/v1/items/:id`: Get details of an item by ID.
- **PUT** `/api/v1/items/:id`: Update an item.
- **DELETE** `/api/v1/items/:id`: Delete an item.

### 3. **Transactions**

- **POST** `/api/v1/transactions`: Create a new transaction.
- **GET** `/api/v1/transactions`: Get all transactions.
- **GET** `/api/v1/transactions/:id`: Get details of a specific transaction.
- **PATCH** `/api/v1/transactions/:id/status`: Update transaction status (e.g., Accept, Reject, Complete).
- **POST** `/api/v1/transactions/:id/messages`: Send a message in the transaction.

## Deployment

The application is deployed on [Vercel](https://vercel.com/). You can visit the live API at:

- **Live API URL**: [https://barter-backend-five.vercel.app/](https://barter-backend-five.vercel.app/)
- **API Publication**: [https://documenter.getpostman.com/view/31551887/2sAYQanBmZ](https://documenter.getpostman.com/view/31551887/2sAYQanBmZ)

## MongoDB Setup

1. Create a MongoDB cluster on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Get your connection string and update it in the `.env` file.
3. Add your IP address to the MongoDB Atlas IP whitelist.

## Environment Variables

The application requires the following environment variables:

- `MONGODB_URI`: MongoDB Atlas connection string.
- `JWT_SECRET`: Secret key for signing JWT tokens.
- `JWT_EXPIRES_IN`: Expiration time for JWT tokens.
- `NODE_ENV`: The environment the app is running in (e.g., `development`, `production`).

## Contributing

Feel free to fork this repository, make changes, and create pull requests. Contributions are welcome!

## License

MIT License

---

Created by [Prateek Prasanna Savanur](https://prateeksavanur.xyz).

```

This document can now be used for the project `Swapify` with all necessary details, including installation, API endpoints, and other project information.
```
