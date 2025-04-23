# Dracofit - Your Gamified Fitness Companion with AI Chat

Dracofit is a full-stack web application designed to make fitness engaging and personalized. It combines workout planning, tracking, scheduling, and gamification features with an integrated AI chatbot (DracoBot) to assist users on their fitness journey.

## Features

*   **User Authentication:** Secure user registration, login, email verification, and password management using JWT.
*   **Profile Management:** Users can set up and update personal details (gender, birthdate, weight, height) and fitness preferences (goals, level, frequency).
*   **Exercise Library:** Browse and view details of various exercises, including descriptions, target muscles, difficulty, and video links.
*   **Workout Plan Creation:** Create custom workout plans by selecting exercises, defining sets, reps, duration, and rest times.
*   **Workout Logging:** Log completed workout sessions, tracking duration and exercises performed.
*   **Exercise Logging:** Detailed logging for each exercise within a workout, including weight, reps per set, and automatic personal record tracking.
*   **Weekly Scheduling:** Plan workouts for specific days of the week using a visual schedule editor.
*   **Gamification (DracoPet):** A virtual pet (dragon) that levels up and evolves based on user activity (XP earned from workouts, maintaining streaks).
*   **AI Chatbot (DracoBot):** An integrated chatbot powered by a local Gemma model to answer fitness-related questions and provide assistance. Chat history is saved per user.
*   **Responsive UI:** Frontend built with React and Tailwind CSS for a modern and responsive user experience.

## Technology Stack

*   **Frontend (`dracofit-frontend`):**
    *   React (Vite)
    *   Tailwind CSS
    *   Axios (for API calls)
    *   React Router
    *   Framer Motion (for animations)
    *   React Markdown
*   **Backend (`dracofit-backend`):**
    *   NestJS (Node.js framework)
    *   TypeScript
    *   TypeORM (ORM for database interaction)
    *   PostgreSQL (or other SQL database compatible with TypeORM)
    *   JWT (for authentication)
    *   Class-Validator / Class-Transformer
*   **Chatbot Backend (`dracofit-chatbot`):**
    *   Python
    *   Flask (Web framework)
    *   Hugging Face Transformers (for loading and running the AI model)
    *   PyTorch
    *   BitsAndBytes (for model quantization)
    *   Gunicorn (recommended for production deployment)

## Project Structure

```
dracofit/
├── dracofit-backend/     # NestJS REST API backend
│   ├── src/              # Source code (modules, controllers, services, entities)
│   ├── seed/             # Database seeding scripts
│   ├── .env.example      # Example environment variables
│   └── ...
├── dracofit-chatbot/     # Python Flask AI Chatbot backend
│   ├── api.py            # Flask API endpoints (/chat, /health)
│   ├── model.py          # Handles loading and interacting with the Gemma model
│   ├── config.py         # Chatbot configuration (model name, prompts, etc.)
│   ├── requirements.txt  # Python dependencies
│   └── ...
├── dracofit-frontend/    # React Vite frontend application
│   ├── public/           # Static assets (images, etc.)
│   ├── src/              # Source code (components, pages, services, hooks, contexts)
│   ├── .env.example      # Example environment variables
│   └── ...
└── README.md             # This file
```

## Setup and Installation

**Prerequisites:**

*   Node.js and npm (or yarn)
*   Python 3.x and pip
*   A running PostgreSQL instance (or other SQL database)
*   Git

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd dracofit
    ```

2.  **Setup Backend (`dracofit-backend`):**
    *   Navigate to the backend directory: `cd dracofit-backend`
    *   Install dependencies: `npm install`
    *   Create a `.env` file based on `.env.example` and configure database connection details, JWT secret, etc.
    *   Run database migrations (if using TypeORM migrations): `npm run migration:run`
    *   (Optional) Seed the database: `npm run seed:exercises`, `npm run seed:workouts`

3.  **Setup Chatbot (`dracofit-chatbot`):**
    *   Navigate to the chatbot directory: `cd ../dracofit-chatbot`
    *   Create a virtual environment (recommended): `python -m venv venv` and activate it (`source venv/bin/activate` or `.\venv\Scripts\activate`)
    *   Install dependencies: `pip install -r requirements.txt`
    *   Configure `config.py` with your Hugging Face token (`HF_TOKEN`) and desired model settings. Ensure you have accepted the terms for the Gemma model on Hugging Face.
    *   Ensure you have necessary GPU drivers (like CUDA) if using GPU acceleration.

4.  **Setup Frontend (`dracofit-frontend`):**
    *   Navigate to the frontend directory: `cd ../dracofit-frontend`
    *   Install dependencies: `npm install`
    *   Create a `.env` file based on `.env.example` and set `VITE_API_URL` to point to your running `dracofit-backend` instance (e.g., `http://localhost:3000/api`).

## Running the Application

You need to run all three parts concurrently.

1.  **Run Backend (`dracofit-backend`):**
    *   In the `dracofit-backend` directory:
    ```bash
    npm run start:dev
    ```
    *   The API server should typically start on `http://localhost:3000`.

2.  **Run Chatbot (`dracofit-chatbot`):**
    *   In the `dracofit-chatbot` directory (with the virtual environment activated):
    ```bash
    # For development (Flask's built-in server)
    python api.py
    # Or using Gunicorn (recommended for better performance)
    # gunicorn --workers 1 --threads 4 --bind 0.0.0.0:5000 api:app
    ```
    *   The chatbot server should typically start on `http://localhost:5000`.

3.  **Run Frontend (`dracofit-frontend`):**
    *   In the `dracofit-frontend` directory:
    ```bash
    npm run dev
    ```
    *   The frontend development server should typically start on `http://localhost:5173` (or another port if 5173 is busy). Open this URL in your browser.

## Key Configuration

*   **`dracofit-backend/.env`**: Database connection, JWT secrets, port, etc.
*   **`dracofit-chatbot/config.py`**: Hugging Face token, model name, system prompt, generation parameters, GPU settings.
*   **`dracofit-frontend/.env`**: URL of the `dracofit-backend` API (`VITE_API_URL`).


