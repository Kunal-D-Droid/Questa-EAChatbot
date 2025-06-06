# Emotion Text Analyzer Chatbot Integration

This project implements an LSTM-based emotion detection model and deploys it as a FastAPI application. It is intended to be integrated with a chatbot or other applications to provide emotion-aware text analysis.

## Project Structure

```
.
├── emotion_model/
│   ├── train_model.py
│   ├── emotion_model.h5 (generated after training)
│   ├── data_preparation.py
│   ├── emotions.json
│   ├── emotion_mapping.json
│   └── tokenizer.json
├── main.py             # FastAPI application entry point
├── requirements.txt    # Project dependencies
└── emotion_env/        # Local Python virtual environment
```

## Setup Instructions

1.  **Clone the repository** (if applicable):
    ```bash
    # Assuming your project is in a git repository
    git clone <repository_url>
    cd <project_directory>
    ```
    *(Skip this step if you are working directly in your project folder)*

2.  **Create and activate a Python virtual environment**:
    ```bash
    python3 -m venv emotion_env
    source emotion_env/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare the dataset** (if necessary, using `data_preparation.py` if applicable) and **train the emotion detection model**:
    ```bash
    # Assuming train_model.py handles data loading and training
    python emotion_model/train_model.py
    ```
    *(This step generates `emotion_model.h5`, `tokenizer.json`, `emotions.json`, and `emotion_mapping.json`)*

5.  **Run the FastAPI application locally** (for testing):
    ```bash
    # Make sure your virtual environment is activated
    python emotion_api/uvicorn main:app --reload 
    ```
    *(Access the local API docs at `http://localhost:8000/docs`)*

## API Usage

The emotion detection API provides the following endpoints:

-   `POST /predict`: Predict emotion from text
    ```bash
    curl -X POST "http://localhost:8000/docs" \\
         -H "Content-Type: application/json" \\
         -d '{"text": "I am very happy today!"}'
    ```

## Deployment to AWS EC2

This section outlines the manual steps we followed to deploy the FastAPI application to an Ubuntu EC2 instance.

1.  **Launch an EC2 instance**:
    *   Choose an Ubuntu Server LTS AMI.
    *   Select an appropriate instance type (e.g., t2.micro for free tier).
    *   Create a new key pair (e.g., `emotion-analyzer-key.pem`) and download the `.pem` file securely.

2.  **Configure Security Group**:
    *   Ensure the security group associated with your instance allows inbound traffic on:
        *   Port 22 (SSH) from your IP or a known range.
        *   Port 8000 (Custom TCP Rule) from `Anywhere (0.0.0.0/0)` or a specific IP range for API access.

3.  **Connect to the instance via SSH**:
    ```bash
    ssh -i "path/to/your/emotion-analyzer-key.pem" ubuntu@your-instance-public-ip
    ```
    *(Replace placeholders with your key path and instance IP)*

4.  **Update the system and install necessary packages**:
    ```bash
    sudo apt update
    sudo apt upgrade -y
    # Install python3-venv if not already present
    sudo apt install python3-venv -y
    ```

5.  **Create project directory, virtual environment, and activate**:
    ```bash
    mkdir -p ~/emotion-analyzer
    cd ~/emotion-analyzer
    python3 -m venv chatbot_venv # Using chatbot_venv as used in our steps
    source chatbot_venv/bin/activate
    ```

6.  **Transfer project files from your local machine**:
    *   Exit the SSH session (`exit`).
    *   On your local machine, navigate to your project directory (`C:\Users\dasku\OneDrive\Desktop\Questa-EAchatbot`).
    *   Create a zip archive of the required files:
        ```bash
        zip -r emotion-analyzer.zip emotion_model/ main.py requirements.txt
        ```
        *(Adjust files to include if your structure differs)*
    *   Transfer the zip file to the server:
        ```bash
        scp -i "path/to/your/emotion-analyzer-key.pem" emotion-analyzer.zip ubuntu@your-instance-public-ip:~/emotion-analyzer/
        ```
    *   SSH back into the server and unzip:
        ```bash
        ssh -i "path/to/your/emotion-analyzer-key.pem" ubuntu@your-instance-public-ip
        cd ~/emotion-analyzer
        unzip emotion-analyzer.zip
        ```

7.  **Install Python dependencies on the server**:
    ```bash
    # Make sure you are in the ~/emotion-analyzer directory and virtual env is active
    source chatbot_venv/bin/activate
    pip install -r requirements.txt
    ```

8.  **Create and configure a systemd service**:
    ```bash
    sudo nano /etc/systemd/system/emotion-analyzer.service
    ```
    *   Paste the following content (using `uvicorn main:app` as `main.py` is in the root and `chatbot_venv`):
        ```ini
        [Unit]
        Description=Emotion Analyzer API
        After=network.target

        [Service]
        User=ubuntu
        WorkingDirectory=/home/ubuntu/emotion-analyzer
        Environment="PATH=/home/ubuntu/emotion-analyzer/chatbot_venv/bin"
        ExecStart=/home/ubuntu/emotion-analyzer/chatbot_venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
        Restart=always

        [Install]
        WantedBy=multi-user.target
        ```
    *   Save and exit (`Ctrl + X`, `Y`, `Enter`).

9.  **Reload systemd, enable, and start the service**:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable emotion-analyzer
    sudo systemctl start emotion-analyzer
    ```

10. **Check the service status and access the API**:
    ```bash
    sudo systemctl status emotion-analyzer
    ```
    *   Access the API docs in a browser: `http://your-instance-public-ip:8000/docs`

## Integration with Chatbot

The emotion detection system can be integrated with any chatbot by:

1. Sending user messages to the emotion API
2. Storing emotion predictions
3. Adjusting responses based on detected emotions
4. Logging conversation emotions for analysis

## Model Details

- Architecture: LSTM-based neural network
- Emotions: joy, sadness, anger, fear, surprise, neutral
- Input: Text sequences (max length: 100 words)
- Output: Emotion probabilities

## Contributing

Feel free to submit issues and enhancement requests! 