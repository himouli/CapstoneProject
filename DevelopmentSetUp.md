# Development Setup
## Prerequisites

    Python 3.8+
    Git
    (Optional) Docker and Docker Compose

# Local Development Setup
## Clone the repository:
git clone https://github.com/yourusername/openstack-rca.git
cd openstack-rca

## Set up the development environment:
make setup
Run tests to verify the setup:
make test
Using Docker for Development
Build the development container:
docker build -t openstack-rca-dev -f docker/Dockerfile.dev .
Run the development container:
docker run -it -v $(pwd):/app -p 8501:8501 openstack-rca-dev
Using the Log Analyzer
Processing Logs
Process a log file:
make analyze logfile=path/to/your/logfile.log

## Running the Web Interface
Start the Streamlit web interface:
streamlit run src/app.py
Then open your browser at http://localhost:8501

# Project Structure
![image](https://github.com/user-attachments/assets/0f718a28-2ef4-4b44-b248-8d71952e7b63)

# CI/CD Pipeline
This project uses GitHub Actions for continuous integration and deployment:

    CI Pipeline: Runs tests, linting, and code quality checks on every push and pull request
    CD Pipeline: Builds and pushes Docker images for tagged releases

# Contributing 

    Fork the repository
    Create a feature branch (git checkout -b feature/amazing-feature)
    Commit your changes (git commit -m 'Add amazing feature')
    Push to the branch (git push origin feature/amazing-feature)
    Open a Pull Request

Please make sure your code passes all tests and lint checks before submitting a pull request.
