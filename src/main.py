import os
from dotenv import load_dotenv
from src.api.endpoints import app
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger()
import nltk
nltk.download('punkt_tab')

def main():
    app.run("0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()
