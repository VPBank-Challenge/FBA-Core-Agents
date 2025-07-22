from dotenv import load_dotenv
from src.api.endpoints import app

import uvicorn

load_dotenv()



def main():
    uvicorn.run(
        "src.api.endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )


if __name__ == "__main__":
    main()