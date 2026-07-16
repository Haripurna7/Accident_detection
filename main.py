import uvicorn
import os
from webapp.app import app

def main():
    print("Starting  - Aegis Urban Response & Analysis Web Server...")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("webapp.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
