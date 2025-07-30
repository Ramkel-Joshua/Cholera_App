from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Ramkel Gabriel-Developed Cholera App"}
