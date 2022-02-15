
import logging

import uvicorn
from app.dependencies.config_dependencies import load_config
from fastapi import FastAPI
from app.routes import administration_routes, prediction_routes, training_routes, evaluation_routes
from prometheus_client import start_http_server
from routes import statistical_routes
from sqlalchemy import create_engine
from app.models import anomaly, evaluation

logging_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
logging.basicConfig(format=logging_format, level=logging.INFO)

app = FastAPI()
app.include_router(administration_routes.router)
app.include_router(prediction_routes.router)
app.include_router(training_routes.router)
app.include_router(evaluation_routes.router)
app.include_router(statistical_routes.router)

config = load_config()

db = config["management_database"]
DATABASE_URI = 'postgresql+psycopg2://' + db["username"] + ":" + db["password"] + "@" + db["host"] + ":" + db["port"] \
    + "/" + db["db_name"]

# Create database engine
try:
    engine = create_engine(DATABASE_URI)
    # Migration
    anomaly.Base.metadata.create_all(engine)
    evaluation.Base.metadata.create_all(engine)
except:
    logging.warning("Failed to establish connection to the database and migrate the schemas")
    logging.warning("FADIP application will run anyways, but the evaluation and the training results won't be stored.")


if __name__ == "__main__":
    start_http_server(8000)
    uvicorn.run(app, host="0.0.0.0", port=80, log_config=logging.basicConfig(format=logging_format,
                                                                             level=logging.INFO))
