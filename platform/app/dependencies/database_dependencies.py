from app.dependencies.config_dependencies import load_config
from app.models import anomaly
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def load_session():

    """Loads the database session from the config map

    :return: returns the database session object

    """

    # Create database engine
    config = load_config()

    db = config["management_database"]
    database_uri = 'postgresql+psycopg2://' + db["username"] + ":" + db["password"] + "@" + db["host"] + ":" + db[
        "port"] + "/" + db["db_name"]
    engine = create_engine(database_uri)
    # Migration
    anomaly.Base.metadata.create_all(engine)

    # Create database session
    session = sessionmaker(bind=engine)
    s = session()

    return s
