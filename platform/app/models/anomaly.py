from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Boolean, String, DateTime

Base = declarative_base()


class Anomaly(Base):
    __tablename__ = 'anomalies'
    id = Column(Integer, primary_key=True)
    query_name = Column(String)
    datasource_id = Column(String)
    detected_by = Column(String)
    detection_date = Column(DateTime)
    start_investigated_datetime = Column(DateTime)
    end_investigated_datetime = Column(DateTime)
    first_anomaly_datetime = Column(DateTime)
    last_anomaly_datetime = Column(DateTime)
    valid = Column(Boolean)  # Is true if user accepts this as a valid anomaly
    checked = Column(Boolean)  # Is true if user has checked the Anomaly for validity


    def __repr__(self):
        return "<Anomaly(query_name='{}', detection_date='{}', verified={})>" \
            .format(self.query_name, self.detection_date, self.verified)
