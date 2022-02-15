from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean

Base = declarative_base()


class Evaluation(Base):
    __tablename__ = 'evaluation'
    id = Column(Integer, primary_key=True)
    query_name = Column(String)
    dataset_id = Column(String)
    dataset_location = Column(String)
    dataset_univariate = Column(Boolean)
    dataset_labeled = Column(Boolean)
    contamination_train = Column(Float)
    evaluated_by = Column(String)
    evaluation_date = Column(DateTime)
    train_percentage = Column(Float)
    unsupervised = Column(Boolean)
    train_samples = Column(Integer)
    test_samples = Column(Integer)
    true_positives = Column(Integer)
    true_negatives = Column(Integer)
    false_positives = Column(Integer)
    false_negatives = Column(Integer)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    specificity = Column(Float)
    f1_score = Column(Float)
    mcc = Column(Float)

    def __repr__(self):
        return "<Evaluation(query_name='{}', dataset_id='{}', evaluated_by='{}', evaluation_date='{}')>" \
            .format(self.query_name, self.dataset_id, self.evaluated_by, self.evaluation_date)
